"""What was known about one event at a decision time, built from stored snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from odds_core.as_of import (
    EXCHANGE_AT_T_KINDS,
    MAX_EXCHANGE_AGE,
    RETAIL_KINDS,
    UPPER_BOUND_TIME_KINDS,
)
from odds_core.betfair_close import (
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
    is_in_play,
)
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.market_prices import Outcome, book_prices, market_outcomes, outcome_names
from odds_core.models import Event, OddsSnapshot
from odds_core.snapshot_sources import SnapshotKind, snapshot_kind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.constants import SPORT_CONFIG

# The slowest OddsPortal cadence (2 h at 24 h out, ``odds_lambda.jobs.fetch_oddsportal.CADENCE``).
MAX_RETAIL_AGE = timedelta(hours=2)
# The loader's lookback: no quote or exchange price older than this can enter a state.
SNAPSHOT_LOOKBACK = max(MAX_RETAIL_AGE, MAX_EXCHANGE_AGE)


@dataclass(frozen=True)
class RetailQuote:
    """One UK retail book's decimal prices from one snapshot at or before t."""

    time: datetime
    prices: dict[Outcome, float]


@dataclass(frozen=True)
class MarketState:
    """One event at ``decision_time``, from snapshots stamped at or before it."""

    event_id: str
    sport_key: str
    market: str
    kickoff: datetime
    decision_time: datetime
    retail: dict[str, RetailQuote]
    exchange: ExchangePrice | None
    # Ids of every snapshot the state was built from, so a decision can record its inputs.
    snapshot_ids: tuple[int, ...] = ()


def sport_market(sport_key: str) -> str:
    for key, config in SPORT_CONFIG.items():
        if key == sport_key:
            return config.market_key
    raise ValueError(f"No candidate market defined for sport {sport_key!r}")


def build_market_state(
    event: Event,
    snapshots: Sequence[OddsSnapshot],
    decision_time: datetime,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> MarketState:
    """The event's state at ``decision_time``; snapshots stamped after it are ignored.

    Retail: for each retail snapshot kind, only its latest snapshot at or before t, if
    within ``MAX_RETAIL_AGE``, counts, so a book absent from it is absent from the state.
    A book quoted by two kinds takes the more recent snapshot's price. The exchange price
    is the latest complete, not in-play live Betfair price within ``MAX_EXCHANGE_AGE``
    whose back overround is at most ``max_exchange_overround``.

    FDUK opening snapshots (``UPPER_BOUND_TIME_KINDS``) are excluded: their stamp is an
    upper bound on the price's time, which a live run at t never sees. This differs from
    ``odds_analytics.retail_ev.build_event_rows``, which takes the exchange baseline of
    an FDUK opening row from that row's own snapshot and takes retail prices per book
    rather than per whole snapshot.
    """
    market = sport_market(event.sport_key)
    outcomes = market_outcomes(market)
    names = outcome_names(event, market)
    known = sorted(
        (s for s in snapshots if s.snapshot_time <= decision_time),
        key=lambda s: (s.snapshot_time, s.id or 0),
    )

    latest_retail: dict[SnapshotKind, OddsSnapshot] = {}
    exchange: ExchangePrice | None = None
    for snapshot in known:
        kind = snapshot_kind(snapshot.raw_data)
        if kind in UPPER_BOUND_TIME_KINDS:
            continue
        if kind in RETAIL_KINDS:
            latest_retail[kind] = snapshot
        if kind not in EXCHANGE_AT_T_KINDS:
            continue
        if kind is SnapshotKind.BETFAIR_LIVE and is_in_play(snapshot.raw_data):
            continue
        price = exchange_price(snapshot, kind, market, names, outcomes)
        if price is not None and price.overround <= max_exchange_overround:
            exchange = price

    retail: dict[str, RetailQuote] = {}
    fresh = [
        (kind, snapshot)
        for kind, snapshot in latest_retail.items()
        if decision_time - snapshot.snapshot_time <= MAX_RETAIL_AGE
    ]
    for kind, snapshot in sorted(fresh, key=lambda ks: (ks[1].snapshot_time, ks[1].id or 0)):
        for book, prices in book_prices(snapshot.raw_data, kind, market, names).items():
            if book in UK_RETAIL_BOOKMAKERS:
                retail[book] = RetailQuote(snapshot.snapshot_time, prices)

    if exchange is not None and decision_time - exchange.time > MAX_EXCHANGE_AGE:
        exchange = None
    return MarketState(
        event_id=event.id,
        sport_key=event.sport_key,
        market=market,
        kickoff=event.commence_time,
        decision_time=decision_time,
        retail=retail,
        exchange=exchange,
        snapshot_ids=tuple(sorted(s.id for s in snapshots if s.id is not None)),
    )


async def load_market_state(
    session: AsyncSession,
    event: Event,
    decision_time: datetime,
    known_at: datetime | None = None,
) -> MarketState:
    """Build the state from one query for the snapshots in the lookback before ``decision_time``.

    With ``known_at``, only snapshots stored (``created_at``) by then count: the data the
    system had when it acted. The live runner passes its run time and the parity check
    passes the decision's recorded ``known_at``. A historical backtest passes ``None``, because
    backfilled rows were created long after their snapshot time.
    """
    conditions = [
        OddsSnapshot.event_id == event.id,
        OddsSnapshot.snapshot_time <= decision_time,
        OddsSnapshot.snapshot_time >= decision_time - SNAPSHOT_LOOKBACK,
    ]
    if known_at is not None:
        conditions.append(OddsSnapshot.created_at <= known_at)
    result = await session.execute(select(OddsSnapshot).where(*conditions))
    return build_market_state(event, list(result.scalars().all()), decision_time)
