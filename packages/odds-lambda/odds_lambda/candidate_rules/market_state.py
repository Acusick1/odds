"""What was known about one event at a decision time, built from one co-timed stored snapshot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from odds_core.as_of import RETAIL_KINDS, UPPER_BOUND_TIME_KINDS
from odds_core.betfair_close import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.market_prices import Outcome, book_prices, market_outcomes, outcome_names
from odds_core.models import Event, OddsSnapshot
from odds_core.snapshot_sources import snapshot_kind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.constants import SPORT_CONFIG


@dataclass(frozen=True)
class MarketState:
    """One event at ``decision_time``, from the snapshot ``snapshot_id`` stamped at or before it.

    A state without a snapshot has no inputs: no retail prices and no exchange price.
    """

    event_id: str
    sport_key: str
    market: str
    kickoff: datetime
    decision_time: datetime
    snapshot_id: int | None = None
    snapshot_time: datetime | None = None
    retail: dict[str, dict[Outcome, float]] = field(default_factory=dict)
    exchange: ExchangePrice | None = None

    @property
    def has_input(self) -> bool:
        return self.snapshot_time is not None


def sport_market(sport_key: str) -> str:
    for key, config in SPORT_CONFIG.items():
        if key == sport_key:
            return config.market_key
    raise ValueError(f"No candidate market defined for sport {sport_key!r}")


def build_market_state(
    event: Event,
    snapshots: Sequence[OddsSnapshot],
    decision_time: datetime,
    max_input_age: timedelta,
) -> MarketState:
    """The event's state at ``decision_time`` from one snapshot; later snapshots are ignored.

    The input is the latest retail-kind snapshot at or before t that quotes the market by a
    UK retail book and by the Betfair Exchange, so both prices were taken together, when it
    is at most ``max_input_age`` old; otherwise the state has no inputs. Retail is its UK
    retail books. The exchange price is its devigged Betfair Exchange row, when that prices
    every outcome with a back overround of at most ``MAX_EXCHANGE_OVERROUND``.

    FDUK opening snapshots (``UPPER_BOUND_TIME_KINDS``) are excluded: their stamp is an
    upper bound on the prices' time, which a live run at t never sees.
    """
    market = sport_market(event.sport_key)
    names = outcome_names(event, market)
    state = MarketState(
        event_id=event.id,
        sport_key=event.sport_key,
        market=market,
        kickoff=event.commence_time,
        decision_time=decision_time,
    )
    inputs: list[tuple[OddsSnapshot, dict[str, dict[Outcome, float]]]] = []
    for s in snapshots:
        kind = snapshot_kind(s.raw_data)
        if (
            s.snapshot_time > decision_time
            or kind not in RETAIL_KINDS
            or kind in UPPER_BOUND_TIME_KINDS
        ):
            continue
        books = book_prices(s.raw_data, kind, market, names)
        retail = {book: prices for book, prices in books.items() if book in UK_RETAIL_BOOKMAKERS}
        if retail and EXCHANGE_BOOKMAKER in books:
            inputs.append((s, retail))
    if not inputs:
        return state
    snapshot, retail = max(inputs, key=lambda sr: (sr[0].snapshot_time, sr[0].id or 0))
    if decision_time - snapshot.snapshot_time > max_input_age:
        return state

    exchange = exchange_price(
        snapshot, snapshot_kind(snapshot.raw_data), market, names, market_outcomes(market)
    )
    return replace(
        state,
        snapshot_id=snapshot.id,
        snapshot_time=snapshot.snapshot_time,
        retail=retail,
        exchange=(
            exchange
            if exchange is not None and exchange.overround <= MAX_EXCHANGE_OVERROUND
            else None
        ),
    )


async def load_market_state(
    session: AsyncSession,
    event: Event,
    decision_time: datetime,
    max_input_age: timedelta,
    known_at: datetime | None = None,
) -> MarketState:
    """Build the state from one query for the snapshots within ``max_input_age`` before t.

    With ``known_at``, only snapshots stored (``created_at``) by then count: the data the
    system had when it acted. The live runner passes its run time and the parity check
    passes the decision's recorded ``known_at``. A historical backtest passes ``None``, because
    backfilled rows were created long after their snapshot time.
    """
    conditions = [
        OddsSnapshot.event_id == event.id,
        OddsSnapshot.snapshot_time <= decision_time,
        OddsSnapshot.snapshot_time >= decision_time - max_input_age,
    ]
    if known_at is not None:
        conditions.append(OddsSnapshot.created_at <= known_at)
    result = await session.execute(select(OddsSnapshot).where(*conditions))
    return build_market_state(event, list(result.scalars().all()), decision_time, max_input_age)
