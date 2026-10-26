"""What was known about one event at a decision time, built from one co-timed stored snapshot."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from odds_core.as_of import latest_snapshot
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.market_prices import Outcome
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.vocabulary import Side
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.candidate_rules.inputs import RETAIL_SOURCES, rule_quotes


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


def _retail(snapshot: Sequence[Quote]) -> dict[str, dict[Outcome, float]]:
    retail: dict[str, dict[Outcome, float]] = {}
    for q in snapshot:
        if q.bookmaker in UK_RETAIL_BOOKMAKERS:
            retail.setdefault(q.bookmaker, {})[Outcome(q.outcome.value)] = q.price
    return retail


def _co_timed(snapshot: Sequence[Quote]) -> bool:
    books = {q.bookmaker for q in snapshot}
    return EXCHANGE_BOOKMAKER in books and bool(books & UK_RETAIL_BOOKMAKERS)


def build_market_state(
    event: Event,
    quoted: Sequence[Quote],
    decision_time: datetime,
    max_input_age: timedelta,
) -> MarketState:
    """The event's state at ``decision_time`` from one snapshot's quotes; later ones are ignored.

    The input is the latest snapshot of the state's market from a ``RETAIL_SOURCES`` source,
    stamped at or before t, whose back prices quote a UK retail book and the Betfair Exchange, so both
    prices were taken together, when it is at most ``max_input_age`` old; otherwise the state
    has no inputs. Retail is its UK retail books' back prices. The exchange price is its
    devigged Betfair Exchange backs, when they price every outcome with an overround of at most
    ``MAX_EXCHANGE_OVERROUND``.
    """
    market = sport_market(event.sport_key)
    state = MarketState(
        event_id=event.id,
        sport_key=event.sport_key,
        market=market,
        kickoff=event.commence_time,
        decision_time=decision_time,
    )
    snapshot = latest_snapshot(
        (
            q
            for q in quoted
            if q.market == market and q.source in RETAIL_SOURCES and q.side is Side.BACK
        ),
        decision_time,
        _co_timed,
    )
    if snapshot is None:
        return state
    snapshot_time = snapshot[0].snapshot_time
    if decision_time - snapshot_time > max_input_age:
        return state

    exchange = exchange_price([q for q in snapshot if q.bookmaker == EXCHANGE_BOOKMAKER], market)
    return replace(
        state,
        snapshot_id=snapshot[0].snapshot_id,
        snapshot_time=snapshot_time,
        retail=_retail(snapshot),
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
    snapshot_ids: Collection[int] | None = None,
) -> MarketState:
    """Build the state from the quotes ``rule_quotes`` loads within ``max_input_age`` before t.

    With ``known_at``, only snapshots stored (``created_at``) by then count: the data the
    system had when it acted. The live runner passes its run time and the parity check
    passes the decision's recorded ``known_at``. A historical backtest passes ``None``, because
    backfilled rows were created long after their snapshot time. With ``snapshot_ids``, only
    those snapshots count, whatever event they now belong to: the parity check passes a
    decision's recorded inputs.
    """
    quoted = await rule_quotes(
        session,
        event.id,
        sport_market(event.sport_key),
        since=decision_time - max_input_age,
        until=decision_time,
        known_at=known_at,
        snapshot_ids=snapshot_ids,
    )
    return build_market_state(event, quoted, decision_time, max_input_age)
