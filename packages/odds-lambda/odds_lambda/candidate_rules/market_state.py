"""What was known about one event at a decision time, built from one stored retail snapshot and
the sharp prices its rule needs."""

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
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Side, Source
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules.inputs import (
    POLYMARKET_BOOKMAKER,
    RETAIL_SOURCES,
    SharpInput,
    rule_quotes,
)

# A market's line, None in a market without lines.
Line = float | None
# Markets whose prices are on a line.
LINE_MARKETS: frozenset[Market] = frozenset({Market.SPREADS, Market.TOTALS})


@dataclass(frozen=True)
class PolymarketPrice:
    """Polymarket's mids on one line of one snapshot, as implied probabilities normalised to
    sum to 1."""

    snapshot_id: int
    time: datetime
    fair: dict[Outcome, float]


@dataclass(frozen=True)
class MarketState:
    """One event's ``market`` at ``decision_time``, from the retail snapshot ``snapshot_id``
    stamped at or before it.

    ``retail`` holds each line's UK retail back prices per book, ``exchange`` each line's
    devigged exchange price in that snapshot, and ``polymarket`` each line's Polymarket price;
    a market without lines has the one line None. A state without a snapshot has no inputs: no
    retail prices and no sharp prices.
    """

    event_id: int
    sport_key: str
    market: Market
    kickoff: datetime
    decision_time: datetime
    snapshot_id: int | None = None
    snapshot_time: datetime | None = None
    retail: dict[Line, dict[str, dict[Outcome, float]]] = field(default_factory=dict)
    exchange: dict[Line, ExchangePrice] = field(default_factory=dict)
    polymarket: dict[Line, PolymarketPrice] = field(default_factory=dict)

    @property
    def has_input(self) -> bool:
        return self.snapshot_time is not None

    @property
    def input_snapshot_ids(self) -> list[int]:
        """The retail snapshot, then each Polymarket snapshot the state read."""
        if self.snapshot_id is None:
            return []
        polymarket = sorted({p.snapshot_id for p in self.polymarket.values()})
        return [self.snapshot_id, *polymarket]


def is_bettable_line(market: Market, line: Line) -> bool:
    """A line of ``market`` no result can push on: None in a market without lines, else a half
    line."""
    if market not in LINE_MARKETS:
        return line is None
    return line is not None and line % 1 == 0.5


def _retail(
    snapshot: Sequence[Quote], market: Market
) -> dict[Line, dict[str, dict[Outcome, float]]]:
    retail: dict[Line, dict[str, dict[Outcome, float]]] = {}
    for q in snapshot:
        if q.bookmaker in UK_RETAIL_BOOKMAKERS and is_bettable_line(market, q.line):
            retail.setdefault(q.line, {}).setdefault(q.bookmaker, {})[q.outcome] = q.price
    return retail


def _exchange(snapshot: Sequence[Quote], market: Market) -> dict[Line, ExchangePrice]:
    exchange_quotes = [q for q in snapshot if q.bookmaker == EXCHANGE_BOOKMAKER]
    lines = {q.line for q in exchange_quotes if is_bettable_line(market, q.line)}
    prices: dict[Line, ExchangePrice] = {}
    # A market's bettable lines are all None or all numbers, so they sort.
    for line in sorted(lines):  # type: ignore[type-var]
        price = exchange_price(exchange_quotes, market, line)
        if price is not None and price.overround <= MAX_EXCHANGE_OVERROUND:
            prices[line] = price
    return prices


def _polymarket(snapshot: Sequence[Quote], market: Market) -> dict[Line, PolymarketPrice]:
    outcomes = market_outcomes(market)
    mids: dict[Line, dict[Outcome, float]] = {}
    for q in snapshot:
        if is_bettable_line(market, q.line):
            mids.setdefault(q.line, {})[q.outcome] = q.price
    prices: dict[Line, PolymarketPrice] = {}
    # A market's bettable lines are all None or all numbers, so they sort.
    for line in sorted(mids):  # type: ignore[type-var]
        if set(mids[line]) != set(outcomes):
            continue
        implied = {o: 1 / mids[line][o] for o in outcomes}
        total = sum(implied.values())
        prices[line] = PolymarketPrice(
            snapshot_id=snapshot[0].snapshot_id,
            time=snapshot[0].snapshot_time,
            fair={o: p / total for o, p in implied.items()},
        )
    return prices


def _co_timed(snapshot: Sequence[Quote]) -> bool:
    books = {q.bookmaker for q in snapshot}
    return EXCHANGE_BOOKMAKER in books and bool(books & UK_RETAIL_BOOKMAKERS)


def _quotes_retail(snapshot: Sequence[Quote]) -> bool:
    return any(q.bookmaker in UK_RETAIL_BOOKMAKERS for q in snapshot)


def build_market_state(
    event: Event,
    market: Market,
    quoted: Sequence[Quote],
    decision_time: datetime,
    max_input_age: timedelta,
    sharp_inputs: frozenset[SharpInput],
) -> MarketState:
    """The event's state in ``market`` at ``decision_time`` from the quotes stamped at or before
    it; later ones are ignored. ``sharp_inputs`` are the sharp prices the rule needs.

    The retail input is the latest snapshot of the market from a ``RETAIL_SOURCES`` source,
    stamped at or before t, whose back prices quote a UK retail book and, when the rule needs
    ``SharpInput.EXCHANGE``, the Betfair Exchange too, so both prices were taken together. When
    the rule needs ``SharpInput.POLYMARKET``, the Polymarket input is the latest Polymarket
    snapshot of the market stamped at or before t whose mids price every outcome of a line.
    The state has no inputs unless each needed input is at most ``max_input_age`` old. Only
    bettable lines (``is_bettable_line``) count. Retail is its UK retail books' back prices per
    line. A line's exchange price is the retail snapshot's devigged Betfair Exchange backs on
    that line, when they price every outcome with an overround of at most
    ``MAX_EXCHANGE_OVERROUND``. A line's Polymarket price is its mids' implied probabilities
    normalised to sum to 1.
    """
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
        _co_timed if SharpInput.EXCHANGE in sharp_inputs else _quotes_retail,
    )
    if snapshot is None or decision_time - snapshot[0].snapshot_time > max_input_age:
        return state
    polymarket: dict[Line, PolymarketPrice] = {}
    if SharpInput.POLYMARKET in sharp_inputs:
        polymarket_snapshot = latest_snapshot(
            (
                q
                for q in quoted
                if q.market == market
                and q.source is Source.POLYMARKET
                and q.bookmaker == POLYMARKET_BOOKMAKER
                and q.side is Side.MID
            ),
            decision_time,
            lambda s: bool(_polymarket(s, market)),
        )
        if (
            polymarket_snapshot is None
            or decision_time - polymarket_snapshot[0].snapshot_time > max_input_age
        ):
            return state
        polymarket = _polymarket(polymarket_snapshot, market)
    return replace(
        state,
        snapshot_id=snapshot[0].snapshot_id,
        snapshot_time=snapshot[0].snapshot_time,
        retail=_retail(snapshot, market),
        exchange=_exchange(snapshot, market),
        polymarket=polymarket,
    )


async def load_market_state(
    session: AsyncSession,
    event: Event,
    market: Market,
    decision_time: datetime,
    max_input_age: timedelta,
    sharp_inputs: frozenset[SharpInput],
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
        market,
        sharp_inputs,
        since=decision_time - max_input_age,
        until=decision_time,
        known_at=known_at,
        snapshot_ids=snapshot_ids,
    )
    return build_market_state(event, market, quoted, decision_time, max_input_age, sharp_inputs)
