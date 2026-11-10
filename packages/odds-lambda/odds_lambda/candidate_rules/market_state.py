"""What was known about one event at a decision time, built from one stored retail snapshot and
the sharp prices its rule needs."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from types import MappingProxyType

from odds_core.as_of import latest_snapshot
from odds_core.book_price import polymarket_price
from odds_core.bookmakers import POLYMARKET_BOOKMAKER, UK_RETAIL_BOOKMAKERS
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.market_prices import Outcome
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Side, Source
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules.fixture_context import FixtureContext
from odds_lambda.candidate_rules.inputs import RETAIL_SOURCES, SharpInput, rule_quotes

# A market's line, None in a market without lines.
Line = float | None
# Markets whose prices are on a line.
LINE_MARKETS: frozenset[Market] = frozenset({Market.SPREADS, Market.TOTALS})


@dataclass(frozen=True)
class Lookback:
    """A point ``minutes_before_decision`` before the decision time, whose state may be at most
    ``max_input_age`` old at that point."""

    minutes_before_decision: int
    max_input_age: timedelta


@dataclass(frozen=True)
class PolymarketPrice:
    """Polymarket's mids on one line of one snapshot, as implied probabilities normalised to
    sum to 1."""

    snapshot_id: int
    time: datetime
    fair: Mapping[Outcome, float]


@dataclass(frozen=True)
class MarketState:
    """One event's ``market`` at ``decision_time``, from the retail snapshot ``snapshot_id``
    stamped at or before it.

    ``retail`` holds each line's UK retail back prices per book, ``exchange`` each line's
    devigged exchange price in that snapshot, and ``polymarket`` each line's Polymarket price;
    a market without lines has the one line None. Every mapping is read-only, and
    ``input_ids`` is fixed when the state is built, so nothing a rule does changes what the
    decision records. ``also`` holds the state of each extra market
    the rule declared, read-only, from the same source run as the retail snapshot where the
    source stored one. ``lookbacks`` holds the state at each lookback point the rule declared,
    keyed by its minutes before the decision time, each built as of that point with the same
    extra markets and no lookbacks of its own. A state without a snapshot has no inputs: no
    retail prices, no sharp prices, no extra markets and no lookbacks. ``fixture`` holds the
    fixture inputs the rule declared, read-only, which the loader of the rule adds
    (``odds_lambda.candidate_rules.rules.add_fixture_context``); a state whose ``fixture`` has
    unknown results has no inputs too.
    """

    event_id: int
    sport_key: str
    market: Market
    kickoff: datetime
    decision_time: datetime
    snapshot_id: int | None = None
    snapshot_time: datetime | None = None
    retail: Mapping[Line, Mapping[str, Mapping[Outcome, float]]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    exchange: Mapping[Line, ExchangePrice] = field(default_factory=lambda: MappingProxyType({}))
    polymarket: Mapping[Line, PolymarketPrice] = field(default_factory=lambda: MappingProxyType({}))
    also: Mapping[Market, MarketState] = field(default_factory=lambda: MappingProxyType({}))
    lookbacks: Mapping[int, MarketState] = field(default_factory=lambda: MappingProxyType({}))
    input_ids: tuple[int, ...] = ()
    fixture: FixtureContext = field(default_factory=FixtureContext)

    @property
    def has_input(self) -> bool:
        return self.snapshot_time is not None

    @property
    def input_snapshot_ids(self) -> list[int]:
        """The retail snapshot, then each Polymarket snapshot, then each extra market's inputs
        in market order, then each lookback's inputs from the nearest point to the furthest;
        empty for a state without inputs."""
        return list(self.input_ids)

    @property
    def input_event_ids(self) -> list[int]:
        """The events the fixture inputs were read from."""
        return list(self.fixture.event_ids)

    @property
    def missing_fixture(self) -> bool:
        """Whether a fixture input has no known result, which leaves the state without inputs."""
        return bool(self.fixture.unknown_results)

    @property
    def fixture_overdue(self) -> bool:
        """Whether a previous result is overdue, not merely pending."""
        return bool(self.fixture.overdue_results)


def is_bettable_line(market: Market, line: Line) -> bool:
    """A line of ``market`` no result can push on: None in a market without lines, else a half
    line."""
    if market not in LINE_MARKETS:
        return line is None
    return line is not None and line % 1 == 0.5


def _retail(
    snapshot: Sequence[Quote], market: Market
) -> Mapping[Line, Mapping[str, Mapping[Outcome, float]]]:
    retail: dict[Line, dict[str, dict[Outcome, float]]] = {}
    for q in snapshot:
        if q.bookmaker in UK_RETAIL_BOOKMAKERS and is_bettable_line(market, q.line):
            retail.setdefault(q.line, {}).setdefault(q.bookmaker, {})[q.outcome] = q.price
    return MappingProxyType(
        {
            line: MappingProxyType(
                {book: MappingProxyType(prices) for book, prices in books.items()}
            )
            for line, books in retail.items()
        }
    )


def _exchange(snapshot: Sequence[Quote], market: Market) -> Mapping[Line, ExchangePrice]:
    exchange_quotes = [q for q in snapshot if q.bookmaker == EXCHANGE_BOOKMAKER]
    lines = {q.line for q in exchange_quotes if is_bettable_line(market, q.line)}
    prices: dict[Line, ExchangePrice] = {}
    # A market's bettable lines are all None or all numbers, so they sort.
    for line in sorted(lines):  # type: ignore[type-var]
        price = exchange_price(exchange_quotes, market, line)
        if price is not None and price.overround <= MAX_EXCHANGE_OVERROUND:
            prices[line] = replace(price, fair=MappingProxyType(dict(price.fair)))
    return MappingProxyType(prices)


def _polymarket(snapshot: Sequence[Quote], market: Market) -> Mapping[Line, PolymarketPrice]:
    lines = {q.line for q in snapshot if is_bettable_line(market, q.line)}
    prices: dict[Line, PolymarketPrice] = {}
    # A market's bettable lines are all None or all numbers, so they sort.
    for line in sorted(lines):  # type: ignore[type-var]
        price = polymarket_price(snapshot, market, line)
        if price is not None:
            prices[line] = PolymarketPrice(
                snapshot[0].snapshot_id, price.time, MappingProxyType(dict(price.fair))
            )
    return MappingProxyType(prices)


def _co_timed(snapshot: Sequence[Quote]) -> bool:
    books = {q.bookmaker for q in snapshot}
    return EXCHANGE_BOOKMAKER in books and bool(books & UK_RETAIL_BOOKMAKERS)


def _quotes_retail(snapshot: Sequence[Quote]) -> bool:
    return any(q.bookmaker in UK_RETAIL_BOOKMAKERS for q in snapshot)


def _in_run(quoted: Iterable[Quote], market: Market, run: Sequence[Quote]) -> list[Quote]:
    """The quotes of ``market`` from the source run of the snapshot ``run``: its payload when
    the writer stored one, else its source."""
    head = run[0]
    return [
        q
        for q in quoted
        if q.market == market
        and q.source == head.source
        and (head.payload_id is None or q.payload_id == head.payload_id)
    ]


def _no_input(event: Event, market: Market, decision_time: datetime) -> MarketState:
    return MarketState(
        event_id=event.id,
        sport_key=event.sport_key,
        market=market,
        kickoff=event.commence_time,
        decision_time=decision_time,
    )


def _own_state(
    event: Event,
    market: Market,
    quoted: Sequence[Quote],
    decision_time: datetime,
    max_input_age: timedelta,
    sharp_inputs: frozenset[SharpInput],
    also_markets: frozenset[Market],
) -> MarketState:
    state = _no_input(event, market, decision_time)
    retail_quotes = [
        q
        for q in quoted
        if q.market == market and q.source in RETAIL_SOURCES and q.side is Side.BACK
    ]
    accept = _co_timed if SharpInput.EXCHANGE in sharp_inputs else _quotes_retail
    extra_sharp = sharp_inputs & {SharpInput.EXCHANGE}

    def extras(run: Sequence[Quote]) -> dict[Market, MarketState]:
        return {
            m: _own_state(
                event,
                m,
                _in_run(quoted, m, run),
                decision_time,
                max_input_age,
                extra_sharp,
                frozenset(),
            )
            for m in sorted(also_markets)
        }

    def accepted(run: Sequence[Quote]) -> bool:
        return accept(run) and all(s.has_input for s in extras(run).values())

    snapshot = latest_snapshot(retail_quotes, decision_time, accepted)
    if snapshot is None or decision_time - snapshot[0].snapshot_time > max_input_age:
        return state
    polymarket: Mapping[Line, PolymarketPrice] = MappingProxyType({})
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
    also = extras(snapshot)
    snapshot_id = snapshot[0].snapshot_id
    return replace(
        state,
        snapshot_id=snapshot_id,
        snapshot_time=snapshot[0].snapshot_time,
        retail=_retail(snapshot, market),
        exchange=_exchange(snapshot, market),
        polymarket=polymarket,
        also=MappingProxyType(also),
        input_ids=(
            snapshot_id,
            *sorted({p.snapshot_id for p in polymarket.values()}),
            *(i for m in sorted(also) for i in also[m].input_ids),
        ),
    )


def build_market_state(
    event: Event,
    market: Market,
    quoted: Sequence[Quote],
    decision_time: datetime,
    max_input_age: timedelta,
    sharp_inputs: frozenset[SharpInput],
    also_markets: frozenset[Market] = frozenset(),
    lookbacks: Sequence[Lookback] = (),
) -> MarketState:
    """The event's state in ``market`` at ``decision_time`` from the quotes stamped at or before
    it; later ones are ignored. ``sharp_inputs`` are the sharp prices the rule needs and
    ``also_markets`` the extra markets it reads.

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

    Each of ``also_markets`` is read as a state of its own, in ``MarketState.also``, with the
    retail input chosen by the same rule (and the exchange when needed, never Polymarket) from
    the quotes of the same source run as the market's retail snapshot: the same payload where
    the writer stored one, else the same source. The market's retail snapshot is the latest one
    for which every extra market has such an input, and the state has no inputs when that
    snapshot or any extra input is older than ``max_input_age``.

    Each of ``lookbacks`` is read as a state of the same market, in ``MarketState.lookbacks``,
    built by this same procedure as of its point, ``decision_time`` less its minutes, from the
    quotes stamped at or before that point, with the lookback's own max input age and the same
    sharp inputs and extra markets. The state has no inputs when any lookback has none.
    """
    state = _own_state(
        event,
        market,
        quoted,
        decision_time,
        max_input_age,
        sharp_inputs,
        frozenset(also_markets),
    )
    if not state.has_input or not lookbacks:
        return state
    points = {
        point.minutes_before_decision: _own_state(
            event,
            market,
            quoted,
            decision_time - timedelta(minutes=point.minutes_before_decision),
            point.max_input_age,
            sharp_inputs,
            frozenset(also_markets),
        )
        for point in sorted(lookbacks, key=lambda p: p.minutes_before_decision)
    }
    if not all(p.has_input for p in points.values()):
        return _no_input(event, market, decision_time)
    return replace(
        state,
        lookbacks=MappingProxyType(points),
        input_ids=(*state.input_ids, *(i for p in points.values() for i in p.input_ids)),
    )


def input_window(max_input_age: timedelta, lookbacks: Sequence[Lookback] = ()) -> timedelta:
    """How far before a decision time its inputs can lie: the decision's max input age, or a
    lookback's point and max input age if further."""
    reaches = [timedelta(minutes=p.minutes_before_decision) + p.max_input_age for p in lookbacks]
    return max([max_input_age, *reaches])


async def load_market_state(
    session: AsyncSession,
    event: Event,
    market: Market,
    decision_time: datetime,
    max_input_age: timedelta,
    sharp_inputs: frozenset[SharpInput],
    known_at: datetime | None = None,
    snapshot_ids: Collection[int] | None = None,
    also_markets: frozenset[Market] = frozenset(),
    lookbacks: Sequence[Lookback] = (),
) -> MarketState:
    """Build the state from the quotes ``rule_quotes`` loads within ``input_window`` before t.

    With ``known_at``, only snapshots stored (``created_at``) by then count: the data the
    system had when it acted. The live runner passes its run time and the parity check
    passes the decision's recorded ``known_at``. A historical backtest passes ``None``, because
    backfilled rows were created long after their snapshot time. With ``snapshot_ids``, only
    those snapshots count, whatever event they now belong to: the parity check passes a
    decision's recorded inputs, those of ``also_markets`` and ``lookbacks`` included. Every
    lookback reads the quotes ``known_at`` allows, and none read past its own point.
    """
    quoted = await rule_quotes(
        session,
        event.id,
        market,
        sharp_inputs,
        since=decision_time - input_window(max_input_age, lookbacks),
        until=decision_time,
        known_at=known_at,
        snapshot_ids=snapshot_ids,
        also_markets=also_markets,
    )
    return build_market_state(
        event, market, quoted, decision_time, max_input_age, sharp_inputs, also_markets, lookbacks
    )
