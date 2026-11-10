"""The close of an event's market: the live Betfair Exchange price ADR-0012 takes, else the
fallback of the event's sport (ADR-0019)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from odds_core.as_of import by_snapshot
from odds_core.book_price import POLYMARKET_BOOK_SIDE, POLYMARKET_SIDE, book_price
from odds_core.bookmakers import POLYMARKET_BOOKMAKER
from odds_core.exchange_price import EXCHANGE_BOOKMAKER, MAX_EXCHANGE_OVERROUND, ExchangePrice
from odds_core.models import Event
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis

LIVE_CLOSE_WINDOW = timedelta(minutes=30)


@dataclass(frozen=True)
class FallbackSpec:
    """Where a close comes from when no live Betfair price is in the window.

    Its price is ``bookmaker``'s ``side`` in snapshots of ``source`` and ``time_basis``. A
    windowed price is one observed within ``LIVE_CLOSE_WINDOW`` before kick-off; any other is
    stamped at the close. ``book_side`` is the side whose overround the cap checks, where the
    snapshot has it. ``arrives_after`` is how long after kick-off the price can still be stored.
    """

    source: Source
    bookmaker: str
    side: Side
    time_basis: TimeBasis
    windowed: bool
    book_side: Side | None
    arrives_after: timedelta

    @property
    def sides(self) -> list[Side]:
        return [self.side] if self.book_side is None else [self.side, self.book_side]


@dataclass(frozen=True)
class CloseSpec:
    """A (sport, market) that has a close; ``fallback`` is None when only a live Betfair price
    can be it."""

    fallback: FallbackSpec | None


FDUK_CLOSE = FallbackSpec(
    Source.FOOTBALL_DATA_UK,
    EXCHANGE_BOOKMAKER,
    Side.BACK,
    TimeBasis.CLOSE,
    windowed=False,
    book_side=None,
    arrives_after=timedelta(days=7),
)
POLYMARKET_CLOSE = FallbackSpec(
    Source.POLYMARKET,
    POLYMARKET_BOOKMAKER,
    POLYMARKET_SIDE,
    TimeBasis.OBSERVED,
    windowed=True,
    book_side=POLYMARKET_BOOK_SIDE,
    arrives_after=timedelta(0),
)

# The football leagues FDUK closes the three-way and totals markets of.
FDUK_CLOSED_LEAGUES: tuple[str, ...] = (
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
)

# Every (sport, market) a candidate can decide, and its close; None where it has none (MLB,
# ADR-0019).
CLOSES: dict[tuple[str, Market], CloseSpec | None] = {
    **{
        (league, market): CloseSpec(FDUK_CLOSE)
        for league in FDUK_CLOSED_LEAGUES
        for market in (Market.ONE_X_TWO, Market.TOTALS)
    },
    ("basketball_nba", Market.H2H): CloseSpec(POLYMARKET_CLOSE),
    ("baseball_mlb", Market.H2H): None,
    ("baseball_mlb", Market.TOTALS): None,
}


def close_spec(sport: str, market: str) -> CloseSpec | None:
    return CLOSES.get((sport, Market(market)))


def close_grace(sport: str, market: str) -> timedelta:
    """How long after kick-off a missing close may still arrive."""
    spec = close_spec(sport, market)
    return timedelta(0) if spec is None or spec.fallback is None else spec.fallback.arrives_after


def is_live_close(price: ExchangePrice, kickoff: datetime) -> bool:
    """Whether a price can be the live close: a Betfair price, not in-play, in the window."""
    return (
        price.source is Source.BETFAIR
        and not price.in_play
        and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
    )


def is_fallback_close(price: ExchangePrice, kickoff: datetime, fallback: FallbackSpec) -> bool:
    """Whether a price can be the fallback close."""
    return (
        price.source is fallback.source
        and price.time_basis is fallback.time_basis
        and (not fallback.windowed or kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff)
    )


def pick_close(
    prices: Iterable[ExchangePrice],
    kickoff: datetime,
    fallback: FallbackSpec | None,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The close among an event's prices, or None when it has none.

    The close is the last live Betfair price that is not in-play within ``LIVE_CLOSE_WINDOW``
    before kick-off, else the last ``fallback`` price (``is_fallback_close``). Prices whose
    overround exceeds ``max_exchange_overround`` are not prices.
    """
    live_close: ExchangePrice | None = None
    fallback_close: ExchangePrice | None = None
    for price in sorted(prices, key=lambda p: p.time):
        if price.overround > max_exchange_overround:
            continue
        if fallback is not None and is_fallback_close(price, kickoff, fallback):
            fallback_close = price
        elif is_live_close(price, kickoff):
            live_close = price
    return live_close or fallback_close


def event_close(
    quoted: Iterable[Quote],
    kickoff: datetime,
    sport: str,
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
    line: float | None = None,
) -> ExchangePrice | None:
    """The devigged close of an event of ``sport`` on ``line`` from its quotes (``pick_close``);
    a market without lines has the line None. None when the sport has no close for ``market``."""
    spec = close_spec(sport, market)
    if spec is None:
        return None
    fallback = spec.fallback
    prices = (
        book_price(group, market, line, fallback.bookmaker, fallback.side, fallback.book_side)
        if fallback is not None and group[0].source is fallback.source
        else book_price(group, market, line, EXCHANGE_BOOKMAKER, Side.BACK)
        for group in by_snapshot(quoted)
    )
    return pick_close(
        (p for p in prices if p is not None),
        kickoff,
        fallback,
        max_exchange_overround=max_exchange_overround,
    )


class CloseKind(StrEnum):
    LIVE = "live"
    FALLBACK = "fallback"
    PENDING = "pending"
    NONE = "none"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CloseStatus:
    """Where an event's close on one line stands. ``source`` supplied the close: Betfair for a
    live one, the sport's ``FallbackSpec.source`` for a fallback; None without a close."""

    kind: CloseKind
    source: Source | None = None
    price: ExchangePrice | None = None


def close_status(
    quoted: Iterable[Quote],
    kickoff: datetime,
    sport: str,
    market: str,
    now: datetime,
    line: float | None = None,
) -> CloseStatus:
    """Classify the close of an event of ``sport`` on ``line`` from its quotes.

    ``UNSUPPORTED`` when the sport has no close for ``market``; else the close ``event_close``
    picks, ``LIVE`` or ``FALLBACK`` by which rule it met; else ``PENDING`` until ``close_grace``
    after kick-off and ``NONE`` from then on.
    """
    spec = close_spec(sport, market)
    if spec is None:
        return CloseStatus(CloseKind.UNSUPPORTED)
    price = event_close(quoted, kickoff, sport, market, line=line)
    if price is None:
        if now < kickoff + close_grace(sport, market):
            return CloseStatus(CloseKind.PENDING)
        return CloseStatus(CloseKind.NONE)
    if spec.fallback is None or is_live_close(price, kickoff):
        return CloseStatus(CloseKind.LIVE, Source.BETFAIR, price)
    return CloseStatus(CloseKind.FALLBACK, spec.fallback.source, price)


async def live_close_quotes(
    session: AsyncSession, event_ids: Collection[int], market: str
) -> list[Quote]:
    """The Betfair Exchange backs of the events' live Betfair snapshots within
    ``LIVE_CLOSE_WINDOW`` before each one's kick-off, in one query."""
    return await quotes(
        session,
        event_ids,
        Market(market),
        sources=[Source.BETFAIR],
        bookmakers=[EXCHANGE_BOOKMAKER],
        sides=[Side.BACK],
        before_kickoff=LIVE_CLOSE_WINDOW,
    )


async def _fallback_quotes(
    session: AsyncSession, event_ids: Collection[int], market: str, fallback: FallbackSpec
) -> list[Quote]:
    return await quotes(
        session,
        event_ids,
        Market(market),
        sources=[fallback.source],
        time_bases=[fallback.time_basis],
        bookmakers=[fallback.bookmaker],
        sides=fallback.sides,
        before_kickoff=LIVE_CLOSE_WINDOW if fallback.windowed else None,
    )


async def close_quotes(
    session: AsyncSession, events: Sequence[Event], market: str
) -> dict[int, list[Quote]]:
    """The quotes of ``events`` that ``event_close`` can pick, per event id: the live ones
    (``live_close_quotes``) and those of the event's fallback. An event of a sport with no
    close for ``market`` has none."""
    by_event: dict[int, list[Quote]] = {}
    by_spec: dict[CloseSpec, list[int]] = defaultdict(list)
    for event in events:
        if event.id is None:
            continue
        by_event[event.id] = []
        spec = close_spec(event.sport_key, market)
        if spec is not None:
            by_spec[spec].append(event.id)
    for spec, event_ids in by_spec.items():
        found = await live_close_quotes(session, event_ids, market)
        if spec.fallback is not None:
            found += await _fallback_quotes(session, event_ids, market, spec.fallback)
        for quote in found:
            by_event[quote.event_id].append(quote)
    return by_event
