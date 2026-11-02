"""The close of an event's match-odds market: the live Betfair Exchange price ADR-0012 takes,
else the fallback of the event's sport (ADR-0019)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from odds_core.as_of import by_snapshot
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    devig,
    exchange_price,
    overround,
)
from odds_core.market_prices import market_outcomes
from odds_core.models import Event
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis

LIVE_CLOSE_WINDOW = timedelta(minutes=30)

# The source a (sport, market) falls back to when no live Betfair price is in the window. A pair
# without an entry has no close (MLB's h2h).
CLOSE_FALLBACKS: dict[tuple[str, Market], Source] = {
    ("soccer_epl", Market.ONE_X_TWO): Source.FOOTBALL_DATA_UK,
    ("soccer_epl", Market.TOTALS): Source.FOOTBALL_DATA_UK,
    ("basketball_nba", Market.H2H): Source.POLYMARKET,
}
POLYMARKET_BOOKMAKER = "polymarket"


def _polymarket_price(snapshot: Sequence[Quote], market: str) -> ExchangePrice | None:
    """The Polymarket mid price of every outcome of ``market`` in one snapshot's quotes, devigged,
    or None when any is missing. It lives here, not in ``odds_core.exchange_price``, because that
    module's source is part of every candidate rule's pinned hash."""
    outcomes = market_outcomes(market)
    mids = {
        q.outcome: q.price
        for q in snapshot
        if q.bookmaker == POLYMARKET_BOOKMAKER and q.side is Side.MID and q.line is None
    }
    if set(mids) != set(outcomes):
        return None
    first = snapshot[0]
    return ExchangePrice(
        time=first.snapshot_time,
        source=first.source,
        time_basis=first.time_basis,
        in_play=bool(first.in_play),
        fair=devig(mids, outcomes),
        overround=overround(mids, outcomes),
    )


def _snapshot_price(
    snapshot: Sequence[Quote], market: str, line: float | None
) -> ExchangePrice | None:
    if snapshot[0].source is Source.POLYMARKET:
        return _polymarket_price(snapshot, market)
    return exchange_price(snapshot, market, line)


def close_fallback(sport: str, market: str) -> Source | None:
    """The source ``market`` of ``sport`` falls back to for its close, None when it has no close."""
    return CLOSE_FALLBACKS.get((sport, Market(market)))


def has_close(sport: str, market: str) -> bool:
    return close_fallback(sport, market) is not None


def is_live_close(price: ExchangePrice, kickoff: datetime) -> bool:
    """Whether a price can be the live close: a Betfair price, not in-play, in the window."""
    return (
        price.source is Source.BETFAIR
        and not price.in_play
        and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
    )


def is_fallback_close(price: ExchangePrice, kickoff: datetime, fallback: Source) -> bool:
    """Whether a price can be the fallback close: FDUK's closing price (stamped at kick-off), or
    a price of another fallback source observed in the window."""
    if price.source is not fallback:
        return False
    if fallback is Source.FOOTBALL_DATA_UK:
        return price.time_basis is TimeBasis.CLOSE
    return (
        price.time_basis is TimeBasis.OBSERVED
        and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
    )


def pick_close(
    prices: Iterable[ExchangePrice],
    kickoff: datetime,
    fallback: Source | None,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The close among an event's exchange prices, or None when it has none.

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
    fallback = close_fallback(sport, market)
    if fallback is None:
        return None
    prices = (_snapshot_price(group, market, line) for group in by_snapshot(quoted))
    return pick_close(
        (p for p in prices if p is not None),
        kickoff,
        fallback,
        max_exchange_overround=max_exchange_overround,
    )


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
    session: AsyncSession, event_ids: Collection[int], market: str, fallback: Source
) -> list[Quote]:
    if fallback is Source.POLYMARKET:
        return await quotes(
            session,
            event_ids,
            Market(market),
            sources=[Source.POLYMARKET],
            time_bases=[TimeBasis.OBSERVED],
            bookmakers=[POLYMARKET_BOOKMAKER],
            sides=[Side.MID],
            before_kickoff=LIVE_CLOSE_WINDOW,
        )
    return await quotes(
        session,
        event_ids,
        Market(market),
        sources=[fallback],
        time_bases=[TimeBasis.CLOSE],
        bookmakers=[EXCHANGE_BOOKMAKER],
        sides=[Side.BACK],
    )


async def close_quotes(
    session: AsyncSession, events: Sequence[Event], market: str
) -> dict[int, list[Quote]]:
    """The quotes of ``events`` that ``event_close`` can pick, per event id: the live ones
    (``live_close_quotes``) and those of the event's fallback source. An event of a sport
    with no close for ``market`` has none."""
    by_event: dict[int, list[Quote]] = {e.id: [] for e in events}
    by_fallback: dict[Source, list[int]] = defaultdict(list)
    for event in events:
        fallback = close_fallback(event.sport_key, market)
        if fallback is not None:
            by_fallback[fallback].append(event.id)
    for fallback, event_ids in by_fallback.items():
        found = await live_close_quotes(session, event_ids, market)
        found += await _fallback_quotes(session, event_ids, market, fallback)
        for quote in found:
            by_event[quote.event_id].append(quote)
    return by_event
