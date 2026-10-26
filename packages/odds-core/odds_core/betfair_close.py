"""The Betfair Exchange price ADR-0012 takes as the close of an event."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from odds_core.as_of import by_snapshot
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.models import Event
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis

LIVE_CLOSE_WINDOW = timedelta(minutes=30)


def is_live_close(price: ExchangePrice, kickoff: datetime) -> bool:
    """Whether a price can be the live close: a Betfair price, not in-play, in the window."""
    return (
        price.source is Source.BETFAIR
        and not price.in_play
        and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
    )


def is_fduk_close(price: ExchangePrice) -> bool:
    return price.source is Source.FOOTBALL_DATA_UK and price.time_basis is TimeBasis.CLOSE


def pick_close(
    prices: Iterable[ExchangePrice],
    kickoff: datetime,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The close among an event's exchange prices, or None when it has none.

    The close is the last live Betfair price that is not in-play within ``LIVE_CLOSE_WINDOW``
    before kick-off, else the FDUK closing price (stamped at kick-off). Prices whose back
    overround exceeds ``max_exchange_overround`` are not prices.
    """
    live_close: ExchangePrice | None = None
    fduk_close: ExchangePrice | None = None
    for price in sorted(prices, key=lambda p: p.time):
        if price.overround > max_exchange_overround:
            continue
        if is_fduk_close(price):
            fduk_close = price
        elif is_live_close(price, kickoff):
            live_close = price
    return live_close or fduk_close


def event_close(
    quoted: Iterable[Quote],
    kickoff: datetime,
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The devigged Betfair Exchange close of an event from its quotes (``pick_close``)."""
    prices = (exchange_price(group, market) for group in by_snapshot(quoted))
    return pick_close(
        (p for p in prices if p is not None), kickoff, max_exchange_overround=max_exchange_overround
    )


async def live_close_quotes(
    session: AsyncSession, event_ids: Collection[str], market: str
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


async def close_quotes(
    session: AsyncSession, events: Sequence[Event], market: str
) -> dict[str, list[Quote]]:
    """The quotes of ``events`` that ``event_close`` can pick, per event id: the live ones
    (``live_close_quotes``) and the Betfair Exchange backs of FDUK closing snapshots."""
    by_event: dict[str, list[Quote]] = {e.id: [] for e in events}
    if not events:
        return by_event
    fduk = await quotes(
        session,
        list(by_event),
        Market(market),
        sources=[Source.FOOTBALL_DATA_UK],
        time_bases=[TimeBasis.CLOSE],
        bookmakers=[EXCHANGE_BOOKMAKER],
        sides=[Side.BACK],
    )
    for quote in fduk + await live_close_quotes(session, list(by_event), market):
        by_event[quote.event_id].append(quote)
    return by_event
