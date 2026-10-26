"""Latest and sharp prices of an event's market, resolved over a lookback window of quotes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from odds_core.match_brief_models import SharpPriceMap, SharpPriceMeta
from odds_core.models import Event, OddsSnapshot, Price
from odds_core.odds_math import decimal_to_american, decimal_to_implied_probability
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, PriceOutcome, Side, Source
from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

OUTCOME_LABELS: dict[PriceOutcome, str] = {
    PriceOutcome.DRAW: "Draw",
    PriceOutcome.OVER: "Over",
    PriceOutcome.UNDER: "Under",
}

BookKey = tuple[str, str, float | None]
SharpKey = tuple[str, float | None]

# A book's price is its back price. Polymarket's back and lay are its ask and bid, so it is its mid.
BOOK_SIDES: tuple[Side, ...] = (Side.BACK, Side.MID)


def is_book_price(source: Source, side: Side) -> bool:
    """Whether a price row of a snapshot from ``source`` is the book's price."""
    return (source is Source.POLYMARKET) == (side is Side.MID) and side in BOOK_SIDES


# The same rule in SQL, over a snapshot and one of its price rows.
BOOK_PRICE_CONDITION = or_(
    and_(OddsSnapshot.source == Source.POLYMARKET, Price.side == Side.MID),  # type: ignore[arg-type]
    and_(OddsSnapshot.source != Source.POLYMARKET, Price.side == Side.BACK),  # type: ignore[arg-type]
)


@dataclass
class SharpLines:
    """Sharp prices per (outcome label, line), with the snapshot each came from."""

    prices: dict[SharpKey, dict[str, object]] = field(default_factory=dict)
    meta: dict[SharpKey, SharpPriceMeta] = field(default_factory=dict)

    def by_outcome(self) -> SharpPriceMap:
        """One price per outcome label, from its lowest line (no line first)."""
        chosen: dict[str, SharpKey] = {}
        for key in sorted(self.prices, key=lambda k: (k[0], -math.inf if k[1] is None else k[1])):
            chosen.setdefault(key[0], key)
        return {name: self.prices[key] for name, key in chosen.items()}  # type: ignore[misc]


@dataclass(frozen=True)
class BookPrice:
    """A book's price for one outcome and line, with the snapshot it is from."""

    bookmaker: str
    market: Market
    outcome_name: str
    line: float | None
    price_decimal: float
    changed_at: datetime | None
    snapshot_id: int
    snapshot_time: datetime
    age_seconds: float

    @classmethod
    def from_quote(cls, event: Event, quote: Quote, anchor_time: datetime) -> BookPrice:
        return cls(
            bookmaker=quote.bookmaker,
            market=quote.market,
            outcome_name=outcome_label(event, quote.outcome),
            line=quote.line,
            price_decimal=quote.price,
            changed_at=quote.changed_at,
            snapshot_id=quote.snapshot_id,
            snapshot_time=quote.snapshot_time,
            age_seconds=round((anchor_time - quote.snapshot_time).total_seconds(), 1),
        )

    @property
    def american_price(self) -> int:
        return decimal_to_american(self.price_decimal)

    @property
    def implied_probability(self) -> float:
        return decimal_to_implied_probability(self.price_decimal)

    @property
    def last_update(self) -> datetime:
        return self.changed_at or self.snapshot_time


def outcome_label(event: Event, outcome: PriceOutcome) -> str:
    """The label a tool shows for a price outcome: the team name, ``Draw``, ``Over`` or ``Under``."""
    if outcome is PriceOutcome.HOME:
        return event.home_team
    if outcome is PriceOutcome.AWAY:
        return event.away_team
    return OUTCOME_LABELS[outcome]


async def latest_priced_snapshot(
    session: AsyncSession, event_id: str, market: Market
) -> OddsSnapshot | None:
    """The newest snapshot of ``market`` that holds a book price."""
    has_book_price = exists().where(
        Price.snapshot_id == OddsSnapshot.id,  # type: ignore[arg-type]
        BOOK_PRICE_CONDITION,
    )
    result = await session.execute(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.event_id == event_id,  # type: ignore[arg-type]
            OddsSnapshot.market == market,  # type: ignore[arg-type]
            has_book_price,
        )
        .order_by(OddsSnapshot.snapshot_time.desc(), OddsSnapshot.id)  # type: ignore[attr-defined,arg-type]
        .limit(1)
    )
    return result.scalar_one_or_none()


async def book_quotes(
    session: AsyncSession,
    event_id: str,
    market: Market,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Quote]:
    """The book prices of ``market`` in snapshot time order: back prices, and Polymarket's mid."""
    found = await quotes(session, [event_id], market, sides=BOOK_SIDES, since=since, until=until)
    return [q for q in found if is_book_price(q.source, q.side)]


def newest_first(found: list[Quote]) -> list[Quote]:
    """``found`` by snapshot time descending then snapshot id, keeping price order in a snapshot."""
    return sorted(found, key=lambda q: (-q.snapshot_time.timestamp(), q.snapshot_id))


async def latest_book_prices(
    session: AsyncSession,
    event: Event,
    market: Market,
    anchor_time: datetime,
    lookback_hours: float,
) -> dict[BookKey, BookPrice]:
    """The newest book price per (bookmaker, outcome, line) within the lookback window.

    The window runs ``lookback_hours`` back from ``anchor_time``. Snapshots are walked newest
    first and the first price of a key wins, so a snapshot holding only some books (a Betfair
    snapshot beside an older retail one) does not hide the others.
    """
    window = await book_quotes(
        session,
        event.id,
        market,
        since=anchor_time - timedelta(hours=lookback_hours),
        until=anchor_time,
    )
    prices: dict[BookKey, BookPrice] = {}
    for quote in newest_first(window):
        price = BookPrice.from_quote(event, quote, anchor_time)
        prices.setdefault((price.bookmaker, price.outcome_name, price.line), price)
    return prices


async def resolve_sharp_prices(
    session: AsyncSession,
    event: Event,
    market: Market,
    sharp_bookmakers: Sequence[str],
    lookback_hours: float,
    anchor_time: datetime,
) -> SharpLines:
    """The newest sharp price per (outcome label, line) within the lookback window.

    Snapshots are walked newest first; an outcome at a line takes the first sharp book, in the
    priority order of ``sharp_bookmakers``, that prices it in the snapshot, and is not revisited.
    """
    window = await book_quotes(
        session,
        event.id,
        market,
        since=anchor_time - timedelta(hours=lookback_hours),
        until=anchor_time,
    )
    by_snapshot: dict[int, list[Quote]] = {}
    for quote in newest_first(window):
        by_snapshot.setdefault(quote.snapshot_id, []).append(quote)

    result = SharpLines()
    seen: set[SharpKey] = set()
    for snapshot_quotes in by_snapshot.values():
        keyed = [((outcome_label(event, q.outcome), q.line), q) for q in snapshot_quotes]
        seen.update(key for key, _ in keyed)
        for key in {key for key, _ in keyed}:
            if key in result.prices:
                continue
            for bookmaker in sharp_bookmakers:
                match = next(
                    (q for k, q in keyed if k == key and q.bookmaker == bookmaker),
                    None,
                )
                if match is None:
                    continue
                result.prices[key] = {
                    "bookmaker": bookmaker,
                    "price": decimal_to_american(match.price),
                    "price_decimal": match.price,
                    "implied_prob": round(decimal_to_implied_probability(match.price), 6),
                }
                result.meta[key] = SharpPriceMeta(
                    snapshot_id=match.snapshot_id,
                    snapshot_time=match.snapshot_time,
                    age_seconds=round((anchor_time - match.snapshot_time).total_seconds(), 1),
                )
                break
        if seen <= result.prices.keys():
            break
    return result
