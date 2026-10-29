"""Stored prices as quotes: each price row with the snapshot it belongs to."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Float, cast, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_core.models import Event, OddsSnapshot, Price
from odds_core.vocabulary import Market, MarketStatus, PriceOutcome, Side, Source, TimeBasis


@dataclass(frozen=True)
class Quote:
    """One price row with its snapshot's id, time, storage time, source, time basis and market
    metadata. Numeric columns are read as floats."""

    snapshot_id: int
    event_id: int
    snapshot_time: datetime
    created_at: datetime
    source: Source
    market: Market
    time_basis: TimeBasis
    in_play: bool | None
    market_status: MarketStatus | None
    total_matched: float | None
    market_start_time: datetime | None
    bookmaker: str
    outcome: PriceOutcome
    line: float | None
    side: Side
    price: float
    size: float | None
    matched: float | None
    changed_at: datetime | None


# One labelled column per Quote field.
QUOTE_COLUMNS: tuple[Any, ...] = (
    col(OddsSnapshot.id).label("snapshot_id"),
    col(OddsSnapshot.event_id).label("event_id"),
    col(OddsSnapshot.snapshot_time).label("snapshot_time"),
    col(OddsSnapshot.created_at).label("created_at"),
    col(OddsSnapshot.source).label("source"),
    col(OddsSnapshot.market).label("market"),
    col(OddsSnapshot.time_basis).label("time_basis"),
    col(OddsSnapshot.in_play).label("in_play"),
    col(OddsSnapshot.market_status).label("market_status"),
    cast(OddsSnapshot.total_matched, Float).label("total_matched"),
    col(OddsSnapshot.market_start_time).label("market_start_time"),
    col(Price.bookmaker).label("bookmaker"),
    col(Price.outcome).label("outcome"),
    cast(Price.line, Float).label("line"),
    col(Price.side).label("side"),
    cast(Price.price, Float).label("price"),
    cast(Price.size, Float).label("size"),
    cast(Price.matched, Float).label("matched"),
    col(Price.changed_at).label("changed_at"),
)


async def quotes(
    session: AsyncSession,
    event_ids: Collection[int],
    market: Market,
    *,
    sources: Collection[Source] | None = None,
    time_bases: Collection[TimeBasis] | None = None,
    bookmakers: Collection[str] | None = None,
    sides: Collection[Side] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    known_at: datetime | None = None,
    snapshot_ids: Collection[int] | None = None,
    before_kickoff: timedelta | None = None,
) -> list[Quote]:
    """The quotes of ``event_ids`` in ``market``, in snapshot time, snapshot id and price id order.

    Every other argument narrows the result when given: snapshots of those ``sources`` and
    ``time_bases``, stamped from ``since`` to ``until`` inclusive, stored (``created_at``) by
    ``known_at``, or among ``snapshot_ids``; prices of those ``bookmakers`` and ``sides``. With
    ``before_kickoff``, each snapshot is stamped within that long before its own event's
    current kick-off, kick-off included.
    """
    conditions: list[Any] = [
        col(OddsSnapshot.event_id).in_(list(event_ids)),
        col(OddsSnapshot.market) == market,
    ]
    if sources is not None:
        conditions.append(col(OddsSnapshot.source).in_(list(sources)))
    if time_bases is not None:
        conditions.append(col(OddsSnapshot.time_basis).in_(list(time_bases)))
    if bookmakers is not None:
        conditions.append(col(Price.bookmaker).in_(list(bookmakers)))
    if sides is not None:
        conditions.append(col(Price.side).in_(list(sides)))
    if since is not None:
        conditions.append(col(OddsSnapshot.snapshot_time) >= since)
    if until is not None:
        conditions.append(col(OddsSnapshot.snapshot_time) <= until)
    if known_at is not None:
        conditions.append(col(OddsSnapshot.created_at) <= known_at)
    if snapshot_ids is not None:
        conditions.append(col(OddsSnapshot.id).in_(list(snapshot_ids)))
    query = select(*QUOTE_COLUMNS).join(Price, col(Price.snapshot_id) == col(OddsSnapshot.id))
    if before_kickoff is not None:
        query = query.join(Event, col(Event.id) == col(OddsSnapshot.event_id))
        conditions.append(
            col(OddsSnapshot.snapshot_time).between(
                col(Event.commence_time) - before_kickoff, col(Event.commence_time)
            )
        )
    rows = await session.execute(
        query.where(*conditions).order_by(
            col(OddsSnapshot.snapshot_time), col(OddsSnapshot.id), col(Price.id)
        )
    )
    return [Quote(**row._mapping) for row in rows.all()]
