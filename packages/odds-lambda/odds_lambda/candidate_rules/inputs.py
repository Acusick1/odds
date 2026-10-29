"""The only query behind a candidate rule's inputs, kept apart from ``odds_core.quotes`` so the
rule hash covers it and nothing else that reads prices."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from enum import StrEnum
from typing import Any

from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.exchange_price import EXCHANGE_BOOKMAKER
from odds_core.models import OddsSnapshot, Price
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Side, Source
from sqlalchemy import Float, and_, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

# Sources whose snapshots quote retail books; FDUK opening prices are stamped only with an
# upper bound on when they were taken, which a live run at t never sees.
RETAIL_SOURCES: tuple[Source, ...] = (Source.ODDS_API, Source.ODDSPORTAL)
RULE_BOOKMAKERS: frozenset[str] = UK_RETAIL_BOOKMAKERS | {EXCHANGE_BOOKMAKER}
POLYMARKET_BOOKMAKER = "polymarket"


class SharpInput(StrEnum):
    """A sharp price a rule compares retail with: the Betfair Exchange backs in the retail
    snapshot itself, or Polymarket's mids from its own latest snapshot."""

    EXCHANGE = "exchange"
    POLYMARKET = "polymarket"


_COLUMNS: tuple[Any, ...] = (
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


async def rule_quotes(
    session: AsyncSession,
    event_id: int,
    market: Market,
    sharp_inputs: Collection[SharpInput],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    known_at: datetime | None = None,
    snapshot_ids: Collection[int] | None = None,
) -> list[Quote]:
    """The back prices of UK retail books and the Betfair Exchange in ``event_id``'s retail
    snapshots of ``market``, and with ``SharpInput.POLYMARKET`` in ``sharp_inputs`` the mids of
    its Polymarket snapshots, in snapshot time, snapshot id and price id order.

    Narrowed when given to snapshots stamped from ``since`` to ``until`` inclusive, or stored
    (``created_at``) by ``known_at``. With ``snapshot_ids``, those snapshots in place of the
    event's, whatever event they now belong to.
    """
    retail = and_(
        col(OddsSnapshot.source).in_(RETAIL_SOURCES),
        col(Price.side) == Side.BACK,
        col(Price.bookmaker).in_(sorted(RULE_BOOKMAKERS)),
    )
    polymarket = and_(
        col(OddsSnapshot.source) == Source.POLYMARKET,
        col(Price.side) == Side.MID,
        col(Price.bookmaker) == POLYMARKET_BOOKMAKER,
    )
    conditions: list[Any] = [
        col(OddsSnapshot.market) == Market(market),
        or_(retail, polymarket) if SharpInput.POLYMARKET in sharp_inputs else retail,
    ]
    if since is not None:
        conditions.append(col(OddsSnapshot.snapshot_time) >= since)
    if until is not None:
        conditions.append(col(OddsSnapshot.snapshot_time) <= until)
    if known_at is not None:
        conditions.append(col(OddsSnapshot.created_at) <= known_at)
    if snapshot_ids is None:
        conditions.append(col(OddsSnapshot.event_id) == event_id)
    else:
        conditions.append(col(OddsSnapshot.id).in_(list(snapshot_ids)))
    rows = await session.execute(
        select(*_COLUMNS)
        .join(Price, col(Price.snapshot_id) == col(OddsSnapshot.id))
        .where(*conditions)
        .order_by(col(OddsSnapshot.snapshot_time), col(OddsSnapshot.id), col(Price.id))
    )
    return [Quote(**row._mapping) for row in rows.all()]
