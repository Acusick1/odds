"""Store a snapshot and its prices under the snapshot key, keeping the first on a conflict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

import structlog
from odds_core.models import DataQualityLog, OddsSnapshot, Price
from odds_core.time import ensure_utc
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

type PriceRow = tuple[
    str, str, Decimal | None, str, Decimal, Decimal | None, Decimal | None, datetime | None
]
type SnapshotMarketState = tuple[
    str | None, bool | None, str | None, Decimal | None, datetime | None
]


class SnapshotWriteOutcome(StrEnum):
    INSERTED = "inserted"
    IDENTICAL = "identical"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class SnapshotWrite:
    """``snapshot`` is the stored one: the new snapshot, or the one already holding the key."""

    outcome: SnapshotWriteOutcome
    snapshot: OddsSnapshot


async def store_snapshot(
    session: AsyncSession, snapshot: OddsSnapshot, prices: Sequence[Price]
) -> SnapshotWrite:
    """Insert ``snapshot`` with ``prices`` unless a snapshot already holds its key (event, source,
    market, snapshot time). The stored one is kept: an identical incoming snapshot is skipped,
    and a differing one is logged to ``data_quality_logs``. Flushes; the caller commits.

    Raises:
        ValueError: when the snapshot's ``source`` or ``market`` is unset.
    """
    if snapshot.source is None or snapshot.market is None:
        raise ValueError("a snapshot needs its source and market to be stored under its key")
    held = await _holder(session, snapshot)
    if held is None:
        try:
            # A concurrent writer can take the key first; the savepoint keeps its unique
            # violation from aborting the caller's transaction.
            async with session.begin_nested():
                session.add(snapshot)
                await session.flush()
                for price in prices:
                    price.snapshot_id = snapshot.id
                session.add_all(prices)
                await session.flush()
            return SnapshotWrite(SnapshotWriteOutcome.INSERTED, snapshot)
        except IntegrityError:
            held = await _holder(session, snapshot)
            if held is None:
                raise
    if await _same_content(session, held, snapshot, prices):
        return SnapshotWrite(SnapshotWriteOutcome.IDENTICAL, held)
    _log_conflict(session, held, snapshot)
    return SnapshotWrite(SnapshotWriteOutcome.CONFLICTING, held)


async def _holder(session: AsyncSession, snapshot: OddsSnapshot) -> OddsSnapshot | None:
    result = await session.execute(
        select(OddsSnapshot).where(
            OddsSnapshot.event_id == snapshot.event_id,
            OddsSnapshot.source == snapshot.source,
            OddsSnapshot.market == snapshot.market,
            OddsSnapshot.snapshot_time == snapshot.snapshot_time,
        )
    )
    return result.scalar_one_or_none()


def _decimal(value: Decimal | float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _price_row(price: Price) -> PriceRow:
    amount = _decimal(price.price)
    assert amount is not None
    return (
        price.bookmaker,
        str(price.outcome),
        _decimal(price.line),
        str(price.side),
        amount,
        _decimal(price.size),
        _decimal(price.matched),
        None if price.changed_at is None else ensure_utc(price.changed_at),
    )


def _market_state(snapshot: OddsSnapshot) -> SnapshotMarketState:
    return (
        None if snapshot.time_basis is None else str(snapshot.time_basis),
        snapshot.in_play,
        snapshot.market_status,
        _decimal(snapshot.total_matched),
        None if snapshot.market_start_time is None else ensure_utc(snapshot.market_start_time),
    )


async def _same_content(
    session: AsyncSession, held: OddsSnapshot, snapshot: OddsSnapshot, prices: Sequence[Price]
) -> bool:
    if _market_state(held) != _market_state(snapshot):
        return False
    stored = (await session.execute(select(Price).where(Price.snapshot_id == held.id))).scalars()
    return {_price_row(p) for p in stored} == {_price_row(p) for p in prices}


def _log_conflict(session: AsyncSession, held: OddsSnapshot, snapshot: OddsSnapshot) -> None:
    context = {
        "kept_snapshot_id": held.id,
        "source": str(snapshot.source),
        "market": str(snapshot.market),
        "snapshot_time": ensure_utc(snapshot.snapshot_time).isoformat(),
    }
    logger.warning("snapshot_key_conflict", event_id=snapshot.event_id, **context)
    session.add(
        DataQualityLog(
            event_id=snapshot.event_id,
            severity="warning",
            issue_type="snapshot_key_conflict",
            description=(
                f"{snapshot.source} {snapshot.market} snapshot at {context['snapshot_time']} "
                f"differs from stored snapshot {held.id}; the stored one is kept"
            ),
            raw_data=context,
        )
    )
