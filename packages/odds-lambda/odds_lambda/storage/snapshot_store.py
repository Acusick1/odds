"""Store a snapshot and its prices under the snapshot key, keeping the first on a conflict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from odds_core.models import OddsSnapshot, Price
from odds_core.time import ensure_utc
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.data_quality import record_issue


@dataclass(frozen=True)
class PriceRow:
    """A price's content, compared by value whatever numeric type it was built with."""

    bookmaker: str
    outcome: str
    line: Decimal | None
    side: str
    price: Decimal
    size: Decimal | None
    matched: Decimal | None
    changed_at: datetime | None


@dataclass(frozen=True)
class SnapshotMarketState:
    """The market-level content of a snapshot beyond its key."""

    time_basis: str | None
    in_play: bool | None
    market_status: str | None
    total_matched: Decimal | None
    market_start_time: datetime | None


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


def _optional_str(value: object | None) -> str | None:
    return None if value is None else str(value)


def _optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else ensure_utc(value)


def _price_row(price: Price) -> PriceRow:
    amount = _decimal(price.price)
    assert amount is not None
    return PriceRow(
        bookmaker=price.bookmaker,
        outcome=str(price.outcome),
        line=_decimal(price.line),
        side=str(price.side),
        price=amount,
        size=_decimal(price.size),
        matched=_decimal(price.matched),
        changed_at=_optional_utc(price.changed_at),
    )


def _market_state(snapshot: OddsSnapshot) -> SnapshotMarketState:
    return SnapshotMarketState(
        time_basis=_optional_str(snapshot.time_basis),
        in_play=snapshot.in_play,
        market_status=_optional_str(snapshot.market_status),
        total_matched=_decimal(snapshot.total_matched),
        market_start_time=_optional_utc(snapshot.market_start_time),
    )


async def _same_content(
    session: AsyncSession, held: OddsSnapshot, snapshot: OddsSnapshot, prices: Sequence[Price]
) -> bool:
    if _market_state(held) != _market_state(snapshot):
        return False
    stored = (await session.execute(select(Price).where(Price.snapshot_id == held.id))).scalars()
    return {_price_row(p) for p in stored} == {_price_row(p) for p in prices}


def _log_conflict(session: AsyncSession, held: OddsSnapshot, snapshot: OddsSnapshot) -> None:
    snapshot_time = ensure_utc(snapshot.snapshot_time).isoformat()
    record_issue(
        session,
        snapshot.event_id,
        "snapshot_key_conflict",
        f"{snapshot.source} {snapshot.market} snapshot at {snapshot_time} differs from stored "
        f"snapshot {held.id}; the stored one is kept",
        {
            "kept_snapshot_id": held.id,
            "source": str(snapshot.source),
            "market": str(snapshot.market),
            "snapshot_time": snapshot_time,
        },
    )
