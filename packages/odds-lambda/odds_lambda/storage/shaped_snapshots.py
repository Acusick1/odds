"""Store an envelope as snapshots with their source, market and price rows, one per market."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from odds_core.models import Event, OddsSnapshot, Price, SnapshotPayload
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.price_rows import MarketSnapshot, PriceSpec, expand_envelope
from odds_lambda.storage.data_quality import record_issue
from odds_lambda.storage.snapshot_store import SnapshotWrite, SnapshotWriteOutcome, store_snapshot


@dataclass(frozen=True)
class SnapshotShape:
    """What a writer knows about a snapshot beyond its envelope: its provider, what its time
    means, and the archived fetch response it came in."""

    source: Source
    time_basis: TimeBasis
    payload_id: int | None


@dataclass(frozen=True)
class ShapedWrite:
    """``snapshot`` is the stored snapshot of the envelope's first market. ``first`` is that
    market's write, ``None`` when the envelope quoted no market and ``snapshot`` was added
    without a key; ``others`` are its other markets' writes."""

    snapshot: OddsSnapshot
    first: SnapshotWrite | None
    others: tuple[SnapshotWrite, ...]

    @property
    def inserted(self) -> bool:
        return self.first is None or self.first.outcome is SnapshotWriteOutcome.INSERTED


async def store_payload(session: AsyncSession, payload: dict[str, Any] | list[Any]) -> int:
    """Archive what one fetch returned; its snapshots reference the returned id. Flushes."""
    row = SnapshotPayload(payload=payload)
    session.add(row)
    await session.flush()
    assert row.id is not None
    return row.id


async def discard_payload(session: AsyncSession, payload_id: int) -> None:
    """Remove a payload no snapshot references, for a fetch whose snapshots were all skipped."""
    await session.execute(delete(SnapshotPayload).where(SnapshotPayload.id == payload_id))  # type: ignore[arg-type]


async def store_shaped_snapshot(
    session: AsyncSession,
    event: Event,
    envelope: Mapping[str, Any],
    snapshot_time: datetime,
    shape: SnapshotShape,
    *,
    created_at: datetime | None = None,
) -> ShapedWrite:
    """Store ``envelope``, what ``shape.source`` gave for ``event`` at ``snapshot_time``, as one
    snapshot per market it quotes, each with its source, market, time basis, market metadata,
    payload and price rows.

    The envelope is read with ``odds_lambda.price_rows.expand_envelope``; its markets are stored
    in ``odds_lambda.price_rows.MARKET_ORDER``, the first as ``ShapedWrite.first``. A snapshot
    whose key is already stored is not written (``odds_lambda.storage.snapshot_store.store_snapshot``).
    An envelope that quotes no market is stored as one snapshot with no market and no prices.
    What the envelope held but no row could take is logged to ``data_quality_logs``. Flushes;
    the caller commits.
    """
    expanded = expand_envelope(envelope, shape.source, event.home_team, event.away_team)
    for issue in expanded.issues:
        record_issue(session, event.id, issue.issue_type.value, issue.description, issue.context)
    if not expanded.markets:
        snapshot = _snapshot(event, snapshot_time, shape, created_at)
        session.add(snapshot)
        await session.flush()
        return ShapedWrite(snapshot, None, ())

    writes: list[SnapshotWrite] = []
    for market in expanded.markets:
        snapshot = _snapshot(event, snapshot_time, shape, created_at)
        _set_market(snapshot, market)
        writes.append(await store_snapshot(session, snapshot, _prices(market.prices)))
    first, *others = writes
    return ShapedWrite(first.snapshot, first, tuple(others))


def _snapshot(
    event: Event, snapshot_time: datetime, shape: SnapshotShape, created_at: datetime | None
) -> OddsSnapshot:
    snapshot = OddsSnapshot(
        event_id=event.id,
        snapshot_time=snapshot_time,
        source=shape.source,
        time_basis=shape.time_basis,
        payload_id=shape.payload_id,
    )
    if created_at is not None:
        snapshot.created_at = created_at
    return snapshot


def _set_market(snapshot: OddsSnapshot, market: MarketSnapshot) -> None:
    snapshot.market = market.market
    snapshot.in_play = market.meta.in_play
    snapshot.market_status = market.meta.market_status
    snapshot.total_matched = market.meta.total_matched
    snapshot.market_start_time = market.meta.market_start_time


def _prices(specs: Sequence[PriceSpec]) -> list[Price]:
    return [
        Price(
            bookmaker=spec.bookmaker,
            outcome=spec.outcome,
            line=spec.line,
            side=spec.side,
            price=spec.price,
            size=spec.size,
            matched=spec.matched,
            changed_at=spec.changed_at,
        )
        for spec in specs
    ]
