"""Store an envelope snapshot with its source, market and price rows, one snapshot per market."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from odds_core.models import Event, OddsSnapshot, Price, SnapshotPayload
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.price_rows import MarketSnapshot, PriceSpec, expand_envelope, market_envelope
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
    """``first`` is the write of the snapshot the writer built, ``None`` when the envelope
    quoted no market and the snapshot was added without a key; ``others`` are its other
    markets' snapshots."""

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


async def store_shaped_snapshot(
    session: AsyncSession, snapshot: OddsSnapshot, event: Event, shape: SnapshotShape
) -> ShapedWrite:
    """Store ``snapshot`` (its ``raw_data`` envelope and the envelope columns already set) under
    the key of the envelope's first market, filling its source, market, time basis, market
    metadata, payload and price rows.

    An envelope quoting several markets (the Odds API's 1x2, spreads and totals) becomes one
    snapshot per market, each with ``raw_data`` cut to its own market; ``snapshot`` takes the
    first. A snapshot whose key is already stored is not written
    (``odds_lambda.storage.snapshot_store.store_snapshot``). An envelope that quotes no market is
    added as it is, with no market and no prices. What the envelope held but no row could take is
    logged to ``data_quality_logs``. Flushes; the caller commits.
    """
    expanded = expand_envelope(snapshot.raw_data, shape.source, event.home_team, event.away_team)
    for issue in expanded.issues:
        record_issue(session, event.id, issue.issue_type.value, issue.description, issue.context)
    if not expanded.markets:
        _set_shape(snapshot, shape)
        session.add(snapshot)
        await session.flush()
        return ShapedWrite(snapshot, None, ())

    envelope = snapshot.raw_data
    split = len(expanded.markets) > 1
    writes: list[SnapshotWrite] = []
    for i, market in enumerate(expanded.markets):
        target = snapshot if i == 0 else _sibling(snapshot)
        if split:
            target.raw_data = market_envelope(envelope, market.market)
            target.bookmaker_count = len(target.raw_data["bookmakers"])
        _set_shape(target, shape)
        _set_market(target, market)
        writes.append(await store_snapshot(session, target, _prices(market.prices)))
    first, *others = writes
    return ShapedWrite(first.snapshot, first, tuple(others))


def _sibling(snapshot: OddsSnapshot) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=snapshot.event_id,
        snapshot_time=snapshot.snapshot_time,
        raw_data={},
        bookmaker_count=0,
        api_request_id=snapshot.api_request_id,
        fetch_tier=snapshot.fetch_tier,
        hours_until_commence=snapshot.hours_until_commence,
    )


def _set_shape(snapshot: OddsSnapshot, shape: SnapshotShape) -> None:
    snapshot.source = shape.source
    snapshot.time_basis = shape.time_basis
    snapshot.payload_id = shape.payload_id


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
