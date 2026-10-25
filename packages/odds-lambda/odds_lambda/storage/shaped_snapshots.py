"""Store an envelope snapshot with its new-shape columns, the other markets' snapshots and prices."""

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
class ShapedWrite:
    """``envelope`` is the write of the snapshot holding the whole envelope, ``None`` when the
    envelope quoted no market and was added without a key; ``others`` are its other markets."""

    snapshot: OddsSnapshot
    envelope: SnapshotWrite | None
    others: tuple[SnapshotWrite, ...]

    @property
    def inserted(self) -> bool:
        return self.envelope is None or self.envelope.outcome is SnapshotWriteOutcome.INSERTED


async def store_payload(session: AsyncSession, payload: dict[str, Any] | list[Any]) -> int:
    """Archive what one fetch returned; its snapshots reference the returned id. Flushes."""
    row = SnapshotPayload(payload=payload)
    session.add(row)
    await session.flush()
    assert row.id is not None
    return row.id


async def store_shaped_snapshot(
    session: AsyncSession,
    snapshot: OddsSnapshot,
    event: Event,
    *,
    source: Source,
    time_basis: TimeBasis,
    payload_id: int | None,
) -> ShapedWrite:
    """Store ``snapshot`` (its ``raw_data`` envelope and old-shape columns already set) under
    the key of the envelope's first market, filling its new-shape columns and price rows.

    Each further market of the envelope (the Odds API's spreads and totals) becomes its own
    snapshot, whose ``raw_data`` holds only that market. A snapshot whose key is already stored
    is not written (``odds_lambda.storage.snapshot_store.store_snapshot``). An envelope that quotes
    no market is added as it is, with no market and no prices. What the envelope held but no row
    could take is logged to ``data_quality_logs``. Flushes; the caller commits.
    """
    expanded = expand_envelope(snapshot.raw_data, source, event.home_team, event.away_team)
    for issue in expanded.issues:
        record_issue(session, event.id, issue.issue_type, issue.description, issue.context)
    snapshot.source = source
    snapshot.time_basis = time_basis
    snapshot.payload_id = payload_id
    if not expanded.markets:
        session.add(snapshot)
        await session.flush()
        return ShapedWrite(snapshot, None, ())

    first, *rest = expanded.markets
    _set_market(snapshot, first)
    envelope = await store_snapshot(session, snapshot, _prices(first.prices))
    others: list[SnapshotWrite] = []
    for market in rest:
        raw_data = market_envelope(snapshot.raw_data, market.market)
        other = OddsSnapshot(
            event_id=snapshot.event_id,
            snapshot_time=snapshot.snapshot_time,
            raw_data=raw_data,
            bookmaker_count=len(raw_data["bookmakers"]),
            api_request_id=snapshot.api_request_id,
            fetch_tier=snapshot.fetch_tier,
            hours_until_commence=snapshot.hours_until_commence,
            source=source,
            time_basis=time_basis,
            payload_id=payload_id,
        )
        _set_market(other, market)
        others.append(await store_snapshot(session, other, _prices(market.prices)))
    return ShapedWrite(envelope.snapshot, envelope, tuple(others))


def _set_market(snapshot: OddsSnapshot, market: MarketSnapshot) -> None:
    snapshot.market = market.market
    snapshot.in_play = market.in_play
    snapshot.market_status = market.market_status
    snapshot.total_matched = market.total_matched
    snapshot.market_start_time = market.market_start_time


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
