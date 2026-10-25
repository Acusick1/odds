"""Fill the new-shape columns, payload and price rows of every snapshot stored before them.

A legacy snapshot (``source`` unset) is read with the live writers' reader
(``odds_lambda.price_rows.expand_envelope``) and takes its source, market, time basis, market
metadata and price rows. Its ``raw_data`` is archived as stored in one ``snapshot_payloads`` row.
An envelope quoting several markets (the Odds API) is split as the live writers split it: the
first market keeps the snapshot id, the others become new snapshots with ``raw_data`` cut to their
market, and each ``odds`` row moves to the snapshot of its market.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from odds_core.models import (
    DataQualityLog,
    Event,
    Odds,
    OddsSnapshot,
    Price,
    SnapshotPayload,
    utc_now,
)
from odds_core.paper_trade_models import CandidateDecision
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotSource
from odds_core.time import ensure_utc
from odds_core.vocabulary import Market, Source, TimeBasis
from sqlalchemy import and_, case, delete, exists, func, insert, or_, select, tuple_, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from odds_lambda.price_rows import (
    MarketMeta,
    MarketSnapshot,
    PriceSpec,
    RowIssue,
    expand_envelope,
    market_envelope,
    stored_market,
)

# Odds API snapshots carry no tag telling a live fetch from a historical one. On production a
# live fetch is stored within seconds of its snapshot time, a historical one hours to months
# after it; nothing falls between 12 s and 2.8 h.
ODDS_API_HISTORICAL_GAP = timedelta(hours=1)
KEY_CONFLICT_ISSUE = "snapshot_key_conflict"
# A batch that loses a race with a live writer, for a snapshot key or in a deadlock over the
# rows a re-timing writer locks, is planned again.
BATCH_ATTEMPTS = 3
_RETRIED_SQLSTATES = frozenset({"23505", "40P01", "40001"})

SnapshotKey = tuple[str, Source, Market, datetime]

_SOURCE_TAGS: Mapping[SnapshotSource, tuple[Source, TimeBasis]] = {
    SnapshotSource.ODDSPORTAL: (Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED),
    SnapshotSource.ODDSPORTAL_LIVE: (Source.ODDSPORTAL, TimeBasis.OBSERVED),
    SnapshotSource.BETFAIR_API: (Source.BETFAIR, TimeBasis.OBSERVED),
    # History and live alike: the history is an as-of resample at our own grid times.
    SnapshotSource.POLYMARKET: (Source.POLYMARKET, TimeBasis.OBSERVED),
}


class UnknownSnapshotTag(ValueError):
    """A snapshot's ``raw_data`` tags name no known source or FDUK timing."""


class SplitSnapshotReferenced(RuntimeError):
    """A candidate decision recorded a snapshot the backfill would split or delete."""


@dataclass(frozen=True)
class StoredSnapshot:
    """A snapshot as the backfill reads it, with its event's teams."""

    id: int
    event_id: str
    snapshot_time: datetime
    created_at: datetime
    raw_data: dict[str, Any]
    api_request_id: str | None
    fetch_tier: str | None
    hours_until_commence: float | None
    home_team: str
    away_team: str
    source: Source | None
    market: Market | None = None
    time_basis: TimeBasis | None = None
    payload_id: int | None = None
    meta: MarketMeta = MarketMeta()
    # An untagged football-data.co.uk snapshot with a later untagged one on its event, which
    # makes it the opening of the pair the loader wrote before timing tags existed.
    earlier_untagged_fduk: bool = False


def classify(snapshot: StoredSnapshot) -> tuple[Source, TimeBasis]:
    """The source and time basis of a legacy snapshot, read from its ``raw_data`` tags.

    Raises:
        UnknownSnapshotTag: for a tag or FDUK timing basis that names nothing known.
    """
    tag = snapshot.raw_data.get("source")
    if tag is None:
        gap = ensure_utc(snapshot.created_at) - ensure_utc(snapshot.snapshot_time)
        if gap > ODDS_API_HISTORICAL_GAP:
            return Source.ODDS_API, TimeBasis.UPPER_BOUND
        return Source.ODDS_API, TimeBasis.OBSERVED
    try:
        source = SnapshotSource(tag)
    except ValueError as exc:
        raise UnknownSnapshotTag(f"snapshot {snapshot.id} has source tag {tag!r}") from exc
    if source is not SnapshotSource.FOOTBALL_DATA_UK:
        return _SOURCE_TAGS[source]
    basis = (snapshot.raw_data.get("timing") or {}).get("basis")
    if basis == FdukTimeBasis.KICKOFF:
        return Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE
    if basis == FdukTimeBasis.COLLECTION_DEADLINE:
        return Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND
    if basis is None:
        if snapshot.earlier_untagged_fduk:
            return Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND
        return Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE
    raise UnknownSnapshotTag(f"snapshot {snapshot.id} has FDUK timing basis {basis!r}")


@dataclass(frozen=True)
class KeyConflict:
    """A market of a legacy snapshot whose key another snapshot holds; the holder is kept."""

    snapshot_id: int
    event_id: str
    source: Source
    market: Market
    snapshot_time: datetime
    held_by: int


@dataclass
class SnapshotPlan:
    """What the backfill writes for one legacy snapshot: ``kept`` markets become snapshots
    (the first on the legacy row), ``conflicts`` are markets another snapshot holds the key of."""

    snapshot: StoredSnapshot
    source: Source
    time_basis: TimeBasis
    expanded: tuple[MarketSnapshot, ...]
    issues: tuple[RowIssue, ...]
    kept: list[MarketSnapshot] = field(default_factory=list)
    conflicts: list[KeyConflict] = field(default_factory=list)

    @property
    def split(self) -> bool:
        return len(self.expanded) > 1

    @property
    def deleted(self) -> bool:
        """Every market's key is held elsewhere, so the snapshot goes (its payload stays)."""
        return bool(self.expanded) and not self.kept

    def key_of(self, market: Market) -> SnapshotKey:
        return (
            self.snapshot.event_id,
            self.source,
            market,
            ensure_utc(self.snapshot.snapshot_time),
        )


def plan_snapshot(snapshot: StoredSnapshot) -> SnapshotPlan:
    source, time_basis = classify(snapshot)
    expanded = expand_envelope(snapshot.raw_data, source, snapshot.home_team, snapshot.away_team)
    return SnapshotPlan(snapshot, source, time_basis, expanded.markets, expanded.issues)


def assign_keys(plans: Sequence[SnapshotPlan], held: dict[SnapshotKey, int]) -> None:
    """Give each planned market its snapshot key unless ``held`` has it already; ``held`` takes
    every key claimed. Plans claim in id order, so of legacy snapshots sharing a key the first
    stored keeps it, as ``store_snapshot`` keeps the stored snapshot over a later write."""
    for plan in sorted(plans, key=lambda p: p.snapshot.id):
        for market in plan.expanded:
            key = plan.key_of(market.market)
            holder = held.get(key)
            if holder is None:
                held[key] = plan.snapshot.id
                plan.kept.append(market)
                continue
            plan.conflicts.append(
                KeyConflict(
                    plan.snapshot.id, key[0], plan.source, market.market, key[3], held_by=holder
                )
            )


@dataclass
class SnapshotBackfillReport:
    shaped: int = 0
    by_kind: Counter[tuple[Source, TimeBasis]] = field(default_factory=Counter)
    by_market: Counter[tuple[Source, Market]] = field(default_factory=Counter)
    empty: int = 0
    split: int = 0
    new_snapshots: int = 0
    prices: int = 0
    payloads: int = 0
    payload_only: int = 0
    deleted: int = 0
    odds_moved: int = 0
    odds_deleted: int = 0
    odds_unplaced: int = 0
    issues: Counter[str] = field(default_factory=Counter)
    unmatched_outcome_events: Counter[tuple[str, Source]] = field(default_factory=Counter)
    conflicts: list[KeyConflict] = field(default_factory=list)
    referenced: list[int] = field(default_factory=list)

    def merge(self, other: SnapshotBackfillReport) -> None:
        for name in (
            "shaped",
            "empty",
            "split",
            "new_snapshots",
            "prices",
            "payloads",
            "payload_only",
            "deleted",
            "odds_moved",
            "odds_deleted",
            "odds_unplaced",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.by_kind.update(other.by_kind)
        self.by_market.update(other.by_market)
        self.issues.update(other.issues)
        self.unmatched_outcome_events.update(other.unmatched_outcome_events)
        self.conflicts.extend(other.conflicts)
        self.referenced.extend(other.referenced)

    def count_plan(self, plan: SnapshotPlan) -> None:
        self.payloads += 1
        if plan.deleted:
            self.deleted += 1
        else:
            self.shaped += 1
            self.by_kind[(plan.source, plan.time_basis)] += 1
        if not plan.expanded:
            self.empty += 1
        if plan.split and plan.kept:
            self.split += 1
            self.new_snapshots += len(plan.kept) - 1
        for market in plan.kept:
            self.by_market[(plan.source, market.market)] += 1
            self.prices += len(market.prices)
        for issue in plan.issues:
            self.issues[issue.issue_type.value] += 1
            if issue.issue_type.value == "unmatched_outcome":
                self.unmatched_outcome_events[(plan.snapshot.event_id, plan.source)] += 1
        self.issues[KEY_CONFLICT_ISSUE] += len(plan.conflicts)
        self.conflicts.extend(plan.conflicts)


async def capture_cutoff(session: AsyncSession) -> datetime:
    """The database's clock: the backfill and verify only touch snapshots stored before it."""
    now = (await session.execute(select(func.now()))).scalar_one()
    await session.rollback()
    return ensure_utc(now)


async def load_snapshots(
    session: AsyncSession,
    *,
    cutoff: datetime,
    after_id: int,
    size: int,
    pending_only: bool,
    lock: bool,
) -> list[StoredSnapshot]:
    """Snapshots stored before ``cutoff`` with an id above ``after_id``, in id order. With
    ``pending_only``, only those the backfill has still to fill: no source, or no payload."""
    later = aliased(OddsSnapshot)
    untagged_fduk = and_(
        OddsSnapshot.raw_data["source"].as_string() == SnapshotSource.FOOTBALL_DATA_UK.value,  # type: ignore[index]
        OddsSnapshot.raw_data["timing"].is_(None),  # type: ignore[index]
    )
    has_later = exists().where(
        later.event_id == OddsSnapshot.event_id,
        later.raw_data["source"].as_string() == SnapshotSource.FOOTBALL_DATA_UK.value,  # type: ignore[index]
        later.raw_data["timing"].is_(None),  # type: ignore[index]
        later.snapshot_time > OddsSnapshot.snapshot_time,  # type: ignore[operator]
    )
    query = (
        select(
            OddsSnapshot.id,
            OddsSnapshot.event_id,
            OddsSnapshot.snapshot_time,
            OddsSnapshot.created_at,
            OddsSnapshot.raw_data,
            OddsSnapshot.api_request_id,
            OddsSnapshot.fetch_tier,
            OddsSnapshot.hours_until_commence,
            Event.home_team,
            Event.away_team,
            OddsSnapshot.source,
            OddsSnapshot.market,
            OddsSnapshot.time_basis,
            OddsSnapshot.payload_id,
            OddsSnapshot.in_play,
            OddsSnapshot.market_status,
            OddsSnapshot.total_matched,
            OddsSnapshot.market_start_time,
            case((untagged_fduk, has_later), else_=False).label("earlier_untagged_fduk"),
        )
        .join(Event, Event.id == OddsSnapshot.event_id)  # type: ignore[arg-type]
        .where(
            OddsSnapshot.created_at < cutoff,  # type: ignore[operator]
            OddsSnapshot.id > after_id,  # type: ignore[operator]
        )
        .order_by(OddsSnapshot.id)  # type: ignore[arg-type]
        .limit(size)
    )
    if pending_only:
        query = query.where(
            or_(OddsSnapshot.source.is_(None), OddsSnapshot.payload_id.is_(None))  # type: ignore[union-attr]
        )
    if lock:
        query = query.with_for_update(of=OddsSnapshot)  # type: ignore[arg-type]
    rows = (await session.execute(query)).all()
    return [
        StoredSnapshot(
            id=row.id,
            event_id=row.event_id,
            snapshot_time=row.snapshot_time,
            created_at=row.created_at,
            raw_data=row.raw_data if isinstance(row.raw_data, dict) else {},
            api_request_id=row.api_request_id,
            fetch_tier=row.fetch_tier,
            hours_until_commence=row.hours_until_commence,
            home_team=row.home_team,
            away_team=row.away_team,
            source=row.source,
            market=row.market,
            time_basis=row.time_basis,
            payload_id=row.payload_id,
            meta=MarketMeta(
                in_play=row.in_play,
                market_status=row.market_status,
                total_matched=row.total_matched,
                market_start_time=row.market_start_time,
            ),
            earlier_untagged_fduk=bool(row.earlier_untagged_fduk),
        )
        for row in rows
    ]


async def _held_keys(session: AsyncSession, keys: Sequence[SnapshotKey]) -> dict[SnapshotKey, int]:
    """The snapshots already stored under any of ``keys``."""
    if not keys:
        return {}
    columns = (
        OddsSnapshot.event_id,
        OddsSnapshot.source,
        OddsSnapshot.market,
        OddsSnapshot.snapshot_time,
    )
    wanted = [(e, s.value, m.value, t) for e, s, m, t in keys]
    held: dict[SnapshotKey, int] = {}
    for start in range(0, len(wanted), 2000):
        rows = await session.execute(
            select(*columns, OddsSnapshot.id).where(
                OddsSnapshot.source.is_not(None),  # type: ignore[union-attr]
                tuple_(*columns).in_(wanted[start : start + 2000]),
            )
        )
        for row in rows:
            key = (row.event_id, row.source, row.market, ensure_utc(row.snapshot_time))
            held[key] = min(held.get(key, row.id), row.id)
    return held


async def referenced_snapshot_ids(session: AsyncSession) -> set[int]:
    """Every snapshot id a candidate decision recorded as an input."""
    rows = await session.execute(
        select(func.unnest(CandidateDecision.input_snapshot_ids)).distinct()  # type: ignore[arg-type]
    )
    return {int(snapshot_id) for snapshot_id in rows.scalars() if snapshot_id is not None}


def _referenced_plans(plans: Sequence[SnapshotPlan], referenced: set[int]) -> list[int]:
    return [
        p.snapshot.id
        for p in plans
        if p.snapshot.id in referenced and (p.split or p.deleted or p.conflicts)
    ]


async def _plan_batch(
    session: AsyncSession,
    snapshots: Sequence[StoredSnapshot],
    held: dict[SnapshotKey, int],
) -> tuple[list[SnapshotPlan], list[StoredSnapshot]]:
    """Plans for the legacy snapshots, and the shaped ones that only lack a payload."""
    legacy = [s for s in snapshots if s.source is None]
    payload_only = [s for s in snapshots if s.source is not None]
    plans = [plan_snapshot(s) for s in legacy]
    keys = [plan.key_of(m.market) for plan in plans for m in plan.expanded]
    held.update(await _held_keys(session, keys))
    assign_keys(plans, held)
    return plans, payload_only


async def preflight_snapshots(
    session: AsyncSession, *, cutoff: datetime, batch_size: int, limit: int | None = None
) -> SnapshotBackfillReport:
    """What ``backfill_snapshots`` would write, counted without writing. Read-only."""
    report = SnapshotBackfillReport()
    referenced = await referenced_snapshot_ids(session)
    held: dict[SnapshotKey, int] = {}
    after_id = 0
    seen = 0
    while limit is None or seen < limit:
        size = batch_size if limit is None else min(batch_size, limit - seen)
        batch = await load_snapshots(
            session, cutoff=cutoff, after_id=after_id, size=size, pending_only=True, lock=False
        )
        if not batch:
            break
        plans, payload_only = await _plan_batch(session, batch, held)
        for plan in plans:
            report.count_plan(plan)
        report.payloads += len(payload_only)
        report.payload_only += len(payload_only)
        report.referenced.extend(_referenced_plans(plans, referenced))
        after_id = batch[-1].id
        seen += len(batch)
        await session.rollback()
    return report


@dataclass(frozen=True)
class SplitMapping:
    """One market of a split legacy snapshot and the snapshot that now holds it."""

    old_snapshot_id: int
    market: Market
    snapshot_id: int


async def backfill_snapshots(
    session: AsyncSession,
    *,
    cutoff: datetime,
    batch_size: int,
    limit: int | None = None,
    map_out: Path | None = None,
) -> SnapshotBackfillReport:
    """Fill every snapshot stored before ``cutoff`` that has no source or no payload, one
    committed batch at a time. A second run finds nothing left; a stopped run resumes.

    Each batch locks its rows, so a live writer re-timing one of them waits for that batch only.
    A batch that loses a snapshot key to a concurrent writer, or a deadlock to one, is rolled back
    and planned again.
    The split snapshots' old-to-new ids are appended to ``map_out`` as JSON lines.

    Raises:
        SplitSnapshotReferenced: before writing a batch in which a candidate decision's input
            would be split or deleted.
    """
    report = SnapshotBackfillReport()
    after_id = 0
    seen = 0
    while limit is None or seen < limit:
        size = batch_size if limit is None else min(batch_size, limit - seen)
        for attempt in range(BATCH_ATTEMPTS):
            try:
                batch = await load_snapshots(
                    session,
                    cutoff=cutoff,
                    after_id=after_id,
                    size=size,
                    pending_only=True,
                    lock=True,
                )
                if not batch:
                    await session.commit()
                    return report
                batch_report, mappings = await _backfill_batch(session, batch)
                await session.commit()
                break
            except DBAPIError as exc:
                await session.rollback()
                if getattr(exc.orig, "sqlstate", None) not in _RETRIED_SQLSTATES:
                    raise
                if attempt == BATCH_ATTEMPTS - 1:
                    raise
        report.merge(batch_report)
        if map_out is not None and mappings:
            with map_out.open("a") as out:
                for mapping in mappings:
                    out.write(
                        json.dumps(
                            {
                                "old_snapshot_id": mapping.old_snapshot_id,
                                "market": mapping.market.value,
                                "snapshot_id": mapping.snapshot_id,
                            }
                        )
                        + "\n"
                    )
        after_id = batch[-1].id
        seen += len(batch)
    return report


async def _backfill_batch(
    session: AsyncSession, batch: Sequence[StoredSnapshot]
) -> tuple[SnapshotBackfillReport, list[SplitMapping]]:
    report = SnapshotBackfillReport()
    plans, payload_only = await _plan_batch(session, batch, {})
    referenced = _referenced_plans(plans, await referenced_snapshot_ids(session))
    if referenced:
        raise SplitSnapshotReferenced(
            f"candidate decisions recorded snapshots {referenced}, which the backfill would split "
            "or delete; nothing in this batch was written"
        )
    payload_ids = await _store_payloads(session, [p.snapshot for p in plans] + list(payload_only))
    by_snapshot = dict(
        zip([p.snapshot.id for p in plans] + [s.id for s in payload_only], payload_ids, strict=True)
    )
    if payload_only:
        await session.execute(
            update(OddsSnapshot),
            [{"id": s.id, "payload_id": by_snapshot[s.id]} for s in payload_only],
        )
    report.payloads += len(payload_only)
    report.payload_only += len(payload_only)

    kept = [p for p in plans if not p.deleted]
    if kept:
        await session.execute(
            update(OddsSnapshot), [_parent_values(p, by_snapshot[p.snapshot.id]) for p in kept]
        )
    sibling_ids = await _insert_siblings(session, kept, by_snapshot)

    market_ids: dict[int, dict[Market, int]] = {}
    mappings: list[SplitMapping] = []
    for plan in kept:
        ids = {plan.kept[0].market: plan.snapshot.id} if plan.kept else {}
        for market in plan.kept[1:]:
            ids[market.market] = sibling_ids[(plan.snapshot.id, market.market)]
        market_ids[plan.snapshot.id] = ids
        if plan.split:
            mappings += [SplitMapping(plan.snapshot.id, m, i) for m, i in ids.items()]

    await _rehome_odds(session, plans, market_ids, report)
    deleted = [p.snapshot.id for p in plans if p.deleted]
    if deleted:
        await session.execute(delete(OddsSnapshot).where(OddsSnapshot.id.in_(deleted)))  # type: ignore[union-attr]

    price_rows = [
        {**price_values(spec), "snapshot_id": market_ids[plan.snapshot.id][market.market]}
        for plan in kept
        for market in plan.kept
        for spec in market.prices
    ]
    for start in range(0, len(price_rows), 5000):
        await session.execute(insert(Price), price_rows[start : start + 5000])

    issue_rows = [
        _issue_values(
            plan.snapshot.event_id, issue.issue_type.value, issue.description, issue.context
        )
        for plan in plans
        for issue in plan.issues
    ] + [
        _issue_values(
            conflict.event_id,
            KEY_CONFLICT_ISSUE,
            f"{conflict.source.value} {conflict.market.value} snapshot at "
            f"{conflict.snapshot_time.isoformat()} of legacy snapshot {conflict.snapshot_id} "
            f"not stored: snapshot {conflict.held_by} holds its key",
            {
                "source": conflict.source.value,
                "market": conflict.market.value,
                "snapshot_time": conflict.snapshot_time.isoformat(),
                "legacy_snapshot_id": conflict.snapshot_id,
                "kept_snapshot_id": conflict.held_by,
                "payload_id": by_snapshot[conflict.snapshot_id],
            },
        )
        for plan in plans
        for conflict in plan.conflicts
    ]
    if issue_rows:
        await session.execute(insert(DataQualityLog), issue_rows)

    for plan in plans:
        report.count_plan(plan)
    return report, mappings


async def _store_payloads(session: AsyncSession, snapshots: Sequence[StoredSnapshot]) -> list[int]:
    if not snapshots:
        return []
    result = await session.scalars(
        insert(SnapshotPayload).returning(SnapshotPayload.id, sort_by_parameter_order=True),  # type: ignore[arg-type]
        [{"payload": s.raw_data, "received_at": s.created_at} for s in snapshots],
    )
    return [int(i) for i in result.all()]


def _cut(plan: SnapshotPlan, market: Market) -> dict[str, Any]:
    raw_data = plan.snapshot.raw_data
    return market_envelope(raw_data, market) if plan.split else raw_data


def _market_values(market: MarketSnapshot | None) -> dict[str, Any]:
    meta = market.meta if market is not None else None
    return {
        "market": market.market if market is not None else None,
        "in_play": meta.in_play if meta else None,
        "market_status": meta.market_status if meta else None,
        "total_matched": meta.total_matched if meta else None,
        "market_start_time": meta.market_start_time if meta else None,
    }


def _parent_values(plan: SnapshotPlan, payload_id: int) -> dict[str, Any]:
    first = plan.kept[0] if plan.kept else None
    values: dict[str, Any] = {
        "id": plan.snapshot.id,
        "source": plan.source,
        "time_basis": plan.time_basis,
        "payload_id": payload_id,
        **_market_values(first),
    }
    if first is not None and plan.split:
        cut = _cut(plan, first.market)
        values["raw_data"] = cut
        values["bookmaker_count"] = len(cut.get("bookmakers") or [])
    return values


async def _insert_siblings(
    session: AsyncSession, plans: Sequence[SnapshotPlan], payloads: Mapping[int, int]
) -> dict[tuple[int, Market], int]:
    order: list[tuple[int, Market]] = []
    rows: list[dict[str, Any]] = []
    for plan in plans:
        snapshot = plan.snapshot
        for market in plan.kept[1:]:
            cut = _cut(plan, market.market)
            order.append((snapshot.id, market.market))
            rows.append(
                {
                    "event_id": snapshot.event_id,
                    "snapshot_time": snapshot.snapshot_time,
                    "raw_data": cut,
                    "bookmaker_count": len(cut.get("bookmakers") or []),
                    "api_request_id": snapshot.api_request_id,
                    "fetch_tier": snapshot.fetch_tier,
                    "hours_until_commence": snapshot.hours_until_commence,
                    "created_at": snapshot.created_at,
                    "source": plan.source,
                    "time_basis": plan.time_basis,
                    "payload_id": payloads[snapshot.id],
                    **_market_values(market),
                }
            )
    if not rows:
        return {}
    result = await session.scalars(
        insert(OddsSnapshot).returning(OddsSnapshot.id, sort_by_parameter_order=True),  # type: ignore[arg-type]
        rows,
    )
    return dict(zip(order, [int(i) for i in result.all()], strict=True))


def odds_markets(raw_data: Mapping[str, Any]) -> dict[tuple[str, str], Market]:
    """The market each ``(bookmaker key, market key)`` of an envelope's ``odds`` rows is stored
    under, as ``stored_market`` reads the envelope."""
    markets: dict[tuple[str, str], Market] = {}
    for entry in raw_data.get("bookmakers") or []:
        book = str(entry.get("key") or "").strip()
        for raw_market in entry.get("markets") or []:
            read = stored_market(raw_market)
            if read is not None:
                markets[(book, str(raw_market.get("key")))] = read[0]
    return markets


async def _rehome_odds(
    session: AsyncSession,
    plans: Sequence[SnapshotPlan],
    market_ids: Mapping[int, Mapping[Market, int]],
    report: SnapshotBackfillReport,
) -> None:
    """Move each ``odds`` row of a split snapshot to its market's snapshot, and delete those of
    markets whose key another snapshot holds."""
    moving = {p.snapshot.id: p for p in plans if p.split or p.conflicts}
    if not moving:
        return
    rows = (
        await session.execute(
            select(Odds.id, Odds.snapshot_id, Odds.bookmaker_key, Odds.market_key).where(
                Odds.snapshot_id.in_(list(moving))  # type: ignore[union-attr]
            )
        )
    ).all()
    index = {sid: odds_markets(plan.snapshot.raw_data) for sid, plan in moving.items()}
    moves: list[dict[str, int]] = []
    doomed: list[int] = []
    for row in rows:
        market = index[row.snapshot_id].get((str(row.bookmaker_key).strip(), row.market_key))
        if market is None:
            report.odds_unplaced += 1
            continue
        target = market_ids.get(row.snapshot_id, {}).get(market)
        if target is None:
            doomed.append(row.id)
        elif target != row.snapshot_id:
            moves.append({"id": row.id, "snapshot_id": target})
    deleted_ids = {p.snapshot.id for p in plans if p.deleted}
    if deleted_ids:
        # Rows the envelope index could not place still reference the deleted snapshot.
        placed = set(doomed)
        doomed += [r.id for r in rows if r.snapshot_id in deleted_ids and r.id not in placed]
    for start in range(0, len(moves), 5000):
        await session.execute(update(Odds), moves[start : start + 5000])
    if doomed:
        await session.execute(delete(Odds).where(Odds.id.in_(doomed)))  # type: ignore[union-attr]
    report.odds_moved += len(moves)
    report.odds_deleted += len(doomed)


def price_values(spec: PriceSpec) -> dict[str, Any]:
    """A ``prices`` row's columns, without its snapshot."""
    return {
        "bookmaker": spec.bookmaker,
        "outcome": spec.outcome,
        "line": spec.line,
        "side": spec.side,
        "price": spec.price,
        "size": spec.size,
        "matched": spec.matched,
        "changed_at": spec.changed_at,
    }


def _issue_values(
    event_id: str, issue_type: str, description: str, context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "severity": "warning",
        "issue_type": issue_type,
        "description": description,
        "raw_data": dict(context),
        # A bulk insert does not apply the model's defaults.
        "created_at": utc_now(),
    }
