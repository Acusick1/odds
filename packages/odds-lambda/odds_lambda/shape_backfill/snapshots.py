"""Fill the new-shape columns, payload and price rows of every snapshot stored before them.

A legacy snapshot (``source`` unset) is read with the live writers' reader
(``odds_lambda.price_rows.expand_envelope``) and takes its source, market, time basis, market
metadata and price rows. Its ``raw_data`` is archived as stored in one ``snapshot_payloads`` row.
An envelope quoting several markets (the Odds API) is split as the live writers split it: the
first market keeps the snapshot id, the others become new snapshots with ``raw_data`` cut to their
market, and each ``odds`` row moves to the snapshot of its market.

The work is planned by a read-only scan (``scan_snapshots``) before anything is written: the scan
is the preflight's count, decides which legacy snapshot keeps each contested key, and names every
snapshot the backfill would split or delete.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
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
from sqlalchemy import and_, case, delete, exists, func, insert, or_, select, text, tuple_, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from odds_lambda.price_rows import (
    ExpandedEnvelope,
    MarketMeta,
    MarketSnapshot,
    PriceSpec,
    RowIssueType,
    expand_envelope,
    market_envelope,
    stored_market,
)

# Odds API snapshots carry no tag telling a live fetch from a historical one. On production a
# live fetch is stored within seconds of its snapshot time, a historical one hours to months
# after it; nothing falls between 12 s and 2.8 h.
ODDS_API_HISTORICAL_GAP = timedelta(hours=1)
# How long a backfill statement waits for a row or lock a live writer holds before the batch
# gives up and is planned again; the live writer never waits behind the backfill for longer.
LOCK_TIMEOUT = "2s"
# A batch that loses a race with a live writer (a snapshot key, a deadlock or the lock timeout)
# is planned again.
BATCH_ATTEMPTS = 3
_RETRIED_SQLSTATES = frozenset({"23505", "40P01", "40001", "55P03"})

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
    """A snapshot as the backfill reads it, with its event's teams. ``raw_data`` is ``None``
    when the stored value is not a JSON object."""

    id: int
    event_id: str
    snapshot_time: datetime
    created_at: datetime
    raw_data: dict[str, Any] | None
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

    @property
    def envelope(self) -> dict[str, Any]:
        if self.raw_data is None:
            raise ValueError(f"snapshot {self.id} raw_data is not a JSON object")
        return self.raw_data

    @property
    def historical_oddsportal(self) -> bool:
        return self.raw_data is not None and (
            self.raw_data.get("source") == SnapshotSource.ODDSPORTAL.value
        )


def classify(snapshot: StoredSnapshot) -> tuple[Source, TimeBasis]:
    """The source and time basis of a legacy snapshot, read from its ``raw_data`` tags.

    Raises:
        UnknownSnapshotTag: for a tag or FDUK timing basis that names nothing known.
    """
    raw_data = snapshot.envelope
    tag = raw_data.get("source")
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
    timing = raw_data.get("timing")
    if timing is None:
        if snapshot.earlier_untagged_fduk:
            return Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND
        return Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE
    basis = timing.get("basis") if isinstance(timing, dict) else None
    if basis == FdukTimeBasis.KICKOFF:
        return Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE
    if basis == FdukTimeBasis.COLLECTION_DEADLINE:
        return Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND
    raise UnknownSnapshotTag(f"snapshot {snapshot.id} has FDUK timing {timing!r}")


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
    (the first on the legacy row), ``conflicts`` are markets another snapshot holds the key of.
    ``expanded`` is dropped once the scan has counted it."""

    snapshot_id: int
    event_id: str
    snapshot_time: datetime
    source: Source
    time_basis: TimeBasis
    markets: tuple[Market, ...]
    price_counts: dict[Market, int]
    issues: Counter[RowIssueType]
    historical_oddsportal: bool
    api_request_id: str | None
    expanded: ExpandedEnvelope | None
    kept: list[Market] = field(default_factory=list)
    conflicts: list[KeyConflict] = field(default_factory=list)

    @property
    def split(self) -> bool:
        return len(self.markets) > 1

    @property
    def deleted(self) -> bool:
        """Every market's key is held elsewhere, so the snapshot goes (its payload stays)."""
        return bool(self.markets) and not self.kept

    @property
    def rewritten(self) -> bool:
        """Its row is split, or it loses a market to another snapshot."""
        return self.split or bool(self.conflicts)

    def key_of(self, market: Market) -> SnapshotKey:
        return (self.event_id, self.source, market, ensure_utc(self.snapshot_time))

    def market_snapshot(self, market: Market) -> MarketSnapshot:
        assert self.expanded is not None
        return next(m for m in self.expanded.markets if m.market is market)


def plan_snapshot(snapshot: StoredSnapshot) -> SnapshotPlan:
    source, time_basis = classify(snapshot)
    expanded = expand_envelope(snapshot.envelope, source, snapshot.home_team, snapshot.away_team)
    return SnapshotPlan(
        snapshot_id=snapshot.id,
        event_id=snapshot.event_id,
        snapshot_time=snapshot.snapshot_time,
        source=source,
        time_basis=time_basis,
        markets=tuple(m.market for m in expanded.markets),
        price_counts={m.market: len(m.prices) for m in expanded.markets},
        issues=Counter(issue.issue_type for issue in expanded.issues),
        historical_oddsportal=snapshot.historical_oddsportal,
        api_request_id=snapshot.api_request_id,
        expanded=expanded,
    )


def pair_preferences(
    claims: Mapping[SnapshotKey, Sequence[SnapshotPlan]],
) -> dict[SnapshotKey, int]:
    """The legacy snapshot that keeps each contested key when it is not the first stored.

    The historical OddsPortal loader stores a game's opening and then its closing under one
    ``api_request_id``, one market each, so when exactly two such snapshots of one market share
    a key the later is the closing. Its prices are the state at that time, so it keeps the key.
    Any other group (three or more snapshots, other sources, several markets, differing request
    ids) keeps the first stored."""
    preferred: dict[SnapshotKey, int] = {}
    for key, plans in claims.items():
        if len(plans) != 2:
            continue
        opening, closing = sorted(plans, key=lambda p: p.snapshot_id)
        if (
            opening.historical_oddsportal
            and closing.historical_oddsportal
            and opening.api_request_id == closing.api_request_id
            and opening.markets == closing.markets == (key[2],)
        ):
            preferred[key] = closing.snapshot_id
    return preferred


def assign_keys(
    plans: Sequence[SnapshotPlan],
    stored: Mapping[SnapshotKey, int],
    claimed: dict[SnapshotKey, int],
    preferred: Mapping[SnapshotKey, int],
) -> None:
    """Give each planned market its snapshot key unless another snapshot keeps it: a stored
    snapshot (as ``store_snapshot`` keeps the stored one), the ``preferred`` legacy snapshot,
    else the first legacy snapshot in id order. ``claimed`` takes every key a plan keeps."""
    for plan in sorted(plans, key=lambda p: p.snapshot_id):
        for market in plan.markets:
            key = plan.key_of(market)
            holder = stored.get(key)
            if holder is None:
                choice = preferred.get(key)
                holder = choice if choice not in (None, plan.snapshot_id) else claimed.get(key)
            if holder is None:
                claimed[key] = plan.snapshot_id
                plan.kept.append(market)
                continue
            plan.conflicts.append(
                KeyConflict(plan.snapshot_id, key[0], plan.source, market, key[3], held_by=holder)
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
    issues: Counter[RowIssueType] = field(default_factory=Counter)
    unmatched_outcome_events: Counter[tuple[str, Source]] = field(default_factory=Counter)
    conflicts: list[KeyConflict] = field(default_factory=list)
    referenced: list[int] = field(default_factory=list)
    malformed: list[int] = field(default_factory=list)
    # Snapshots a live writer held locked on every pass, left for a rerun.
    pending_left: int = 0

    def merge(self, other: SnapshotBackfillReport) -> None:
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if isinstance(mine, Counter):
                mine.update(theirs)
            elif isinstance(mine, list):
                mine.extend(theirs)
            elif f.name != "pending_left":
                setattr(self, f.name, mine + theirs)

    def count_plan(self, plan: SnapshotPlan) -> None:
        self.payloads += 1
        if plan.deleted:
            self.deleted += 1
        else:
            self.shaped += 1
            self.by_kind[(plan.source, plan.time_basis)] += 1
        if not plan.markets:
            self.empty += 1
        if plan.split and plan.kept:
            self.split += 1
            self.new_snapshots += len(plan.kept) - 1
        for market in plan.kept:
            self.by_market[(plan.source, market)] += 1
            self.prices += plan.price_counts[market]
        self.issues.update(plan.issues)
        if plan.issues[RowIssueType.UNMATCHED_OUTCOME]:
            self.unmatched_outcome_events[(plan.event_id, plan.source)] += plan.issues[
                RowIssueType.UNMATCHED_OUTCOME
            ]
        self.issues[RowIssueType.SNAPSHOT_KEY_CONFLICT] += len(plan.conflicts)
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
    ids: Sequence[int] | None = None,
) -> list[StoredSnapshot]:
    """Snapshots stored before ``cutoff`` with an id above ``after_id`` (or among ``ids``), in id
    order. With ``pending_only``, only those the backfill has still to fill: no source, or no
    payload. With ``lock``, the rows are locked, skipping any a live writer holds."""
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
        .where(OddsSnapshot.created_at < cutoff)  # type: ignore[operator]
        .order_by(OddsSnapshot.id)  # type: ignore[arg-type]
    )
    if ids is not None:
        query = query.where(OddsSnapshot.id.in_(list(ids)))  # type: ignore[union-attr]
    else:
        query = query.where(OddsSnapshot.id > after_id).limit(size)  # type: ignore[operator]
    if pending_only:
        query = query.where(_pending())
    if lock:
        query = query.with_for_update(of=OddsSnapshot, skip_locked=True)  # type: ignore[arg-type]
    rows = (await session.execute(query)).all()
    return [
        StoredSnapshot(
            id=row.id,
            event_id=row.event_id,
            snapshot_time=row.snapshot_time,
            created_at=row.created_at,
            raw_data=row.raw_data if isinstance(row.raw_data, dict) else None,
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


async def _stored_keys(
    session: AsyncSession, keys: Sequence[SnapshotKey]
) -> dict[SnapshotKey, int]:
    """The shaped snapshots already stored under any of ``keys``."""
    columns = (
        OddsSnapshot.event_id,
        OddsSnapshot.source,
        OddsSnapshot.market,
        OddsSnapshot.snapshot_time,
    )
    wanted = [(e, s.value, m.value, t) for e, s, m, t in keys]
    stored: dict[SnapshotKey, int] = {}
    for start in range(0, len(wanted), 2000):
        rows = await session.execute(
            select(*columns, OddsSnapshot.id).where(
                OddsSnapshot.source.is_not(None),  # type: ignore[union-attr]
                tuple_(*columns).in_(wanted[start : start + 2000]),
            )
        )
        for row in rows:
            key = (row.event_id, row.source, row.market, ensure_utc(row.snapshot_time))
            stored[key] = min(stored.get(key, row.id), row.id)
    return stored


async def referenced_snapshot_ids(session: AsyncSession) -> set[int]:
    """Every snapshot id a candidate decision recorded as an input."""
    rows = await session.execute(
        select(func.unnest(CandidateDecision.input_snapshot_ids)).distinct()  # type: ignore[arg-type]
    )
    return {int(snapshot_id) for snapshot_id in rows.scalars()}


@dataclass(frozen=True)
class OddsAction:
    """What happens to one ``odds`` row of a rewritten snapshot: it moves to ``market``'s
    snapshot, or is deleted when ``market`` is ``None``."""

    odds_id: int
    snapshot_id: int
    market: Market | None


def odds_markets(raw_data: Mapping[str, Any]) -> dict[tuple[str, str], Market]:
    """The market each ``(bookmaker key, market key)`` of an envelope's ``odds`` rows is stored
    under, as ``stored_market`` reads the envelope; a book without a key has no rows."""
    markets: dict[tuple[str, str], Market] = {}
    for entry in raw_data.get("bookmakers") or []:
        if not entry.get("key"):
            continue
        book = str(entry["key"]).strip()
        for raw_market in entry.get("markets") or []:
            read = stored_market(raw_market)
            if read is not None:
                markets[(book, str(raw_market.get("key")))] = read[0]
    return markets


async def odds_actions(
    session: AsyncSession,
    plans: Mapping[int, SnapshotPlan],
    envelopes: Mapping[int, dict[str, Any]],
) -> tuple[list[OddsAction], int]:
    """The moves and deletions of the ``odds`` rows of rewritten snapshots, and the count of rows
    no stored market of their envelope takes (left on a kept snapshot)."""
    if not plans:
        return [], 0
    rows = (
        await session.execute(
            select(Odds.id, Odds.snapshot_id, Odds.bookmaker_key, Odds.market_key).where(
                Odds.snapshot_id.in_(list(plans))  # type: ignore[union-attr]
            )
        )
    ).all()
    index = {sid: odds_markets(envelopes[sid]) for sid in plans}
    actions: list[OddsAction] = []
    unplaced = 0
    for row in rows:
        plan = plans[row.snapshot_id]
        market = index[row.snapshot_id].get((str(row.bookmaker_key).strip(), row.market_key))
        if plan.deleted:
            actions.append(OddsAction(row.id, row.snapshot_id, None))
        elif market is None:
            unplaced += 1
        elif market not in plan.kept:
            actions.append(OddsAction(row.id, row.snapshot_id, None))
        elif market is not plan.kept[0]:
            actions.append(OddsAction(row.id, row.snapshot_id, market))
    return actions, unplaced


def _count_odds(
    report: SnapshotBackfillReport, actions: Sequence[OddsAction], unplaced: int
) -> None:
    report.odds_moved += sum(1 for a in actions if a.market is not None)
    report.odds_deleted += sum(1 for a in actions if a.market is None)
    report.odds_unplaced += unplaced


@dataclass
class SnapshotScan:
    """The read-only plan of the snapshot step: its counts, the legacy snapshot that keeps each
    contested key, and the snapshots it would rewrite (split, or losing a market)."""

    cutoff: datetime
    report: SnapshotBackfillReport
    preferred: dict[SnapshotKey, int]
    rewritten: set[int]


async def scan_snapshots(
    session: AsyncSession, *, cutoff: datetime, batch_size: int
) -> SnapshotScan:
    """What ``backfill_snapshots`` will write, counted without writing, over every pending
    snapshot (a contested key is decided over all its snapshots). Read-only."""
    report = SnapshotBackfillReport(malformed=await malformed_snapshot_ids(session, cutoff))
    plans: list[SnapshotPlan] = []
    stored: dict[SnapshotKey, int] = {}
    after_id = 0
    while True:
        batch = await load_snapshots(
            session,
            cutoff=cutoff,
            after_id=after_id,
            size=batch_size,
            pending_only=True,
            lock=False,
        )
        if not batch:
            break
        planned, payload_only = _plan(batch)
        for plan in planned:
            plan.expanded = None
        plans += planned
        report.payloads += len(payload_only)
        report.payload_only += len(payload_only)
        stored.update(
            await _stored_keys(session, [p.key_of(m) for p in planned for m in p.markets])
        )
        after_id = batch[-1].id
        await session.rollback()

    claims: dict[SnapshotKey, list[SnapshotPlan]] = defaultdict(list)
    for plan in plans:
        for market in plan.markets:
            key = plan.key_of(market)
            if key not in stored:
                claims[key].append(plan)
    preferred = pair_preferences({k: v for k, v in claims.items() if len(v) > 1})
    assign_keys(plans, stored, {}, preferred)
    for plan in plans:
        report.count_plan(plan)

    rewritten = {p.snapshot_id: p for p in plans if p.rewritten}
    ids = sorted(rewritten)
    for start in range(0, len(ids), batch_size):
        chunk = await load_snapshots(
            session,
            cutoff=cutoff,
            after_id=0,
            size=0,
            pending_only=False,
            lock=False,
            ids=ids[start : start + batch_size],
        )
        actions, unplaced = await odds_actions(
            session, {s.id: rewritten[s.id] for s in chunk}, {s.id: s.envelope for s in chunk}
        )
        _count_odds(report, actions, unplaced)
        await session.rollback()
    report.referenced = sorted(set(rewritten) & await referenced_snapshot_ids(session))
    await session.rollback()
    return SnapshotScan(cutoff, report, preferred, set(rewritten))


def _plan(batch: Sequence[StoredSnapshot]) -> tuple[list[SnapshotPlan], list[StoredSnapshot]]:
    """Plans for the legacy snapshots, and the shaped ones that only lack a payload."""
    plans = [plan_snapshot(s) for s in batch if s.source is None]
    return plans, [s for s in batch if s.source is not None]


async def check_references(session: AsyncSession, scan: SnapshotScan) -> None:
    """Raise when a candidate decision recorded a snapshot the scan would rewrite.

    Raises:
        SplitSnapshotReferenced: naming those snapshots.
    """
    referenced = sorted(scan.rewritten & await referenced_snapshot_ids(session))
    await session.rollback()
    if referenced:
        raise SplitSnapshotReferenced(
            f"candidate decisions recorded snapshots {referenced}, which the backfill would "
            "split or delete; nothing was written"
        )


@dataclass(frozen=True)
class SplitMapping:
    """One market of a split legacy snapshot and the snapshot that now holds it."""

    old_snapshot_id: int
    market: Market
    snapshot_id: int


async def backfill_snapshots(
    session: AsyncSession,
    scan: SnapshotScan,
    *,
    batch_size: int,
    limit: int | None = None,
    map_out: Path | None = None,
) -> SnapshotBackfillReport:
    """Fill every snapshot stored before the scan's cutoff that has no source or no payload,
    one committed batch at a time, keeping contested keys as the scan decided. A second run
    finds nothing left; a stopped run resumes.

    A batch skips rows a live writer holds locked and waits at most ``LOCK_TIMEOUT`` for anything
    else; a batch that times out, deadlocks or loses a snapshot key to a writer is rolled back
    and planned again. The step makes passes until a pass shaped nothing; what is still pending
    then is reported as ``pending_left``. Split snapshots' old and new ids are appended to
    ``map_out`` as JSON lines.

    Raises:
        SplitSnapshotReferenced: before writing anything, when a candidate decision recorded a
            snapshot the scan would rewrite, and before a batch in which a decision recorded
            during the run would be rewritten.
    """
    await check_references(session, scan)
    report = SnapshotBackfillReport(malformed=await malformed_snapshot_ids(session, scan.cutoff))
    seen = 0
    while True:
        after_id = 0
        progress = False
        while limit is None or seen < limit:
            size = batch_size if limit is None else min(batch_size, limit - seen)
            done = await _backfill_with_retries(session, scan, after_id, size)
            if done is None:
                break
            batch, batch_report, mappings = done
            report.merge(batch_report)
            _write_map(map_out, mappings)
            after_id = batch[-1].id
            seen += len(batch)
            progress = True
        pending = await _pending_count(session, scan.cutoff)
        if not pending or not progress or (limit is not None and seen >= limit):
            report.pending_left = pending
            return report


async def _backfill_with_retries(
    session: AsyncSession, scan: SnapshotScan, after_id: int, size: int
) -> tuple[list[StoredSnapshot], SnapshotBackfillReport, list[SplitMapping]] | None:
    for attempt in range(BATCH_ATTEMPTS):
        try:
            await session.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
            batch = await load_snapshots(
                session,
                cutoff=scan.cutoff,
                after_id=after_id,
                size=size,
                pending_only=True,
                lock=True,
            )
            if not batch:
                await session.commit()
                return None
            batch_report, mappings = await _backfill_batch(session, scan, batch)
            await session.commit()
            return batch, batch_report, mappings
        except DBAPIError as exc:
            await session.rollback()
            if getattr(exc.orig, "sqlstate", None) not in _RETRIED_SQLSTATES:
                raise
            if attempt == BATCH_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


def _pending() -> Any:
    """A snapshot the backfill has still to fill: no source, or no payload. One whose
    ``raw_data`` is not a JSON object is never pending; ``malformed_snapshot_ids`` reports it."""
    return and_(
        or_(OddsSnapshot.source.is_(None), OddsSnapshot.payload_id.is_(None)),  # type: ignore[union-attr]
        func.json_typeof(OddsSnapshot.raw_data) == "object",
    )


async def _pending_count(session: AsyncSession, cutoff: datetime) -> int:
    count = (
        await session.execute(
            select(func.count()).where(
                OddsSnapshot.created_at < cutoff,  # type: ignore[operator]
                _pending(),
            )
        )
    ).scalar_one()
    await session.rollback()
    return int(count)


async def malformed_snapshot_ids(session: AsyncSession, cutoff: datetime) -> list[int]:
    """Legacy snapshots stored before ``cutoff`` whose ``raw_data`` is not a JSON object, which
    no reader can expand; the backfill leaves them and reports them."""
    rows = await session.execute(
        select(OddsSnapshot.id)
        .where(
            OddsSnapshot.created_at < cutoff,  # type: ignore[operator]
            OddsSnapshot.source.is_(None),  # type: ignore[union-attr]
            func.coalesce(func.json_typeof(OddsSnapshot.raw_data), "null") != "object",
        )
        .order_by(OddsSnapshot.id)  # type: ignore[arg-type]
    )
    ids = [int(i) for i in rows.scalars()]
    await session.rollback()
    return ids


def _write_map(map_out: Path | None, mappings: Sequence[SplitMapping]) -> None:
    if map_out is None or not mappings:
        return
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


async def _backfill_batch(
    session: AsyncSession, scan: SnapshotScan, batch: Sequence[StoredSnapshot]
) -> tuple[SnapshotBackfillReport, list[SplitMapping]]:
    report = SnapshotBackfillReport()
    plans, payload_only = _plan(batch)
    stored = await _stored_keys(session, [p.key_of(m) for p in plans for m in p.markets])
    assign_keys(plans, stored, {}, scan.preferred)
    referenced = sorted(
        {p.snapshot_id for p in plans if p.rewritten} & await referenced_snapshot_ids(session)
    )
    if referenced:
        raise SplitSnapshotReferenced(
            f"candidate decisions recorded snapshots {referenced}, which the backfill would split "
            "or delete; nothing in this batch was written"
        )
    snapshots = {s.id: s for s in batch}
    payload_ids = await _store_payloads(
        session, [snapshots[p.snapshot_id] for p in plans] + list(payload_only)
    )
    by_snapshot = dict(
        zip([p.snapshot_id for p in plans] + [s.id for s in payload_only], payload_ids, strict=True)
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
            update(OddsSnapshot),
            [_parent_values(p, snapshots[p.snapshot_id], by_snapshot[p.snapshot_id]) for p in kept],
        )
    sibling_ids = await _insert_siblings(session, kept, snapshots, by_snapshot)

    market_ids: dict[int, dict[Market, int]] = {}
    mappings: list[SplitMapping] = []
    for plan in kept:
        ids = {plan.kept[0]: plan.snapshot_id} if plan.kept else {}
        for market in plan.kept[1:]:
            ids[market] = sibling_ids[(plan.snapshot_id, market)]
        market_ids[plan.snapshot_id] = ids
        if plan.split:
            mappings += [SplitMapping(plan.snapshot_id, m, i) for m, i in ids.items()]

    rewritten = {p.snapshot_id: p for p in plans if p.rewritten}
    actions, unplaced = await odds_actions(
        session, rewritten, {sid: snapshots[sid].envelope for sid in rewritten}
    )
    moves = [
        {"id": a.odds_id, "snapshot_id": market_ids[a.snapshot_id][a.market]}
        for a in actions
        if a.market is not None
    ]
    for start in range(0, len(moves), 5000):
        await session.execute(update(Odds), moves[start : start + 5000])
    doomed = [a.odds_id for a in actions if a.market is None]
    if doomed:
        await session.execute(delete(Odds).where(Odds.id.in_(doomed)))  # type: ignore[union-attr]
    _count_odds(report, actions, unplaced)
    deleted = [p.snapshot_id for p in plans if p.deleted]
    if deleted:
        await session.execute(delete(OddsSnapshot).where(OddsSnapshot.id.in_(deleted)))  # type: ignore[union-attr]

    price_rows = [
        {**price_values(spec), "snapshot_id": market_ids[plan.snapshot_id][market]}
        for plan in kept
        for market in plan.kept
        for spec in plan.market_snapshot(market).prices
    ]
    for start in range(0, len(price_rows), 5000):
        await session.execute(insert(Price), price_rows[start : start + 5000])

    issue_rows = [
        _issue_values(plan.event_id, issue.issue_type, issue.description, issue.context)
        for plan in plans
        if plan.expanded is not None
        for issue in plan.expanded.issues
    ] + [
        _issue_values(
            conflict.event_id,
            RowIssueType.SNAPSHOT_KEY_CONFLICT,
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
        [{"payload": s.envelope, "received_at": s.created_at} for s in snapshots],
    )
    return [int(i) for i in result.all()]


def _cut(plan: SnapshotPlan, snapshot: StoredSnapshot, market: Market) -> dict[str, Any]:
    return market_envelope(snapshot.envelope, market) if plan.split else snapshot.envelope


def _market_values(plan: SnapshotPlan, market: Market | None) -> dict[str, Any]:
    meta = plan.market_snapshot(market).meta if market is not None else MarketMeta()
    return {
        "market": market,
        "in_play": meta.in_play,
        "market_status": meta.market_status,
        "total_matched": meta.total_matched,
        "market_start_time": meta.market_start_time,
    }


def _parent_values(plan: SnapshotPlan, snapshot: StoredSnapshot, payload_id: int) -> dict[str, Any]:
    first = plan.kept[0] if plan.kept else None
    values: dict[str, Any] = {
        "id": plan.snapshot_id,
        "source": plan.source,
        "time_basis": plan.time_basis,
        "payload_id": payload_id,
        **_market_values(plan, first),
    }
    if first is not None and plan.split:
        cut = _cut(plan, snapshot, first)
        values["raw_data"] = cut
        values["bookmaker_count"] = len(cut["bookmakers"])
    return values


async def _insert_siblings(
    session: AsyncSession,
    plans: Sequence[SnapshotPlan],
    snapshots: Mapping[int, StoredSnapshot],
    payloads: Mapping[int, int],
) -> dict[tuple[int, Market], int]:
    order: list[tuple[int, Market]] = []
    rows: list[dict[str, Any]] = []
    for plan in plans:
        snapshot = snapshots[plan.snapshot_id]
        for market in plan.kept[1:]:
            cut = _cut(plan, snapshot, market)
            order.append((snapshot.id, market))
            rows.append(
                {
                    "event_id": snapshot.event_id,
                    "snapshot_time": snapshot.snapshot_time,
                    "raw_data": cut,
                    "bookmaker_count": len(cut["bookmakers"]),
                    "api_request_id": snapshot.api_request_id,
                    "fetch_tier": snapshot.fetch_tier,
                    "hours_until_commence": snapshot.hours_until_commence,
                    "created_at": snapshot.created_at,
                    "source": plan.source,
                    "time_basis": plan.time_basis,
                    "payload_id": payloads[snapshot.id],
                    **_market_values(plan, market),
                }
            )
    if not rows:
        return {}
    result = await session.scalars(
        insert(OddsSnapshot).returning(OddsSnapshot.id, sort_by_parameter_order=True),  # type: ignore[arg-type]
        rows,
    )
    return dict(zip(order, [int(i) for i in result.all()], strict=True))


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
    event_id: str, issue_type: RowIssueType, description: str, context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "severity": "warning",
        "issue_type": issue_type.value,
        "description": description,
        "raw_data": dict(context),
        # A bulk insert does not apply the model's defaults.
        "created_at": utc_now(),
    }
