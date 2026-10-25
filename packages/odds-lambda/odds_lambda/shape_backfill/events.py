"""Bring stored events into the one-event-per-match shape: merge twins, record the external ids
held in stored data, and move ``no_game_pk_reason`` to the UNMATCHED status."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from odds_core.event_ids import is_odds_api_event_id
from odds_core.event_source import canonical_event_id
from odds_core.models import Event, EventExternalId, HealthCheckAcknowledgement, OddsSnapshot
from odds_core.snapshot_sources import SnapshotSource
from odds_core.time import ensure_utc
from odds_core.vocabulary import HealthCheck, Source
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.oddsportal_common import event_id_from_match_link, parse_match_date
from odds_lambda.polymarket_matching import normalize_team
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY, settle_merged_kickoff
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.event_resolver import (
    IdDecision,
    ListedGame,
    MatchWindow,
    attach_events,
    check_external_id,
    event_lock_key,
    lock_fixture,
    record_external_id,
)
from odds_lambda.storage.event_unmatched import mark_unmatched

# Fixtures whose locks one transaction holds before it commits, so a live writer waiting on one
# waits for a few events only.
EVENTS_PER_COMMIT = 50
# Rows a merge deletes that record what a candidate did; deleting them needs --allow-merge-drops.
DECISION_TABLES: frozenset[str] = frozenset({"candidate_decisions", "paper_trades"})

_TAG_SOURCES: dict[str | None, Source] = {
    None: Source.ODDS_API,
    SnapshotSource.ODDSPORTAL.value: Source.ODDSPORTAL,
    SnapshotSource.ODDSPORTAL_LIVE.value: Source.ODDSPORTAL,
    SnapshotSource.BETFAIR_API.value: Source.BETFAIR,
    SnapshotSource.FOOTBALL_DATA_UK.value: Source.FOOTBALL_DATA_UK,
    SnapshotSource.POLYMARKET.value: Source.POLYMARKET,
}


class MergeDropsRefused(RuntimeError):
    """A twin merge would delete candidate decisions or paper trades."""


@dataclass(frozen=True)
class TwinMerge:
    """One twin group: the event kept, the events merged into it, its kick-off, the rows that
    move to the kept event and the rows the merge deletes, per table."""

    kept_id: str
    merged_ids: tuple[str, ...]
    kickoff_source: Source | None
    commence_from: datetime
    commence_to: datetime
    rows_moved: dict[str, int] = field(default_factory=dict)
    rows_deleted: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def drops_decisions(self) -> bool:
        return any(self.rows_deleted.get(table) for table in DECISION_TABLES)


async def observed_sources(session: AsyncSession, event: Event) -> set[Source]:
    """The sources that hold an id on the event or stored a snapshot of it."""
    held = await session.execute(
        select(EventExternalId.source).where(EventExternalId.event_id == event.surrogate_id)
    )
    sources = set(held.scalars())
    if event.mlb_game_pk is not None:
        sources.add(Source.MLB_STATS_API)
    stored = await session.execute(
        select(OddsSnapshot.source, OddsSnapshot.raw_data["source"].as_string())  # type: ignore[index]
        .where(OddsSnapshot.event_id == event.id)
        .distinct()
    )
    for source, tag in stored.all():
        if source is not None:
            sources.add(source)
        elif tag in _TAG_SOURCES:
            sources.add(_TAG_SOURCES[tag])
    return sources


def kickoff_starts(
    group: Sequence[Event], sources: Mapping[str, set[Source]]
) -> dict[Source, datetime]:
    """For each source that observed a twin, that twin's stored kick-off (the twin stored first
    when several did)."""
    starts: dict[Source, datetime] = {}
    for event in sorted(group, key=lambda e: _stored_key(e)):
        for source in sources[event.id]:
            starts.setdefault(source, ensure_utc(event.commence_time))
    return starts


def _stored_key(event: Event) -> int:
    if event.surrogate_id is None:
        raise ValueError(f"event {event.id} is not stored")
    return event.surrogate_id


async def plan_twin_merge(merger: EventMerger, group: Sequence[Event]) -> TwinMerge:
    """Keep the canonical event and give it the kick-off of the highest-ranked source in its
    sport's ``KICKOFF_AUTHORITY`` order that observed a twin (``settle_merged_kickoff``); count
    the rows that would move and those the merge would delete. Read-only."""
    keep_id = canonical_event_id([e.id for e in group])
    keep = next(e for e in group if e.id == keep_id)
    duplicates = [e for e in group if e.id != keep_id]
    sources = {e.id: await observed_sources(merger.session, e) for e in group}
    starts = kickoff_starts(group, sources)
    order = KICKOFF_AUTHORITY.get(keep.sport_key, ())  # type: ignore[call-overload]
    source = next((s for s in order if s in starts), None)
    moved: dict[str, int] = {}
    for duplicate in duplicates:
        for table, count in (await merger.reference_counts(duplicate.id)).items():
            moved[table] = moved.get(table, 0) + count
    return TwinMerge(
        kept_id=keep_id,
        merged_ids=tuple(sorted(e.id for e in duplicates)),
        kickoff_source=source,
        commence_from=ensure_utc(keep.commence_time),
        commence_to=starts[source] if source is not None else ensure_utc(keep.commence_time),
        rows_moved=moved,
        rows_deleted=await merger.merge_drops(keep, duplicates),
    )


async def plan_twin_merges(session: AsyncSession) -> list[TwinMerge]:
    """``plan_twin_merge`` for every twin group (same sport and teams within ±2 h). Read-only."""
    merger = EventMerger(session)
    plans = [await plan_twin_merge(merger, group) for group in await merger.find_twin_groups()]
    await session.rollback()
    return plans


async def merge_twins(
    session: AsyncSession, plans: Sequence[TwinMerge], *, allow_drops: bool
) -> list[TwinMerge]:
    """Merge the planned twin groups with ``EventMerger``, its clash handling included, each
    under its fixture lock in its own transaction, and settle each kept event's kick-off through
    ``settle_merged_kickoff``. A group that fails (different ``gamePk``s, a snapshot key both
    events hold) is rolled back and reported with its error; the rest go ahead.

    Raises:
        MergeDropsRefused: before any merge, when a planned merge would delete candidate
            decisions or paper trades and ``allow_drops`` is not set.
    """
    if not allow_drops and any(plan.drops_decisions for plan in plans):
        raise MergeDropsRefused(
            "twin merges would delete candidate decisions or paper trades: "
            + "; ".join(
                f"{p.kept_id} <- {p.merged_ids}: {p.rows_deleted}"
                for p in plans
                if p.drops_decisions
            )
        )
    merger = EventMerger(session)
    merges: list[TwinMerge] = []
    for planned in plans:
        ids = [planned.kept_id, *planned.merged_ids]
        try:
            events = list(
                (await session.execute(select(Event).where(Event.id.in_(ids)))).scalars()  # type: ignore[attr-defined]
            )
            if len(events) < 2:
                await session.rollback()
                continue
            await lock_fixture(session, events[0])
            for event in events:
                await session.refresh(event)
            plan = await plan_twin_merge(merger, events)
            if plan.drops_decisions and not allow_drops:
                raise MergeDropsRefused(f"{plan.kept_id} <- {plan.merged_ids}: {plan.rows_deleted}")
            sources = {e.id: await observed_sources(session, e) for e in events}
            starts = kickoff_starts(events, sources)
            keep = next(e for e in events if e.id == plan.kept_id)
            report = await merger.merge(keep, [e for e in events if e.id != plan.kept_id])
            await settle_merged_kickoff(session, keep, starts, now=datetime.now(UTC))
            await session.commit()
            merges.append(
                replace(plan, rows_moved=report.rows_moved, rows_deleted=report.rows_deleted)
            )
        except MergeDropsRefused:
            await session.rollback()
            raise
        except Exception as exc:  # noqa: BLE001 - reported per group, the others go ahead
            await session.rollback()
            merges.append(replace(planned, error=repr(exc)))
    return merges


@dataclass
class UnmatchedReport:
    marked: list[str] = field(default_factory=list)
    already: int = 0


async def migrate_unmatched(session: AsyncSession, *, apply: bool) -> UnmatchedReport:
    """Give every event carrying ``no_game_pk_reason`` the UNMATCHED status and its reason as a
    ``missing_mlb_game_pk`` acknowledgement (``mark_unmatched``; a final event keeps its status).
    An event already acknowledged with the same reason is left alone. The column is left in
    place. Without ``apply`` nothing is written."""
    report = UnmatchedReport()
    events = [
        event
        for event in (
            await session.execute(
                select(Event).where(Event.no_game_pk_reason.is_not(None)).order_by(Event.id)  # type: ignore[union-attr]
            )
        ).scalars()
        if event.no_game_pk_reason is not None
    ]
    acknowledged = {
        (row.event_id, row.reason)
        for row in (
            await session.execute(
                select(HealthCheckAcknowledgement).where(
                    HealthCheckAcknowledgement.check_name == HealthCheck.MISSING_MLB_GAME_PK
                )
            )
        ).scalars()
    }
    for event in sorted(
        events, key=lambda e: event_lock_key(e.sport_key, e.home_team, e.away_team)
    ):
        assert event.no_game_pk_reason is not None
        reason = event.no_game_pk_reason.strip()
        if (event.surrogate_id, reason) in acknowledged:
            report.already += 1
            continue
        report.marked.append(event.id)
        if apply:
            await lock_fixture(session, event)
            await mark_unmatched(session, event, reason)
            await session.commit()
    await session.rollback()
    return report


class IdOutcome(StrEnum):
    NEW = "new"
    HELD = "held"
    # The event holds another id from the source, or another event holds this one.
    REFUSED = "refused"
    # The event already holds an Odds API id; the API reissues ids and the stored one is newer.
    KEPT_CURRENT = "kept_current"


@dataclass(frozen=True)
class StoredId:
    """An id a source issued for an event, read from stored data."""

    event_id: str
    source: Source
    external_id: str


_STORED_ID_QUERIES: dict[Source, str] = {
    # The Odds API reissues ids, so only the latest one an event's snapshots carry.
    Source.ODDS_API: """
        SELECT DISTINCT ON (event_id) event_id, raw_data->>'id' AS external_id
        FROM odds_snapshots
        WHERE raw_data->>'source' IS NULL AND raw_data->>'id' IS NOT NULL
        ORDER BY event_id, snapshot_time DESC, id DESC
    """,
    Source.ODDSPORTAL: """
        SELECT event_id, raw_data->>'oddsportal_event_id' AS external_id
        FROM odds_snapshots WHERE raw_data->>'oddsportal_event_id' IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, min(snapshot_time)
    """,
    Source.BETFAIR: """
        SELECT event_id, raw_data->>'betfair_event_id' AS external_id
        FROM odds_snapshots WHERE raw_data->>'betfair_event_id' IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, min(snapshot_time)
    """,
    Source.POLYMARKET: """
        SELECT event_id, external_id FROM (
            SELECT event_id,
                   raw_data->'bookmakers'->0->'polymarket_meta'->>'pm_event_id' AS external_id,
                   snapshot_time AS seen
            FROM odds_snapshots WHERE raw_data->>'source' = 'polymarket'
            UNION ALL
            SELECT event_id, pm_event_id, created_at FROM polymarket_events
            WHERE event_id IS NOT NULL
        ) ids WHERE external_id IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1, min(seen)
    """,
    Source.MLB_STATS_API: """
        SELECT id AS event_id, mlb_game_pk::text AS external_id FROM events
        WHERE mlb_game_pk IS NOT NULL
    """,
}


async def stored_external_ids(session: AsyncSession) -> list[StoredId]:
    """Every id a source issued that stored data holds: the Odds API id in Odds API envelopes
    (else the key of an event keyed by it), the OddsPortal live and Betfair event ids in their
    envelopes, Polymarket event ids in Polymarket envelopes and the legacy ``polymarket_events``,
    and ``events.mlb_game_pk``. In each event's order of first sight."""
    found: list[StoredId] = []
    for source, query in _STORED_ID_QUERIES.items():
        rows = await session.execute(text(query))
        found += [StoredId(row.event_id, source, str(row.external_id)) for row in rows]
    with_odds_api = {i.event_id for i in found if i.source is Source.ODDS_API}
    keyed = await session.execute(select(Event.id))
    found += [
        StoredId(event_id, Source.ODDS_API, event_id)
        for event_id in keyed.scalars()
        if is_odds_api_event_id(event_id) and event_id not in with_odds_api
    ]
    return found


@dataclass
class ExternalIdReport:
    outcomes: Counter[tuple[Source, IdOutcome]] = field(default_factory=Counter)
    refused: list[StoredId] = field(default_factory=list)


async def record_stored_ids(
    session: AsyncSession, ids: Sequence[StoredId], *, apply: bool
) -> ExternalIdReport:
    """Record each stored id on its event through the resolver's rules: one id per source per
    event (the first seen wins, refusals logged once), one event per id. An event already holding
    an Odds API id keeps it. Events go in fixture-lock order, each under its lock, committed
    every ``EVENTS_PER_COMMIT`` events. Without ``apply`` the outcomes are counted, read-only."""
    report = ExternalIdReport()
    by_event: dict[str, list[StoredId]] = defaultdict(list)
    for stored in ids:
        by_event[stored.event_id].append(stored)
    events = list(
        (
            await session.execute(select(Event).where(Event.id.in_(list(by_event))))  # type: ignore[attr-defined]
        ).scalars()
    )
    events.sort(key=lambda e: (event_lock_key(e.sport_key, e.home_team, e.away_team), e.id))
    # What this run has recorded, for counting a run that writes nothing.
    holders: dict[tuple[Source, str], int] = {}
    held_sources: set[tuple[int, Source]] = set()
    for i, event in enumerate(events):
        if apply:
            await lock_fixture(session, event)
        assert event.surrogate_id is not None
        for stored in by_event[event.id]:
            outcome = await _id_outcome(session, event, stored, holders, held_sources)
            report.outcomes[(stored.source, outcome)] += 1
            if outcome is IdOutcome.REFUSED:
                report.refused.append(stored)
            if outcome in (IdOutcome.HELD, IdOutcome.KEPT_CURRENT):
                continue
            if outcome is IdOutcome.NEW:
                holders[(stored.source, stored.external_id)] = event.surrogate_id
                held_sources.add((event.surrogate_id, stored.source))
            if apply:
                await record_external_id(session, event, stored.source, stored.external_id)
        if apply and (i + 1) % EVENTS_PER_COMMIT == 0:
            await session.commit()
    if apply:
        await session.commit()
    else:
        await session.rollback()
    return report


async def _id_outcome(
    session: AsyncSession,
    event: Event,
    stored: StoredId,
    holders: Mapping[tuple[Source, str], int],
    held_sources: set[tuple[int, Source]],
) -> IdOutcome:
    """The resolver's decision (``check_external_id``), with what this run recorded before
    ``stored`` laid over it so that a run writing nothing counts as one that writes."""
    assert event.surrogate_id is not None
    holder = holders.get((stored.source, stored.external_id))
    if holder is not None:
        return IdOutcome.HELD if holder == event.surrogate_id else IdOutcome.REFUSED
    check = await check_external_id(session, event, stored.source, stored.external_id)
    if check.decision is IdDecision.HELD:
        return IdOutcome.HELD
    if check.refused:
        return IdOutcome.REFUSED
    if check.decision is IdDecision.REPLACES:
        return IdOutcome.KEPT_CURRENT
    if (event.surrogate_id, stored.source) in held_sources:
        return IdOutcome.KEPT_CURRENT if stored.source is Source.ODDS_API else IdOutcome.REFUSED
    return IdOutcome.NEW


_HISTORY_FILE = re.compile(r"^(?P<league>epl|nba)(?:_[a-z0-9_]+)?_(?P<season>\d{4}-\d{4})\.json$")
_LEAGUES: dict[str, tuple[str, str]] = {
    "epl": ("soccer_epl", "EPL"),
    "nba": ("basketball_nba", "NBA"),
}


def oddsportal_history_games(data_dir: Path) -> list[ListedGame]:
    """The games in the historical OddsPortal scrapes (``scripts/ingest_oddsportal.py``'s
    input) whose match link carries an OddsPortal event id, as the loader lists them."""
    games: dict[str, ListedGame] = {}
    for path in sorted(data_dir.glob("*.json")):
        found = _HISTORY_FILE.match(path.name)
        if found is None:
            continue
        sport_key, sport_title = _LEAGUES[found.group("league")]
        records = json.loads(path.read_text())
        for record in records if isinstance(records, list) else []:
            external_id = event_id_from_match_link(record.get("match_link"))
            home, away = record.get("home_team"), record.get("away_team")
            if not external_id or not home or not away or not record.get("match_date"):
                continue
            if sport_key == "basketball_nba":
                home, away = normalize_team(home), normalize_team(away)
                if home is None or away is None:
                    continue
            games.setdefault(
                external_id,
                ListedGame(
                    source=Source.ODDSPORTAL,
                    sport_key=sport_key,
                    sport_title=sport_title,
                    home_team=home,
                    away_team=away,
                    commence_time=parse_match_date(record["match_date"]),
                    external_id=external_id,
                    match_window=MatchWindow.HISTORICAL,
                ),
            )
    return list(games.values())


@dataclass
class OddsportalHistoryReport:
    games: int = 0
    attached: int = 0
    unattached: int = 0


async def attach_oddsportal_history(
    session: AsyncSession, games: Sequence[ListedGame], *, apply: bool
) -> OddsportalHistoryReport:
    """Record each historical game's OddsPortal id on its event, matched as the loader matches
    (``attach_event``, ±24 h); a game with no event or a refused id stays unattached. Nothing
    is created. Without ``apply`` only the games are counted."""
    report = OddsportalHistoryReport(games=len(games))
    if not apply:
        return report
    for start in range(0, len(games), EVENTS_PER_COMMIT):
        attached = await attach_events(session, games[start : start + EVENTS_PER_COMMIT])
        await session.commit()
        report.attached += sum(1 for a in attached if a is not None)
        report.unattached += sum(1 for a in attached if a is None)
    return report
