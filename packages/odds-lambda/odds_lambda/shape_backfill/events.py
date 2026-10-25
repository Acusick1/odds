"""Bring stored events into the one-event-per-match shape: merge twins, record the external ids
held in stored data, and move ``no_game_pk_reason`` to the UNMATCHED status."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from odds_core.event_ids import is_odds_api_event_id
from odds_core.event_source import EventSource, canonical_event_id
from odds_core.models import Event, EventExternalId, HealthCheckAcknowledgement
from odds_core.time import ensure_utc
from odds_core.vocabulary import HealthCheck, Source
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.oddsportal_common import event_id_from_match_link, parse_match_date
from odds_lambda.polymarket_matching import normalize_team
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY
from odds_lambda.storage.event_merge import EventMerger, retime_snapshots
from odds_lambda.storage.event_resolver import (
    ListedGame,
    MatchWindow,
    attach_events,
    event_lock_key,
    lock_fixture,
    record_external_id,
)
from odds_lambda.storage.event_unmatched import events_holding_game_pks, mark_unmatched

# Fixtures whose locks one transaction takes before it commits.
EVENTS_PER_COMMIT = 200

_SOURCE_OF_ID: dict[EventSource, Source] = {
    EventSource.ODDS_API: Source.ODDS_API,
    EventSource.ODDSPORTAL_HISTORICAL: Source.ODDSPORTAL,
    EventSource.ODDSPORTAL_LIVE: Source.ODDSPORTAL,
    EventSource.FOOTBALL_DATA_UK: Source.FOOTBALL_DATA_UK,
}


@dataclass(frozen=True)
class TwinMerge:
    """One twin group: the event kept, the events merged into it, and its kick-off."""

    kept_id: str
    merged_ids: tuple[str, ...]
    kickoff_source: Source | None
    commence_from: datetime
    commence_to: datetime
    rows_moved: dict[str, int] = field(default_factory=dict)
    rows_deleted: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def kickoff_source(event: Event, holds_game_pk: bool) -> Source:
    """The source whose schedule an event's stored kick-off came from: the MLB Stats API for an
    event holding a ``gamePk``, else the source that minted its id."""
    if holds_game_pk:
        return Source.MLB_STATS_API
    return _SOURCE_OF_ID[EventSource.of(event.id)]


def plan_twin_merge(group: Sequence[Event], game_pk_holders: set[str]) -> TwinMerge:
    """Keep the canonical event and give it the kick-off of the twin whose source ranks highest
    in its sport's ``KICKOFF_AUTHORITY`` order (ties to the lowest surrogate id); with no twin
    from a source in the order, the kept event's own."""
    keep_id = canonical_event_id([e.id for e in group])
    keep = next(e for e in group if e.id == keep_id)
    order = KICKOFF_AUTHORITY.get(keep.sport_key, ())  # type: ignore[call-overload]
    ranked = sorted(
        (order.index(src), e.surrogate_id or 0, e, src)
        for e in group
        if (src := kickoff_source(e, e.id in game_pk_holders)) in order
    )
    source, start = (
        (ranked[0][3], ranked[0][2].commence_time) if ranked else (None, keep.commence_time)
    )
    return TwinMerge(
        kept_id=keep_id,
        merged_ids=tuple(sorted(e.id for e in group if e.id != keep_id)),
        kickoff_source=source,
        commence_from=ensure_utc(keep.commence_time),
        commence_to=ensure_utc(start),
    )


async def merge_twins(session: AsyncSession, *, apply: bool) -> list[TwinMerge]:
    """Merge every twin group (same sport and teams within ±2 h) with ``EventMerger``, its clash
    handling included, each group under its fixture lock in its own transaction. A group that
    fails (different ``gamePk``s, a snapshot key both events hold) is rolled back and reported
    with its error; the rest go ahead. Without ``apply`` nothing is written."""
    merger = EventMerger(session)
    groups = [[e.id for e in group] for group in await merger.find_twin_groups()]
    await session.rollback()
    merges: list[TwinMerge] = []
    for ids in groups:
        events = list(
            (await session.execute(select(Event).where(Event.id.in_(ids)))).scalars()  # type: ignore[attr-defined]
        )
        if len(events) < 2:
            await session.rollback()
            continue
        if apply:
            await lock_fixture(session, events[0])
            for event in events:
                await session.refresh(event)
        plan = plan_twin_merge(events, await events_holding_game_pks(session, events))
        if not apply:
            merges.append(plan)
            await session.rollback()
            continue
        try:
            keep = next(e for e in events if e.id == plan.kept_id)
            keep.commence_time = plan.commence_to
            report = await merger.merge(keep, [e for e in events if e.id != plan.kept_id])
            if plan.commence_to != plan.commence_from:
                await retime_snapshots(session, keep.id, plan.commence_to)
            await session.commit()
            merges.append(
                replace(plan, rows_moved=report.rows_moved, rows_deleted=report.rows_deleted)
            )
        except Exception as exc:  # noqa: BLE001 - reported per group, the others go ahead
            await session.rollback()
            merges.append(replace(plan, error=repr(exc)))
    return merges


@dataclass
class UnmatchedReport:
    marked: list[str] = field(default_factory=list)
    already: int = 0


async def migrate_unmatched(session: AsyncSession, *, apply: bool) -> UnmatchedReport:
    """Give every event carrying ``no_game_pk_reason`` the UNMATCHED status and its reason as a
    ``missing_mlb_game_pk`` acknowledgement (``mark_unmatched``; a final event keeps its status).
    An event already acknowledged with the same reason is left alone. The column stays until
    settlement and the health check read the status. Without ``apply`` nothing is written."""
    report = UnmatchedReport()
    events = list(
        (
            await session.execute(
                select(Event).where(Event.no_game_pk_reason.is_not(None)).order_by(Event.id)  # type: ignore[union-attr]
            )
        ).scalars()
    )
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
        reason = (event.no_game_pk_reason or "").strip()
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
            outcome = await _id_outcome(session, event.surrogate_id, stored, holders, held_sources)
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
    event_key: int,
    stored: StoredId,
    holders: dict[tuple[Source, str], int],
    held_sources: set[tuple[int, Source]],
) -> IdOutcome:
    holder = holders.get((stored.source, stored.external_id))
    if holder is None:
        row = await session.get(EventExternalId, (stored.source, stored.external_id))
        holder = row.event_id if row is not None else None
    if holder == event_key:
        return IdOutcome.HELD
    holds_source = (event_key, stored.source) in held_sources or (
        await session.execute(
            select(EventExternalId.external_id).where(
                EventExternalId.event_id == event_key, EventExternalId.source == stored.source
            )
        )
    ).first() is not None
    if holds_source and stored.source is Source.ODDS_API and holder is None:
        return IdOutcome.KEPT_CURRENT
    if holder is not None or holds_source:
        return IdOutcome.REFUSED
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
