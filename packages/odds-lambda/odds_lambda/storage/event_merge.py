from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.event_source import EventSource, canonical_event_id
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.time import ensure_utc
from sqlalchemy import and_, delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier_from_timestamps

logger = structlog.get_logger()

EVENT_MATCH_WINDOW = timedelta(hours=2)
HISTORICAL_MATCH_WINDOW = timedelta(hours=24)
SEASON_START_MONTH = 8


class ConflictingGamePks(ValueError):
    """Events holding different MLB ``gamePk``s are different games and are never merged."""


def _game_pks(events: Sequence[Event]) -> set[int]:
    return {e.mlb_game_pk for e in events if e.mlb_game_pk is not None}


@dataclass(frozen=True)
class MergeReport:
    kept_id: str
    merged_ids: tuple[str, ...]
    rows_moved: dict[str, int] = field(default_factory=dict)
    rows_deleted: dict[str, int] = field(default_factory=dict)


def source_pair(group: Sequence[Event]) -> tuple[EventSource, ...]:
    return tuple(sorted({EventSource.of(event.id) for event in group}, key=lambda s: s.rank))


def season_start_year(commence_time: datetime) -> int:
    return (
        commence_time.year if commence_time.month >= SEASON_START_MONTH else commence_time.year - 1
    )


async def _candidates(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
    window: timedelta,
    sport_key: str | None,
) -> list[Event]:
    commence = ensure_utc(commence_time)
    conditions = [
        Event.home_team == home_team,
        Event.away_team == away_team,
        Event.commence_time >= commence - window,
        Event.commence_time <= commence + window,
    ]
    if sport_key is not None:
        conditions.append(Event.sport_key == sport_key)
    result = await session.execute(select(Event).where(and_(*conditions)).order_by(Event.id))
    return list(result.scalars().all())


async def find_match_id(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
    sport_key: str,
) -> str | None:
    """The canonical event within ``EVENT_MATCH_WINDOW``. Read-only."""
    events = await find_match_candidates(session, home_team, away_team, commence_time, sport_key)
    return canonical_event_id([e.id for e in events]) if events else None


async def find_match_candidates(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
    sport_key: str,
) -> list[Event]:
    """Every event of the same sport and teams within ``EVENT_MATCH_WINDOW``. Read-only."""
    return await _candidates(
        session, home_team, away_team, commence_time, EVENT_MATCH_WINDOW, sport_key
    )


async def find_historical_match_id(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
    sport_key: str,
) -> str | None:
    """The event for a historical row, whose recorded kick-off may be a day off.

    A lone candidate within ``HISTORICAL_MATCH_WINDOW`` matches. Several
    candidates match only through those within ``EVENT_MATCH_WINDOW``, which are
    twins, so neighbouring fixtures of the same teams are never conflated.
    Read-only.
    """
    events = await _candidates(
        session, home_team, away_team, commence_time, HISTORICAL_MATCH_WINDOW, sport_key
    )
    if len(events) <= 1:
        return events[0].id if events else None
    commence = ensure_utc(commence_time)
    twins = [e for e in events if abs(ensure_utc(e.commence_time) - commence) <= EVENT_MATCH_WINDOW]
    if twins:
        return canonical_event_id([e.id for e in twins])
    logger.warning(
        "ambiguous_historical_match",
        home=home_team,
        away=away_team,
        commence_time=commence.isoformat(),
        candidates=[e.id for e in events],
    )
    return None


async def retime_snapshots(session: AsyncSession, event_id: str, commence_time: datetime) -> int:
    """Recompute ``hours_until_commence`` and ``fetch_tier`` of an event's snapshots against
    ``commence_time``. Flushes; returns the number of snapshots re-timed."""
    snapshots = (
        (await session.execute(select(OddsSnapshot).where(OddsSnapshot.event_id == event_id)))
        .scalars()
        .all()
    )
    commence = ensure_utc(commence_time)
    for snapshot in snapshots:
        snapshot_time = ensure_utc(snapshot.snapshot_time)
        snapshot.hours_until_commence = calculate_hours_until_commence(snapshot_time, commence)
        snapshot.fetch_tier = calculate_tier_from_timestamps(snapshot_time, commence).value
    await session.flush()
    return len(snapshots)


async def follow_schedule(
    session: AsyncSession, event: Event, game_pk: int, start: datetime | None
) -> None:
    """Move the event to its game's scheduled ``start`` and re-time its stored snapshots; a
    postponed event becomes scheduled. No-op when ``start`` is ``None`` or unchanged."""
    if start is None or ensure_utc(event.commence_time) == ensure_utc(start):
        return
    event_id = event.id
    moved_from = ensure_utc(event.commence_time)
    event.commence_time = start
    event.updated_at = datetime.now(UTC)
    if event.status == EventStatus.POSTPONED:
        event.status = EventStatus.SCHEDULED
    retimed = await retime_snapshots(session, event_id, start)
    logger.info(
        "mlb_event_rescheduled",
        event_id=event_id,
        game_pk=game_pk,
        commence_time=moved_from.isoformat(),
        scheduled_start=ensure_utc(start).isoformat(),
        snapshots_retimed=retimed,
    )


class EventMerger:
    """The one way to collapse several events for one match into a single event.

    Every ``event_id`` foreign key moves to the kept event, results the kept
    event lacks are carried over, and the duplicates are deleted. The kept
    event keeps its own commence time: it is the best source, and the Odds API
    id is kept because score and status jobs match on it. The kept event takes
    a duplicate's ``mlb_game_pk`` when it has none; events holding different
    ``gamePk``s raise ``ConflictingGamePks``. Runs in the caller's transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _events_by_match(
        self, sport_key: str | None
    ) -> dict[tuple[str, str, str], list[Event]]:
        query = select(Event).order_by(Event.commence_time, Event.id)
        if sport_key is not None:
            query = query.where(Event.sport_key == sport_key)
        by_match: dict[tuple[str, str, str], list[Event]] = defaultdict(list)
        for event in (await self.session.execute(query)).scalars():
            by_match[(event.sport_key, event.home_team, event.away_team)].append(event)
        return by_match

    async def find_twin_groups(
        self, sport_key: str | None = None, window: timedelta = EVENT_MATCH_WINDOW
    ) -> list[list[Event]]:
        """Same-match events within ``window`` of the first, leaving out clusters that hold
        different ``gamePk``s: those are separate games (a doubleheader) and are logged."""
        clusters: list[list[Event]] = []
        for events in (await self._events_by_match(sport_key)).values():
            cluster = [events[0]]
            for event in events[1:]:
                if event.commence_time - cluster[0].commence_time <= window:
                    cluster.append(event)
                    continue
                clusters.append(cluster)
                cluster = [event]
            clusters.append(cluster)
        groups: list[list[Event]] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            if len(_game_pks(cluster)) > 1:
                logger.warning(
                    "twin_group_conflicting_game_pks",
                    event_ids=[e.id for e in cluster],
                    game_pks=sorted(_game_pks(cluster)),
                )
                continue
            groups.append(cluster)
        return groups

    async def find_repeated_fixtures(self, sport_key: str) -> list[list[Event]]:
        """Fixtures stored more than once in a season that are not twins.

        A home and away pair meets once a season in a league, so a second event
        is a rescheduled match stored under a second id.
        """
        groups: list[list[Event]] = []
        for events in (await self._events_by_match(sport_key)).values():
            by_season: dict[int, list[Event]] = defaultdict(list)
            for event in events:
                by_season[season_start_year(event.commence_time)].append(event)
            for season_events in by_season.values():
                span = season_events[-1].commence_time - season_events[0].commence_time
                if len(season_events) > 1 and span > EVENT_MATCH_WINDOW:
                    groups.append(season_events)
        return groups

    async def merge_all_twins(
        self, sport_key: str | None = None, window: timedelta = EVENT_MATCH_WINDOW
    ) -> list[MergeReport]:
        return [
            await self.merge_group(group)
            for group in await self.find_twin_groups(sport_key, window)
        ]

    async def merge_group(self, events: Sequence[Event]) -> MergeReport:
        by_id = {event.id: event for event in events}
        keep_id = canonical_event_id(list(by_id))
        duplicates = sorted(
            (event for event in events if event.id != keep_id),
            key=lambda event: (EventSource.of(event.id).rank, event.id),
        )
        return await self.merge(by_id[keep_id], duplicates)

    async def merge(self, keep: Event, duplicates: Sequence[Event]) -> MergeReport:
        game_pks = _game_pks([keep, *duplicates])
        if len(game_pks) > 1:
            raise ConflictingGamePks(
                f"refusing to merge {[d.id for d in duplicates]} into {keep.id}: "
                f"they hold different MLB gamePks {sorted(game_pks)}"
            )
        rows_moved: dict[str, int] = {}
        rows_deleted: dict[str, int] = {}
        for duplicate in duplicates:
            await self._rebase_snapshot_timing(duplicate, keep)
            for table, count in (await self._drop_clashing_decisions(duplicate, keep)).items():
                rows_deleted[table] = rows_deleted.get(table, 0) + count
            for table, count in (
                await self._repoint_event_references(duplicate.id, keep.id)
            ).items():
                rows_moved[table] = rows_moved.get(table, 0) + count
            self._carry_result(keep, duplicate)
            await self.session.delete(duplicate)
        if keep.mlb_game_pk is None and game_pks:
            # The duplicate's row must be gone before the kept event takes its unique gamePk.
            await self.session.flush()
            keep.assign_game_pk(game_pks.pop())
        report = MergeReport(keep.id, tuple(d.id for d in duplicates), rows_moved, rows_deleted)
        logger.info(
            "events_merged",
            kept_id=report.kept_id,
            merged_ids=report.merged_ids,
            rows_moved=rows_moved,
        )
        return report

    async def _rebase_snapshot_timing(self, duplicate: Event, keep: Event) -> None:
        """Recompute moved snapshots' hours and tier against the kept event's kick-off."""
        if ensure_utc(duplicate.commence_time) == ensure_utc(keep.commence_time):
            return
        await retime_snapshots(self.session, duplicate.id, keep.commence_time)

    async def _drop_clashing_decisions(self, duplicate: Event, keep: Event) -> dict[str, int]:
        """Delete the duplicate's decisions, and their trades, that the kept event has too.

        A candidate version decides an event once (``uq_candidate_decisions_candidate_event``),
        so a decision the kept event already holds wins. The other decisions and trades move
        with the foreign keys: a trade only clashes with the kept event's trades through a
        decision of the same candidate version, and those are the ones deleted here.
        """
        kept = {
            (name, version)
            for name, version in (
                await self.session.execute(
                    select(
                        CandidateDecision.candidate_name, CandidateDecision.candidate_version
                    ).where(CandidateDecision.event_id == keep.id)
                )
            ).all()
        }
        clashing = [
            decision.id
            for decision in (
                await self.session.execute(
                    select(CandidateDecision).where(CandidateDecision.event_id == duplicate.id)
                )
            ).scalars()
            if (decision.candidate_name, decision.candidate_version) in kept
        ]
        if not clashing:
            return {}
        trades = await self.session.execute(
            delete(PaperTrade).where(PaperTrade.decision_id.in_(clashing))  # type: ignore[union-attr]
        )
        await self.session.execute(
            delete(CandidateDecision).where(CandidateDecision.id.in_(clashing))  # type: ignore[union-attr]
        )
        logger.warning(
            "merge_dropped_clashing_decisions",
            kept_id=keep.id,
            duplicate_id=duplicate.id,
            decisions=len(clashing),
            trades=trades.rowcount,
        )
        deleted = {"candidate_decisions": len(clashing), "paper_trades": trades.rowcount}
        return {table: count for table, count in deleted.items() if count}

    @staticmethod
    def _carry_result(keep: Event, duplicate: Event) -> None:
        if duplicate.home_score is None and duplicate.away_score is None:
            return
        if keep.home_score is None and keep.away_score is None:
            keep.status = duplicate.status
            keep.home_score = duplicate.home_score
            keep.away_score = duplicate.away_score
            keep.completed_at = duplicate.completed_at
        elif (keep.home_score, keep.away_score) != (duplicate.home_score, duplicate.away_score):
            logger.warning(
                "merge_conflicting_scores",
                kept_id=keep.id,
                kept_score=(keep.home_score, keep.away_score),
                duplicate_id=duplicate.id,
                duplicate_score=(duplicate.home_score, duplicate.away_score),
            )

    async def adopt_twins(self, events: Sequence[Event]) -> int:
        """Merge every non-Odds-API event for the same match into each given event."""
        adopted = 0
        for event in events:
            stored = await self.session.get(Event, event.id, populate_existing=True)
            if stored is None:
                continue
            twins = [
                twin
                for twin in await _candidates(
                    self.session,
                    stored.home_team,
                    stored.away_team,
                    stored.commence_time,
                    EVENT_MATCH_WINDOW,
                    stored.sport_key,
                )
                if twin.id != stored.id and EventSource.of(twin.id) is not EventSource.ODDS_API
            ]
            if twins:
                await self.merge(stored, twins)
                adopted += len(twins)
        return adopted

    async def _event_reference_columns(self) -> list[tuple[str, str, str]]:
        """``(schema, table, column)`` for every foreign key to ``events.id``.

        Read from the database catalog rather than ``SQLModel.metadata``, which
        only holds models the running process has imported.
        """
        result = await self.session.execute(
            text(
                "SELECT n.nspname, r.relname, a.attname"
                " FROM pg_constraint c"
                " JOIN pg_class r ON r.oid = c.conrelid"
                " JOIN pg_namespace n ON n.oid = r.relnamespace"
                " JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)"
                " WHERE c.contype = 'f' AND c.confrelid = 'events'::regclass"
                " ORDER BY r.relname, a.attname"
            )
        )
        return [(row[0], row[1], row[2]) for row in result.all()]

    async def reference_counts(self, event_id: str) -> dict[str, int]:
        """Rows referencing the event, per table, leaving out tables with none."""
        quote = self.session.get_bind().dialect.identifier_preparer.quote
        counts: dict[str, int] = {}
        for schema, table, column in await self._event_reference_columns():
            count = (
                await self.session.execute(
                    text(
                        f"SELECT count(*) FROM {quote(schema)}.{quote(table)}"
                        f" WHERE {quote(column)} = :event_id"
                    ),
                    {"event_id": event_id},
                )
            ).scalar_one()
            if count:
                counts[table] = counts.get(table, 0) + count
        return counts

    async def _repoint_event_references(self, from_id: str, to_id: str) -> dict[str, int]:
        quote = self.session.get_bind().dialect.identifier_preparer.quote
        moved: dict[str, int] = {}
        for schema, table, column in await self._event_reference_columns():
            result = await self.session.execute(
                text(
                    f"UPDATE {quote(schema)}.{quote(table)} SET {quote(column)} = :to_id"
                    f" WHERE {quote(column)} = :from_id"
                ),
                {"to_id": to_id, "from_id": from_id},
            )
            if result.rowcount:
                moved[table] = moved.get(table, 0) + result.rowcount
        return moved
