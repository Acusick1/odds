from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog
from odds_core.models import Event, EventExternalId, HealthCheckAcknowledgement
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.time import ensure_utc
from odds_core.vocabulary import HealthCheck, Source
from sqlalchemy import and_, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.data_quality import record_issue
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY
from odds_lambda.storage.event_unmatched import clear_unmatched
from odds_lambda.storage.game_pks import game_pks_of, holds_game_pk

logger = structlog.get_logger()

EVENT_MATCH_WINDOW = timedelta(hours=2)
HISTORICAL_MATCH_WINDOW = timedelta(hours=24)
SEASON_START_MONTH = 8


class ConflictingGamePks(ValueError):
    """Events holding different MLB ``gamePk``s are different games and are never merged."""


@dataclass(frozen=True)
class MergeReport:
    kept_id: str
    merged_ids: tuple[str, ...]
    rows_moved: dict[str, int] = field(default_factory=dict)
    rows_deleted: dict[str, int] = field(default_factory=dict)


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
    """The event within ``EVENT_MATCH_WINDOW``; of several, the one a merge would keep
    (``kept_event``). Read-only."""
    events = await find_match_candidates(session, home_team, away_team, commence_time, sport_key)
    return (await kept_event(session, events)).id if events else None


def first_stored(events: Sequence[Event]) -> Event:
    """The event stored first, by surrogate key."""
    return min(events, key=lambda event: (event.surrogate_id is None, event.surrogate_id or 0))


async def kept_event(session: AsyncSession, events: Sequence[Event]) -> Event:
    """The event of one match that a merge keeps: the one holding an id from the source its
    sport's kick-off authority order (``KICKOFF_AUTHORITY``) ranks highest, so the kept kick-off
    is the best source's; events no ranked source names, and ties, go to the first stored.
    Read-only."""
    if len(events) == 1:
        return events[0]
    order = KICKOFF_AUTHORITY.get(events[0].sport_key, ())  # type: ignore[call-overload]
    rows = await session.execute(
        select(EventExternalId.event_id, EventExternalId.source).where(
            EventExternalId.event_id.in_([e.surrogate_id for e in events])  # type: ignore[attr-defined]
        )
    )
    rank: dict[int, int] = {}
    for event_id, source in rows.all():
        if source in order:
            rank[event_id] = min(rank.get(event_id, len(order)), order.index(source))

    def rank_of(event: Event) -> int:
        return rank.get(event.surrogate_id, len(order)) if event.surrogate_id else len(order)

    best = min(rank_of(e) for e in events)
    return first_stored([e for e in events if rank_of(e) == best])


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


async def find_historical_candidates(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
    sport_key: str,
) -> list[Event]:
    """Every event of the same sport and teams within ``HISTORICAL_MATCH_WINDOW``, for a
    historical row whose recorded kick-off may be a day off. Read-only."""
    return await _candidates(
        session, home_team, away_team, commence_time, HISTORICAL_MATCH_WINDOW, sport_key
    )


class EventMerger:
    """The one way to collapse several events for one match into a single event.

    Every foreign key to ``events`` moves to the kept event, results the kept
    event lacks are carried over, and the duplicates are deleted. The kept
    event keeps its own commence time. It takes a duplicate's external ids from
    sources it holds none from, its ``gamePk`` among them; events holding
    different ``gamePk``s raise ``ConflictingGamePks``. Runs in the caller's
    transaction.
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
        clusters = [cluster for cluster in clusters if len(cluster) > 1]
        held = await game_pks_of(self.session, [e for cluster in clusters for e in cluster])
        groups: list[list[Event]] = []
        for cluster in clusters:
            game_pks = {held[e.id] for e in cluster if e.id in held}
            if len(game_pks) > 1:
                logger.warning(
                    "twin_group_conflicting_game_pks",
                    event_ids=[e.id for e in cluster],
                    game_pks=sorted(game_pks),
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

    async def merge_group(self, events: Sequence[Event]) -> MergeReport:
        """Merge twins into the one ``kept_event`` picks."""
        keep = await kept_event(self.session, events)
        duplicates = sorted((e for e in events if e is not keep), key=lambda e: e.id)
        return await self.merge(keep, duplicates)

    async def merge(self, keep: Event, duplicates: Sequence[Event]) -> MergeReport:
        game_pks = set((await game_pks_of(self.session, [keep, *duplicates])).values())
        if len(game_pks) > 1:
            raise ConflictingGamePks(
                f"refusing to merge {[d.id for d in duplicates]} into {keep.id}: "
                f"they hold different MLB gamePks {sorted(game_pks)}"
            )
        rows_moved: dict[str, int] = {}
        rows_deleted: dict[str, int] = {}
        for duplicate in duplicates:
            for table, count in (await self._drop_clashing_decisions(duplicate, keep)).items():
                rows_deleted[table] = rows_deleted.get(table, 0) + count
            for table, count in (await self._drop_clashing_identity_rows(duplicate, keep)).items():
                rows_deleted[table] = rows_deleted.get(table, 0) + count
            for table, count in (await self._repoint_event_references(duplicate, keep)).items():
                rows_moved[table] = rows_moved.get(table, 0) + count
            self._carry_result(keep, duplicate)
            await self.session.delete(duplicate)
        if await holds_game_pk(self.session, keep):
            # An acknowledgement that the event has no game moved with a duplicate's rows.
            await clear_unmatched(self.session, keep)
        report = MergeReport(keep.id, tuple(d.id for d in duplicates), rows_moved, rows_deleted)
        logger.info(
            "events_merged",
            kept_id=report.kept_id,
            merged_ids=report.merged_ids,
            rows_moved=rows_moved,
        )
        return report

    async def merge_drops(self, keep: Event, duplicates: Sequence[Event]) -> dict[str, int]:
        """The rows ``merge(keep, duplicates)`` would delete, per table, leaving out tables with
        none: the duplicates' clashing candidate decisions and their paper trades, external ids
        and acknowledgements (``_drop_clashing_decisions``, ``_drop_clashing_identity_rows``),
        counted as each duplicate's rows join the kept event. Read-only."""
        versions = await self._decision_versions(keep)
        sources = await self._id_sources(keep)
        checks = await self._ack_checks(keep)
        drops: dict[str, int] = {}

        def add(table: str, count: int) -> None:
            if count:
                drops[table] = drops.get(table, 0) + count

        for duplicate in duplicates:
            decisions = (
                await self.session.execute(
                    select(
                        CandidateDecision.id,
                        CandidateDecision.candidate_name,
                        CandidateDecision.candidate_version,
                    ).where(CandidateDecision.event_id == duplicate.id)
                )
            ).all()
            clashing = [
                d.id for d in decisions if (d.candidate_name, d.candidate_version) in versions
            ]
            add("candidate_decisions", len(clashing))
            if clashing:
                trades = await self.session.execute(
                    select(func.count()).where(PaperTrade.decision_id.in_(clashing))  # type: ignore[union-attr]
                )
                add("paper_trades", trades.scalar_one())
            versions |= {(d.candidate_name, d.candidate_version) for d in decisions}
            duplicate_sources = await self._id_sources(duplicate)
            add("event_external_ids", len(duplicate_sources & sources))
            sources |= duplicate_sources
            duplicate_checks = await self._ack_checks(duplicate)
            add("health_check_acknowledgements", len(duplicate_checks & checks))
            checks |= duplicate_checks
        return drops

    async def _decision_versions(self, event: Event) -> set[tuple[str, int]]:
        rows = await self.session.execute(
            select(CandidateDecision.candidate_name, CandidateDecision.candidate_version).where(
                CandidateDecision.event_id == event.id
            )
        )
        return {(name, version) for name, version in rows.all()}

    async def _id_sources(self, event: Event) -> set[Source]:
        rows = await self.session.execute(
            select(EventExternalId.source).where(EventExternalId.event_id == event.surrogate_id)
        )
        return set(rows.scalars())

    async def _ack_checks(self, event: Event) -> set[HealthCheck]:
        rows = await self.session.execute(
            select(HealthCheckAcknowledgement.check_name).where(
                HealthCheckAcknowledgement.event_id == event.surrogate_id
            )
        )
        return set(rows.scalars())

    async def _drop_clashing_decisions(self, duplicate: Event, keep: Event) -> dict[str, int]:
        """Delete the duplicate's decisions, and their trades, that the kept event has too.

        A candidate version decides a match once per decision time and bets each outcome once
        (``uq_candidate_decisions_candidate_event_lead``, ``uq_paper_trades_candidate_bet``),
        so when the kept event holds any decision of a candidate version, that version's
        decisions on the duplicate are dropped. The other decisions and trades move with the
        foreign keys: a trade only clashes with the kept event's trades through a decision of
        the same candidate version, and those are the ones deleted here.
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

    async def _drop_clashing_identity_rows(self, duplicate: Event, keep: Event) -> dict[str, int]:
        """Delete the duplicate's external ids from sources the kept event has an id from, and
        its acknowledgements of checks the kept event is acknowledged for, logging each.

        An event holds one id per source and one acknowledgement per check, so the kept
        event's rows win, as the resolver keeps the first id it recorded.
        """
        deleted: dict[str, int] = {}
        kept_sources = set(
            (
                await self.session.execute(
                    select(EventExternalId.source).where(
                        EventExternalId.event_id == keep.surrogate_id
                    )
                )
            ).scalars()
        )
        for row in (
            await self.session.execute(
                select(EventExternalId).where(EventExternalId.event_id == duplicate.surrogate_id)
            )
        ).scalars():
            if row.source not in kept_sources:
                continue
            record_issue(
                self.session,
                keep.id,
                "merge_dropped_external_id",
                f"{row.source.value} id {row.external_id} of merged event {duplicate.id} dropped: "
                f"the kept event holds a {row.source.value} id",
                {
                    "source": row.source.value,
                    "external_id": row.external_id,
                    "duplicate_id": duplicate.id,
                },
            )
            await self.session.delete(row)
            deleted["event_external_ids"] = deleted.get("event_external_ids", 0) + 1
        kept_checks = set(
            (
                await self.session.execute(
                    select(HealthCheckAcknowledgement.check_name).where(
                        HealthCheckAcknowledgement.event_id == keep.surrogate_id
                    )
                )
            ).scalars()
        )
        for ack in (
            await self.session.execute(
                select(HealthCheckAcknowledgement).where(
                    HealthCheckAcknowledgement.event_id == duplicate.surrogate_id
                )
            )
        ).scalars():
            if ack.check_name not in kept_checks:
                continue
            record_issue(
                self.session,
                keep.id,
                "merge_dropped_acknowledgement",
                f"{ack.check_name.value} acknowledgement of merged event {duplicate.id} dropped: "
                "the kept event has one",
                {
                    "check_name": ack.check_name.value,
                    "reason": ack.reason,
                    "duplicate_id": duplicate.id,
                },
            )
            await self.session.delete(ack)
            deleted["health_check_acknowledgements"] = (
                deleted.get("health_check_acknowledgements", 0) + 1
            )
        if deleted:
            await self.session.flush()
        return deleted

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

    async def _event_reference_columns(self) -> list[tuple[str, str, str, str]]:
        """``(schema, table, column, events column)`` for every foreign key to ``events``.

        Read from the database catalog rather than ``SQLModel.metadata``, which
        only holds models the running process has imported.
        """
        result = await self.session.execute(
            text(
                "SELECT n.nspname, r.relname, a.attname, ra.attname"
                " FROM pg_constraint c"
                " JOIN pg_class r ON r.oid = c.conrelid"
                " JOIN pg_namespace n ON n.oid = r.relnamespace"
                " JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]"
                " JOIN pg_attribute ra ON ra.attrelid = c.confrelid AND ra.attnum = c.confkey[1]"
                " WHERE c.contype = 'f' AND c.confrelid = 'events'::regclass"
                " ORDER BY r.relname, a.attname"
            )
        )
        return [(row[0], row[1], row[2], row[3]) for row in result.all()]

    async def reference_counts(self, event_id: str) -> dict[str, int]:
        """Rows referencing the event, per table, leaving out tables with none."""
        event = await self.session.get(Event, event_id)
        if event is None:
            return {}
        quote = self.session.get_bind().dialect.identifier_preparer.quote
        counts: dict[str, int] = {}
        for schema, table, column, events_column in await self._event_reference_columns():
            count = (
                await self.session.execute(
                    text(
                        f"SELECT count(*) FROM {quote(schema)}.{quote(table)}"
                        f" WHERE {quote(column)} = :key"
                    ),
                    {"key": getattr(event, events_column)},
                )
            ).scalar_one()
            if count:
                counts[table] = counts.get(table, 0) + count
        return counts

    async def _repoint_event_references(self, duplicate: Event, keep: Event) -> dict[str, int]:
        quote = self.session.get_bind().dialect.identifier_preparer.quote
        moved: dict[str, int] = {}
        for schema, table, column, events_column in await self._event_reference_columns():
            result = await self.session.execute(
                text(
                    f"UPDATE {quote(schema)}.{quote(table)} SET {quote(column)} = :to_key"
                    f" WHERE {quote(column)} = :from_key"
                ),
                {
                    "to_key": getattr(keep, events_column),
                    "from_key": getattr(duplicate, events_column),
                },
            )
            if result.rowcount:
                moved[table] = moved.get(table, 0) + result.rowcount
        return moved
