from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import structlog
from odds_core.event_source import canonical_event_id
from odds_core.models import Event, EventStatus
from odds_core.team import normalize_team_name
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.adapter import strip_pitcher_annotation
from odds_lambda.storage.event_merge import EventMerger, MergeReport

logger = structlog.get_logger()


@dataclass(frozen=True)
class NameRepairReport:
    renamed_ids: tuple[str, ...]
    merges: tuple[MergeReport, ...]
    status_by_id: dict[str, EventStatus]

    @property
    def merged_count(self) -> int:
        return sum(len(m.merged_ids) for m in self.merges)

    @property
    def renamed_in_place_ids(self) -> tuple[str, ...]:
        merged = {merged_id for m in self.merges for merged_id in m.merged_ids}
        return tuple(i for i in self.renamed_ids if i not in merged)

    @property
    def renamed_in_place_statuses(self) -> dict[EventStatus, int]:
        return dict(Counter(self.status_by_id[i] for i in self.renamed_in_place_ids))


def _has_annotation(event: Event) -> bool:
    return (
        strip_pitcher_annotation(event.home_team) != event.home_team
        or strip_pitcher_annotation(event.away_team) != event.away_team
    )


async def find_annotated_events(session: AsyncSession, sport_key: str) -> list[Event]:
    """Events whose team names still carry a pitcher annotation, e.g. ``Tampa Bay Rays (D Rasm``."""
    events = (
        (
            await session.execute(
                select(Event)
                .where(Event.sport_key == sport_key)
                .where(Event.home_team.contains("(") | Event.away_team.contains("("))
                .order_by(Event.commence_time, Event.id)
            )
        )
        .scalars()
        .all()
    )
    return [event for event in events if _has_annotation(event)]


async def repair_annotated_names(session: AsyncSession, sport_key: str) -> NameRepairReport:
    """Strip pitcher annotations from team names and merge the events that become twins.

    A merged group keeps its best-sourced event that was never annotated, so the
    kept event has the real kick-off and a clean id; only a group made solely of
    annotated events keeps one of those. Runs in the caller's transaction and is
    a no-op once no annotated events remain.
    """
    annotated = await find_annotated_events(session, sport_key)
    for event in annotated:
        event.home_team = normalize_team_name(strip_pitcher_annotation(event.home_team))
        event.away_team = normalize_team_name(strip_pitcher_annotation(event.away_team))
    await session.flush()

    renamed_ids = {event.id for event in annotated}
    status_by_id = {event.id: event.status for event in annotated}
    merger = EventMerger(session)
    merges: list[MergeReport] = []
    for group in await merger.find_twin_groups(sport_key):
        if not renamed_ids.intersection(event.id for event in group):
            continue
        by_id = {event.id: event for event in group}
        originals = [event_id for event_id in by_id if event_id not in renamed_ids]
        keep_id = canonical_event_id(originals or list(by_id))
        duplicates = [event for event in group if event.id != keep_id]
        merges.append(await merger.merge(by_id[keep_id], duplicates))

    report = NameRepairReport(tuple(sorted(renamed_ids)), tuple(merges), status_by_id)
    logger.info(
        "annotated_names_repaired",
        sport_key=sport_key,
        renamed=len(report.renamed_ids),
        merged=report.merged_count,
    )
    return report
