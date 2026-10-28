from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import structlog
from odds_core.event_source import EventSource, canonical_event_id
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.adapter import canonical_teams
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.storage.event_merge import EventMerger, MergeReport

logger = structlog.get_logger()

BETFAIR_SOURCE = "betfair_api"


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

    def describe_merges(self) -> list[str]:
        return [
            f"{m.kept_id} [{self.status_by_id[m.kept_id].value}] <- "
            + ", ".join(f"{i} [{self.status_by_id[i].value}]" for i in m.merged_ids)
            for m in self.merges
        ]


async def _betfair_runner_names(session: AsyncSession, sport_key: str) -> dict[str, list[str]]:
    """Team runner names from each event's latest Betfair snapshot."""
    rows = await session.execute(
        select(OddsSnapshot.event_id, OddsSnapshot.raw_data)
        .join(Event, Event.id == OddsSnapshot.event_id)
        .where(Event.sport_key == sport_key)
        .where(OddsSnapshot.market == SPORT_CONFIG[sport_key].match_odds.market)
        .where(OddsSnapshot.raw_data["source"].as_string() == BETFAIR_SOURCE)
        .order_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time.desc())
        .distinct(OddsSnapshot.event_id)
    )
    names: dict[str, list[str]] = {}
    for event_id, raw_data in rows:
        for bookmaker in raw_data.get("bookmakers", []):
            runners = bookmaker.get("betfair_meta", {}).get("runners", [])
            names[event_id] = [str(r.get("runner_name") or "") for r in runners]
    return names


async def find_misnamed_events(
    session: AsyncSession, sport_key: str
) -> list[tuple[Event, tuple[str, str]]]:
    """Events whose stored team names differ from the canonical names, with those names.

    Only live-scraped events are considered, since their names were canonicalised
    when created. Those with Betfair snapshots resolve against the full runner
    names; the rest only lose pitcher annotations and game suffixes.
    """
    events = (
        (
            await session.execute(
                select(Event)
                .where(Event.sport_key == sport_key)
                .order_by(Event.commence_time, Event.id)
            )
        )
        .scalars()
        .all()
    )
    runner_names = await _betfair_runner_names(session, sport_key)
    misnamed: list[tuple[Event, tuple[str, str]]] = []
    for event in events:
        if EventSource.of(event.id) is not EventSource.ODDSPORTAL_LIVE:
            continue
        teams = canonical_teams(event.home_team, event.away_team, runner_names.get(event.id, []))
        if teams != (event.home_team, event.away_team):
            misnamed.append((event, teams))
    return misnamed


async def repair_team_names(session: AsyncSession, sport_key: str) -> NameRepairReport:
    """Give Betfair-created events their canonical team names and merge the twins that creates.

    A merged group keeps its best-sourced event that was never renamed, so the
    kept event has the real kick-off and a clean id; only a group made solely of
    renamed events keeps one of those. Doubleheader games stay apart: groups
    are events of the same teams within ``EVENT_MATCH_WINDOW``. Runs in the
    caller's transaction and is a no-op once every name is canonical.
    """
    misnamed = await find_misnamed_events(session, sport_key)
    for event, (home, away) in misnamed:
        event.home_team, event.away_team = home, away
    await session.flush()

    renamed_ids = {event.id for event, _ in misnamed}
    merger = EventMerger(session)
    merges: list[MergeReport] = []
    status_by_id = {event.id: event.status for event, _ in misnamed}
    for group in await merger.find_twin_groups(sport_key):
        if not renamed_ids.intersection(event.id for event in group):
            continue
        status_by_id.update({event.id: event.status for event in group})
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
