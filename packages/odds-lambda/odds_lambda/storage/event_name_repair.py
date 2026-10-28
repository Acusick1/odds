from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from odds_core.event_ids import LIVE_EVENT_ID_PREFIX
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    OddsSnapshot,
    SnapshotPayload,
)
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.betfair.adapter import canonical_teams
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.storage.event_merge import EventMerger, MergeReport, first_stored

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

    def describe_merges(self) -> list[str]:
        return [
            f"{m.kept_id} [{self.status_by_id[m.kept_id].value}] <- "
            + ", ".join(f"{i} [{self.status_by_id[i].value}]" for i in m.merged_ids)
            for m in self.merges
        ]


def _event_envelope(payload: Any, betfair_ids: Sequence[str]) -> dict[str, Any] | None:
    """The event's envelope in a Betfair payload: the payload itself, or in a run's list of
    envelopes the one carrying the first of ``betfair_ids`` that any envelope carries."""
    if isinstance(payload, dict):
        return payload
    by_id = {
        str(envelope.get("betfair_event_id")): envelope
        for envelope in payload
        if isinstance(envelope, dict)
    }
    return next((by_id[i] for i in betfair_ids if i in by_id), None)


async def _betfair_ids(session: AsyncSession, sport_key: str) -> dict[str, list[str]]:
    """Every Betfair id each event has held: its ``betfair`` external id first, then the ids
    the resolver or a merge refused, replaced or dropped for it (``data_quality_logs``), any
    of which a snapshot of the event may have been stored under."""
    ids: dict[str, list[str]] = defaultdict(list)
    current = await session.execute(
        select(Event.id, EventExternalId.external_id)
        .join(EventExternalId, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(Event.sport_key == sport_key, EventExternalId.source == Source.BETFAIR)
    )
    for event_id, external_id in current.all():
        ids[event_id].append(external_id)
    logged = await session.execute(
        select(DataQualityLog.event_id, DataQualityLog.raw_data)
        .join(Event, Event.id == DataQualityLog.event_id)  # type: ignore[arg-type]
        .where(
            Event.sport_key == sport_key,
            DataQualityLog.raw_data["source"].as_string() == Source.BETFAIR.value,  # type: ignore[index]
        )
        .order_by(DataQualityLog.created_at.desc())
    )
    for event_id, context in logged.all():
        for key in ("external_id", "old_external_id"):
            value = (context or {}).get(key)
            if value is not None and str(value) not in ids[event_id]:
                ids[event_id].append(str(value))
    return ids


async def _betfair_runner_names(session: AsyncSession, sport_key: str) -> dict[str, list[str]]:
    """Team runner names from the payload of each event's latest Betfair match odds snapshot,
    read from the envelope's match odds entry (the envelope also carries the event's other
    markets). An event whose envelope cannot be found in that payload is logged and resolves
    without them."""
    match_odds = SPORT_CONFIG[sport_key].match_odds.market
    rows = await session.execute(
        select(OddsSnapshot.event_id, SnapshotPayload.payload)
        .join(Event, Event.id == OddsSnapshot.event_id)
        .join(SnapshotPayload, SnapshotPayload.id == OddsSnapshot.payload_id)  # type: ignore[arg-type]
        .where(
            Event.sport_key == sport_key,
            OddsSnapshot.source == Source.BETFAIR,
            OddsSnapshot.market == match_odds,
        )
        .order_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time.desc())
        .distinct(OddsSnapshot.event_id)
    )
    betfair_ids = await _betfair_ids(session, sport_key)
    names: dict[str, list[str]] = {}
    for event_id, payload in rows:
        envelope = _event_envelope(payload, betfair_ids.get(event_id, []))
        if envelope is None:
            logger.warning(
                "betfair_envelope_not_found",
                event_id=event_id,
                betfair_ids=betfair_ids.get(event_id, []),
            )
            continue
        for bookmaker in envelope.get("bookmakers", []):
            if match_odds.value not in {m.get("key") for m in bookmaker.get("markets", [])}:
                continue
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
        if not event.id.startswith(LIVE_EVENT_ID_PREFIX):
            continue
        teams = canonical_teams(event.home_team, event.away_team, runner_names.get(event.id, []))
        if teams != (event.home_team, event.away_team):
            misnamed.append((event, teams))
    return misnamed


async def repair_team_names(session: AsyncSession, sport_key: str) -> NameRepairReport:
    """Give Betfair-created events their canonical team names and merge the twins that creates.

    A merged group keeps its first-stored event that was never renamed, so the
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
        originals = [event for event in group if event.id not in renamed_ids]
        keep = first_stored(originals or group)
        duplicates = [event for event in group if event is not keep]
        merges.append(await merger.merge(keep, duplicates))

    report = NameRepairReport(tuple(sorted(renamed_ids)), tuple(merges), status_by_id)
    logger.info(
        "annotated_names_repaired",
        sport_key=sport_key,
        renamed=len(report.renamed_ids),
        merged=report.merged_count,
    )
    return report
