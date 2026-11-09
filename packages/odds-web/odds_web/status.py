"""The status strip: last success of each scheduled job, freshness of each collection source,
and the Odds API quota with its burn projection.

Every judgement comes from the readers the health monitor shares; this module only joins them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from odds_core.config import Settings
from odds_lambda.health_readers import (
    Freshness,
    last_heartbeats,
    schedule_heartbeat_name,
    source_freshness,
)
from odds_lambda.scheduling.jobs import sport_key_to_suffix
from odds_lambda.scheduling.schedule_store import read_schedules
from odds_lambda.storage.readers import BurnProjection, OddsReader
from sqlalchemy.ext.asyncio import AsyncSession


class ChipState(StrEnum):
    OK = "ok"
    # A job past its heartbeat expectation, or a source past its staleness threshold.
    LATE = "late"
    # Not enough history to judge.
    UNKNOWN = "unknown"
    # A source with no fixture inside its horizon, so nothing is due.
    IDLE = "idle"


@dataclass(frozen=True)
class JobChip:
    name: str
    state: ChipState
    last_success: datetime | None
    max_hours: float | None
    next_run: datetime | None
    running: bool


@dataclass(frozen=True)
class SourceChip:
    label: str
    state: ChipState
    last_snapshot: datetime | None
    threshold_hours: float | None


@dataclass(frozen=True)
class QuotaChip:
    key_index: int
    sports: tuple[str, ...]
    remaining: int
    limit: int
    projection: BurnProjection | None


@dataclass(frozen=True)
class StatusStrip:
    now: datetime
    jobs: list[JobChip]
    sources: list[SourceChip]
    quotas: list[QuotaChip]


_FRESHNESS_STATE: dict[Freshness, ChipState] = {
    Freshness.FRESH: ChipState.OK,
    Freshness.STALE: ChipState.LATE,
    Freshness.UNKNOWN: ChipState.UNKNOWN,
    Freshness.IDLE: ChipState.IDLE,
}


def _job_state(last: datetime | None, max_hours: float | None, now: datetime) -> ChipState:
    if last is None:
        return ChipState.UNKNOWN
    if max_hours is not None and last < now - timedelta(hours=max_hours):
        return ChipState.LATE
    return ChipState.OK


async def load_status(session: AsyncSession, settings: Settings, now: datetime) -> StatusStrip:
    schedules = await read_schedules(session)
    names = {s.job_name: schedule_heartbeat_name(s.job_name) for s in schedules}
    beats = await last_heartbeats(session, set(names.values()))
    expectations = settings.alerts.heartbeat_expectations
    jobs = [
        JobChip(
            name=s.job_name,
            state=_job_state(
                beats.get(names[s.job_name]), expectations.get(names[s.job_name]), now
            ),
            last_success=beats.get(names[s.job_name]),
            max_hours=expectations.get(names[s.job_name]),
            next_run=s.next_fire_time,
            running=s.running > 0,
        )
        for s in schedules
    ]
    sources = [
        SourceChip(
            label=f"{f.source_key} {sport_key_to_suffix(f.sport_key) or f.sport_key} {f.market}",
            state=_FRESHNESS_STATE[f.status],
            last_snapshot=f.last_snapshot,
            threshold_hours=f.threshold_hours,
        )
        for f in await source_freshness(session, settings, now)
    ]
    reader = OddsReader(session)
    quotas = [
        QuotaChip(index, q.sports, q.remaining, q.limit, await reader.get_quota_burn(q, now))
        for index, q in (await reader.get_latest_quota_by_key(settings.api)).items()
    ]
    return StatusStrip(now, jobs, sources, quotas)
