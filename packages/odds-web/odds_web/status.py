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
    SourceFreshness,
    heartbeat_late,
    last_heartbeats,
    schedule_heartbeat_name,
    source_freshness,
)
from odds_lambda.scheduling.jobs import resolve_job_name, sport_key_to_suffix
from odds_lambda.scheduling.schedule_store import read_schedules
from odds_lambda.storage.readers import BurnProjection, KeyQuota, OddsReader
from sqlalchemy.ext.asyncio import AsyncSession


class ChipState(StrEnum):
    OK = "ok"
    # A job past its heartbeat expectation, or a source past its staleness threshold.
    LATE = "late"
    # Late by more than ``STALE_FACTOR`` times what the job or source is allowed.
    STALE = "stale"
    # Not enough history to judge.
    UNKNOWN = "unknown"
    # A source with no fixture inside its horizon, so nothing is due.
    IDLE = "idle"


# How many times past its allowance a job or source may run before it is shown as stale (red)
# rather than late (amber); the health monitor knows only one threshold.
STALE_FACTOR = 2.0

# The job whose next run the strip shows.
CANDIDATE_JOB = "run-candidates"

# Worst first. Idle sources are dormant, not unhealthy, so they rank last and are never a problem.
_SEVERITY: dict[ChipState, int] = {
    ChipState.STALE: 4,
    ChipState.LATE: 3,
    ChipState.UNKNOWN: 2,
    ChipState.OK: 1,
    ChipState.IDLE: 0,
}
PROBLEM_STATES = frozenset({ChipState.STALE, ChipState.LATE, ChipState.UNKNOWN})


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
    sport: str
    state: ChipState
    last_snapshot: datetime | None
    threshold_hours: float | None
    active_hours: float | None = None


@dataclass(frozen=True)
class QuotaChip:
    key_index: int
    quota: KeyQuota
    projection: BurnProjection | None


@dataclass(frozen=True)
class SportCollection:
    """One sport's collection on the strip: the worst state across its live sources, and the age
    of the source that state comes from."""

    sport: str
    state: ChipState
    last_snapshot: datetime | None
    sources: int


@dataclass(frozen=True)
class NextRun:
    """The soonest ``run-candidates`` fire time across sports, and that job's own state."""

    job: str
    state: ChipState
    next_run: datetime


@dataclass(frozen=True)
class QuotaSummary:
    """Remaining and limit summed over the Odds API keys, and the days left on the worst key
    (None when no key's spend is known; ``no_spend`` when every known key is idle)."""

    remaining: int
    limit: int
    days_left: float | None
    no_spend: bool


@dataclass(frozen=True)
class StatusStrip:
    now: datetime
    jobs: list[JobChip]
    sources: list[SourceChip]
    quotas: list[QuotaChip]

    def collections(self) -> list[SportCollection]:
        """One entry per sport with a live (non-idle) source, in the order sports first appear."""
        by_sport: dict[str, list[SourceChip]] = {}
        for source in self.sources:
            if source.state is not ChipState.IDLE:
                by_sport.setdefault(source.sport, []).append(source)
        return [
            SportCollection(sport, worst.state, worst.last_snapshot, len(live))
            for sport, live in by_sport.items()
            for worst in [max(live, key=lambda c: _badness(c, self.now))]
        ]

    def next_candidate_run(self) -> NextRun | None:
        runs = [
            (job.next_run, job)
            for job in self.jobs
            if job.next_run is not None and resolve_job_name(job.name)[0] == CANDIDATE_JOB
        ]
        if not runs:
            return None
        when, job = min(runs, key=lambda run: run[0])
        return NextRun(job.name, job.state, when)

    def quota(self) -> QuotaSummary | None:
        if not self.quotas:
            return None
        days = [
            q.projection.days_left
            for q in self.quotas
            if q.projection is not None and q.projection.days_left is not None
        ]
        spending_known = any(q.projection is not None for q in self.quotas)
        return QuotaSummary(
            remaining=sum(q.quota.remaining for q in self.quotas),
            limit=sum(q.quota.limit for q in self.quotas),
            days_left=min(days) if days else None,
            no_spend=spending_known and not days,
        )

    def problem_jobs(self) -> list[JobChip]:
        return sorted(
            (j for j in self.jobs if j.state in PROBLEM_STATES), key=lambda j: -_SEVERITY[j.state]
        )

    def problem_sources(self) -> list[SourceChip]:
        return sorted(
            (s for s in self.sources if s.state in PROBLEM_STATES),
            key=lambda s: -_SEVERITY[s.state],
        )


def _badness(source: SourceChip, now: datetime) -> tuple[int, timedelta]:
    """Orders a sport's sources worst first: by state, then by how long since a snapshot."""
    age = timedelta.max if source.last_snapshot is None else now - source.last_snapshot
    return _SEVERITY[source.state], age


_FRESHNESS_STATE: dict[Freshness, ChipState] = {
    Freshness.FRESH: ChipState.OK,
    Freshness.STALE: ChipState.LATE,
    Freshness.UNKNOWN: ChipState.UNKNOWN,
    Freshness.IDLE: ChipState.IDLE,
}


def _job_state(last: datetime | None, max_hours: float | None, now: datetime) -> ChipState:
    if last is None:
        return ChipState.UNKNOWN
    if max_hours is not None and heartbeat_late(last, max_hours, now):
        return (
            ChipState.STALE
            if heartbeat_late(last, max_hours * STALE_FACTOR, now)
            else ChipState.LATE
        )
    return ChipState.OK


def _source_state(freshness: SourceFreshness) -> ChipState:
    state = _FRESHNESS_STATE[freshness.status]
    if (
        state is ChipState.LATE
        and freshness.active_hours is not None
        and freshness.threshold_hours is not None
        and freshness.active_hours > freshness.threshold_hours * STALE_FACTOR
    ):
        return ChipState.STALE
    return state


async def load_status(session: AsyncSession, settings: Settings, now: datetime) -> StatusStrip:
    schedules = await read_schedules(session)
    names = {s.job_name: schedule_heartbeat_name(s.job_name) for s in schedules}
    beats = await last_heartbeats(session, set(names.values()))
    expectations = settings.alerts.heartbeat_expectations
    jobs: list[JobChip] = []
    for schedule in schedules:
        last = beats.get(names[schedule.job_name])
        max_hours = expectations.get(names[schedule.job_name])
        jobs.append(
            JobChip(
                name=schedule.job_name,
                state=_job_state(last, max_hours, now),
                last_success=last,
                max_hours=max_hours,
                next_run=schedule.next_fire_time,
                running=schedule.running > 0,
            )
        )
    sources = [
        SourceChip(
            label=f"{f.source_key} {sport_key_to_suffix(f.sport_key) or f.sport_key} {f.market}",
            sport=sport_key_to_suffix(f.sport_key) or f.sport_key,
            state=_source_state(f),
            last_snapshot=f.last_snapshot,
            threshold_hours=f.threshold_hours,
            active_hours=f.active_hours,
        )
        for f in await source_freshness(session, settings, now)
    ]
    reader = OddsReader(session)
    quotas = [
        QuotaChip(index, q, await reader.get_quota_burn(q, now))
        for index, q in (await reader.get_latest_quota_by_key(settings.api)).items()
    ]
    return StatusStrip(now, jobs, sources, quotas)
