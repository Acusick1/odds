"""Read-only views of collection freshness and job heartbeats, with no alerting.

The health monitor alerts on them and the web UI's status strip shows them, so both judge a
source or a job by the same rule.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from odds_core.config import Settings
from odds_core.job_specs import JOB_SPECS, heartbeat_name
from odds_core.models import AlertHistory
from odds_core.vocabulary import Market
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_lambda.collection_calendar import (
    CollectionSource,
    active_hours_between,
    fixture_calendar,
    latest_snapshot_time,
)
from odds_lambda.scheduling.jobs import (
    get_job_cron,
    make_compound_job_name,
    resolve_job_name,
    resolve_target_sports,
)
from odds_lambda.tier_utils import calculate_tier

HEARTBEAT_PREFIX = "heartbeat:"
SCHEDULE_OVERDUE_GRACE = timedelta(minutes=15)

# Self-scheduling jobs that return before booking a next run while disabled.
JOB_ENABLED: dict[str, Callable[[Settings], bool]] = {
    "fetch-betfair-exchange": lambda settings: settings.betfair.enabled,
    "fetch-polymarket-odds": lambda settings: settings.polymarket.enabled,
}


def collection_sources() -> tuple[CollectionSource, ...]:
    from odds_lambda.jobs import (
        fetch_betfair_exchange,
        fetch_odds,
        fetch_oddsportal,
        fetch_polymarket_odds,
    )

    return (
        fetch_oddsportal.SOURCE,
        fetch_betfair_exchange.SOURCE,
        fetch_odds.SOURCE,
        fetch_polymarket_odds.SOURCE,
    )


class Freshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    # No scheduled fixture inside the source's horizon, so no snapshot is due.
    IDLE = "idle"
    # The source has never written the market, so there is no gap to measure.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceFreshness:
    """One market of one sport a bootstrapped collection source writes.

    ``active_hours`` and ``threshold_hours`` are set when a fixture is due and the market has
    history: the gap without the overnight window, and the most it may be.
    """

    source_key: str
    job_name: str
    sport_key: str
    market: Market
    status: Freshness
    last_snapshot: datetime | None
    active_hours: float | None = None
    threshold_hours: float | None = None


async def source_freshness(
    session: AsyncSession, settings: Settings, now: datetime
) -> list[SourceFreshness]:
    """Each market a bootstrapped source writes for each of its sports, and whether it is stale.

    A market is checked only when a scheduled fixture lies inside the source's horizon for
    that sport (else ``IDLE``) and the source has written it before (else ``UNKNOWN``). The gap
    is the time since the later of the market's last snapshot and the moment the next fixture
    entered the horizon, counted without the sport's overnight window. It is ``STALE`` past the
    largest of ``stale_data_hours``, ``stale_cadence_multiplier`` polling intervals at the tier
    of the next kickoff, and ``stale_last_interval_multiplier`` intervals at the tier the next
    kickoff had when the last snapshot was written (the interval the job then chose to wait).
    """
    alerts = settings.alerts
    bootstrapped = set(settings.scheduler.bootstrap_jobs)
    found: list[SourceFreshness] = []
    for source in collection_sources():
        if source.job_name not in bootstrapped:
            continue
        for sport_key in resolve_target_sports(source.job_name, settings.data_collection.sports):
            if sport_key not in source.fixture_horizons:
                continue
            job_name = make_compound_job_name(source.job_name, sport_key)
            horizon = source.fixture_horizon(sport_key)
            calendar = await fixture_calendar(session, sport_key, now=now, horizon=horizon)
            # The next kick-off, when a fixture inside the horizon makes a snapshot due.
            due = calendar.next_kickoff if calendar.kickoffs_in_horizon else None
            for market in source.markets[sport_key]:
                last = await latest_snapshot_time(session, sport_key, source.source, market)
                status, active, threshold = Freshness.IDLE, None, None
                if due is not None and last is None:
                    status = Freshness.UNKNOWN
                elif due is not None and last is not None:
                    hours_to_kickoff = (due - now).total_seconds() / 3600
                    hours_at_last = (due - last).total_seconds() / 3600
                    threshold = max(
                        alerts.stale_data_hours,
                        alerts.stale_cadence_multiplier
                        * source.cadence.interval_for(calculate_tier(hours_to_kickoff)),
                        alerts.stale_last_interval_multiplier
                        * source.cadence.interval_for(calculate_tier(hours_at_last)),
                    )
                    reference = max(last, due - horizon)
                    active = active_hours_between(
                        reference, now, source.overnight_window(sport_key)
                    )
                    status = Freshness.STALE if active > threshold else Freshness.FRESH
                found.append(
                    SourceFreshness(
                        source.key, job_name, sport_key, market, status, last, active, threshold
                    )
                )
    return found


async def last_heartbeats(session: AsyncSession, job_names: Collection[str]) -> dict[str, datetime]:
    """The newest ``heartbeat:<job>`` row of each job that has one, by job name."""
    if not job_names:
        return {}
    rows = await session.execute(
        select(col(AlertHistory.alert_type), func.max(col(AlertHistory.sent_at)))
        .where(col(AlertHistory.alert_type).in_([f"{HEARTBEAT_PREFIX}{n}" for n in job_names]))
        .group_by(col(AlertHistory.alert_type))
    )
    return {alert_type.removeprefix(HEARTBEAT_PREFIX): sent for alert_type, sent in rows.all()}


def heartbeat_late(last_beat: datetime, max_hours: float, now: datetime) -> bool:
    """Whether a job's newest heartbeat is older than the hours its expectation allows."""
    return last_beat < now - timedelta(hours=max_hours)


def schedule_heartbeat_name(schedule_id: str) -> str:
    """The name a scheduled job records its heartbeat under.

    A job with declared heartbeats (``odds_core.job_specs.Heartbeat``) records the one of its
    sport, which for a few jobs is a legacy bare name; any other job records its schedule id.
    """
    base_name, sport = resolve_job_name(schedule_id)
    if base_name in JOB_SPECS:
        try:
            return heartbeat_name(base_name, sport)
        except ValueError:
            pass
    return schedule_id


@dataclass(frozen=True, slots=True)
class DeadChain:
    """A bootstrapped self-scheduling job with no run booked, or one long overdue."""

    job_name: str
    next_run: datetime | None

    @property
    def alert_type(self) -> str:
        return f"dead_schedule_chain:{self.job_name}"

    def describe(self) -> str:
        if self.next_run is None:
            return f"{self.job_name} has no scheduled run and is not running"
        return f"{self.job_name} run due at {self.next_run:%Y-%m-%d %H:%M} UTC has not started"


def dead_chains(
    expected: Iterable[str],
    next_runs: Mapping[str, datetime | None],
    outstanding: Collection[str],
    settings: Settings,
    now: datetime,
) -> list[DeadChain]:
    """The expected self-scheduling jobs whose chain has ended or stalled, by job name.

    A job is skipped when it is a cron job (APScheduler re-fires it), is outstanding (a released,
    unfinished run books its successor when it ends), or is disabled by a setting. Any other job
    is dead when it has no next run or its next run passed more than ``SCHEDULE_OVERDUE_GRACE``
    ago.
    """
    dead: list[DeadChain] = []
    for job_name in sorted(expected):
        base_name, _ = resolve_job_name(job_name)
        if get_job_cron(base_name) is not None or job_name in outstanding:
            continue
        enabled = JOB_ENABLED.get(base_name)
        if enabled is not None and not enabled(settings):
            continue
        next_run = next_runs.get(job_name)
        if next_run is None or next_run < now - SCHEDULE_OVERDUE_GRACE:
            dead.append(DeadChain(job_name, next_run))
    return dead
