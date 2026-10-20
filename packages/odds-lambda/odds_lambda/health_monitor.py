"""
System health monitoring with automated checks and alerting.

This module implements proactive health monitoring to detect:
- Stale data (no recent fetches)
- Low API quota
- Data quality degradation
- Database connectivity issues

Stale data is detected via OddsSnapshot recency; other checks use OddsReader.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog
from odds_core.betfair_close import LIVE_CLOSE_WINDOW, complete_exchange_prices, is_in_play
from odds_core.config import Settings, get_settings
from odds_core.database import async_session_maker
from odds_core.market_prices import THREE_WAY_MARKET, outcome_names
from odds_core.models import AlertHistory, Event, EventStatus, OddsSnapshot
from odds_core.snapshot_sources import SnapshotKind, SnapshotSource
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.collection_calendar import (
    CollectionSource,
    active_hours_between,
    fixture_calendar,
    latest_snapshot_time,
)
from odds_lambda.scheduling.backends.base import ScheduledJob
from odds_lambda.scheduling.decision import classify_tier
from odds_lambda.scheduling.jobs import (
    expected_compound_job_names,
    get_job_cron,
    make_compound_job_name,
    resolve_job_name,
    resolve_target_sports,
)
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.unsettled_events import find_unsettled_events

logger = structlog.get_logger()

LEAGUE_SPORTS: tuple[str, ...] = ("soccer_epl",)
UNSETTLED_AGE = timedelta(hours=24)
UNSETTLED_IDS_SHOWN = 10
UNSETTLED_ALERT_MINUTES = 24 * 60
BETFAIR_CLOSE_SPORT = "soccer_epl"
BETFAIR_CLOSE_JOB = "fetch-betfair-exchange"
BETFAIR_HISTORY_WINDOW = timedelta(hours=72)
BETFAIR_CLOSE_ALERT_TYPE = "missing_betfair_close"
SCHEDULE_OVERDUE_GRACE = timedelta(minutes=15)

# Self-scheduling jobs that return before booking a next run while disabled.
_JOB_ENABLED: dict[str, Callable[[Settings], bool]] = {
    "fetch-betfair-exchange": lambda settings: settings.betfair.enabled,
    "fetch-polymarket-odds": lambda settings: settings.polymarket.enabled,
    "fetch-polymarket": lambda settings: settings.polymarket.enabled,
}


class ScheduleReader(Protocol):
    async def get_scheduled_jobs(self) -> list[ScheduledJob]: ...

    async def get_outstanding_job_names(self) -> set[str]: ...


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


@dataclass(frozen=True, slots=True)
class StaleSource:
    """A collection source that has written no rows for longer than its cadence allows."""

    source: str
    sport_key: str
    last_snapshot: datetime
    active_hours: float
    threshold_hours: float

    @property
    def alert_type(self) -> str:
        return f"stale_data:{self.source}:{self.sport_key}"

    def describe(self) -> str:
        return (
            f"{self.source} has written no {self.sport_key} snapshot in "
            f"{self.active_hours:.1f} active hours (threshold: {self.threshold_hours:.1f}h, "
            f"last: {self.last_snapshot.isoformat()})"
        )


@dataclass(frozen=True, slots=True)
class MissingBetfairClose:
    """A played EPL event with no live Betfair snapshot in the window before kick-off."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime

    def describe(self) -> str:
        return (
            f"{self.home_team} v {self.away_team} "
            f"({self.commence_time:%Y-%m-%d %H:%M} UTC, {self.event_id})"
        )


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


class HealthMetrics(BaseModel):
    """Health check metrics."""

    hours_since_last_fetch: float | None
    api_quota_remaining: int | None
    data_quality_errors_24h: int
    events_scheduled: int
    events_live: int
    events_final: int


class HealthStatus(BaseModel):
    """Health check result."""

    timestamp: datetime
    overall_healthy: bool
    metrics: HealthMetrics
    alerts_sent: list[str]
    issues_detected: list[str]


class HealthMonitor:
    """Monitors system health and dispatches alerts."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        schedules: ScheduleReader | None = None,
    ):
        """
        Initialize health monitor.

        Args:
            session: Database session
            settings: Application settings
            schedules: Scheduler store to check self-scheduling chains against;
                without one the chain check is skipped
        """
        self.session = session
        self.settings = settings
        self.schedules = schedules

    async def _should_send_alert(
        self, alert_type: str, rate_limit_minutes: int | None = None
    ) -> bool:
        """
        Check if alert should be sent based on rate limiting.

        Args:
            alert_type: Alert type identifier

        Returns:
            True if alert should be sent (no recent alert of this type)
        """
        if rate_limit_minutes is None:
            rate_limit_minutes = self.settings.alerts.alert_rate_limit_minutes
        cutoff_time = datetime.now(UTC) - timedelta(minutes=rate_limit_minutes)

        # Query for recent alerts of this type
        query = select(func.count(AlertHistory.id)).where(
            and_(
                AlertHistory.alert_type == alert_type,
                AlertHistory.sent_at >= cutoff_time,
            )
        )

        result = await self.session.execute(query)
        recent_count = result.scalar_one()

        return recent_count == 0

    async def _record_alert(
        self, alert_type: str, severity: str, message: str, context: dict | None = None
    ):
        """
        Record sent alert in database.

        Args:
            alert_type: Alert type identifier
            severity: Alert severity level
            message: Alert message
            context: Additional context data
        """
        alert_record = AlertHistory(
            alert_type=alert_type,
            severity=severity,
            message=message,
            context=context,
            sent_at=datetime.now(UTC),
        )

        self.session.add(alert_record)
        await self.session.commit()

        logger.info("alert_recorded", alert_type=alert_type, severity=severity)

    async def _send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        context: dict | None = None,
        rate_limit_minutes: int | None = None,
    ):
        """
        Send alert if rate limiting allows.

        Args:
            alert_type: Alert type identifier
            severity: Alert severity level
            message: Alert message
            context: Additional context data
            rate_limit_minutes: Overrides the configured window for this alert

        Returns:
            True if alert was sent
        """
        if not self.settings.alerts_deliverable:
            logger.debug("alert_skipped_disabled", alert_type=alert_type)
            return False

        if not await self._should_send_alert(alert_type, rate_limit_minutes):
            logger.debug(
                "alert_rate_limited",
                alert_type=alert_type,
                rate_limit_minutes=rate_limit_minutes
                or self.settings.alerts.alert_rate_limit_minutes,
            )
            return False

        # Send via alert system
        from odds_core.alerts import alert_manager

        await alert_manager.alert(message, severity)

        # Record in database
        await self._record_alert(alert_type, severity, message, context)

        return True

    async def _get_hours_since_latest_snapshot(self) -> float | None:
        """Query hours since the most recent OddsSnapshot, or None if empty."""
        query = select(func.max(OddsSnapshot.snapshot_time))
        result = await self.session.execute(query)
        latest_snapshot_time = result.scalar_one_or_none()
        if latest_snapshot_time is None:
            return None
        return (datetime.now(UTC) - latest_snapshot_time).total_seconds() / 3600

    async def _job_is_failing(self, job_name: str, now: datetime, recent: timedelta) -> bool:
        """Whether the job's latest failure alert is within ``recent`` and newer than its heartbeat."""
        latest: dict[str, datetime | None] = {}
        for kind in ("job_failure", "heartbeat"):
            latest[kind] = (
                await self.session.execute(
                    select(func.max(AlertHistory.sent_at)).where(
                        AlertHistory.alert_type == f"{kind}:{job_name}"
                    )
                )
            ).scalar_one_or_none()
        failure, heartbeat = latest["job_failure"], latest["heartbeat"]
        return (
            failure is not None
            and failure > now - recent
            and (heartbeat is None or failure > heartbeat)
        )

    async def check_stale_sources(self, now: datetime | None = None) -> list[StaleSource]:
        """Find sources that stopped writing while fixtures they should price are scheduled.

        Only sources whose job is bootstrapped are checked, per sport, and only when
        a scheduled fixture lies inside the source's horizon for that sport and the
        source has written before, so international breaks, off-seasons and jobs
        that were never enabled raise nothing. A source whose job is already
        failing is left to that job's own alert. The gap is the time since the
        later of the last snapshot and the moment the next fixture entered the
        horizon, counted without the sport's overnight window. It is compared with
        the largest of ``stale_data_hours``, ``stale_cadence_multiplier`` polling
        intervals at the tier of the next kickoff, and 1.5 intervals at the tier
        the next kickoff had when the last snapshot was written (the interval the
        job then chose to wait).
        """
        moment = now or datetime.now(UTC)
        alerts = self.settings.alerts
        bootstrapped = set(self.settings.scheduler.bootstrap_jobs)
        stale: list[StaleSource] = []
        for source in collection_sources():
            if source.job_name not in bootstrapped:
                continue
            for sport_key in resolve_target_sports(
                source.job_name, self.settings.data_collection.sports
            ):
                if sport_key not in source.fixture_horizons:
                    continue
                horizon = source.fixture_horizon(sport_key)
                calendar = await fixture_calendar(
                    self.session, sport_key, now=moment, horizon=horizon
                )
                if calendar.next_kickoff is None or not calendar.kickoffs_in_horizon:
                    continue
                last = await latest_snapshot_time(self.session, sport_key, source.snapshot_source)
                if last is None:
                    logger.info("stale_check_no_history", source=source.key, sport=sport_key)
                    continue
                hours_to_kickoff = (calendar.next_kickoff - moment).total_seconds() / 3600
                hours_at_last = (calendar.next_kickoff - last).total_seconds() / 3600
                threshold = max(
                    alerts.stale_data_hours,
                    alerts.stale_cadence_multiplier
                    * source.cadence.interval_for(classify_tier(hours_to_kickoff)),
                    alerts.stale_last_interval_multiplier
                    * source.cadence.interval_for(classify_tier(hours_at_last)),
                )
                reference = max(last, calendar.next_kickoff - horizon)
                active = active_hours_between(reference, moment, source.overnight_window(sport_key))
                if active <= threshold:
                    continue
                recent = timedelta(hours=max(threshold, alerts.stale_reminder_minutes / 60))
                if await self._job_is_failing(
                    make_compound_job_name(source.job_name, sport_key), moment, recent
                ):
                    logger.info("stale_source_job_failing", source=source.key, sport=sport_key)
                    continue
                stale.append(
                    StaleSource(
                        source=source.key,
                        sport_key=sport_key,
                        last_snapshot=last,
                        active_hours=active,
                        threshold_hours=threshold,
                    )
                )
        return stale

    async def check_missing_betfair_closes(
        self, now: datetime | None = None
    ) -> list[MissingBetfairClose]:
        """Find recent EPL events whose ADR-0012 live Betfair close was never collected.

        An event is checked from its kick-off until ``betfair_close_lookback_hours``
        later, when the Betfair job is bootstrapped and has written EPL snapshots
        before. It is skipped when its status is postponed or cancelled, or when it
        has no live Betfair snapshot in the ``BETFAIR_HISTORY_WINDOW`` before
        kick-off (a ghost, a twin that carries the prices, or an unannounced
        postponement). It is missing when no non-in-play live Betfair snapshot with
        a price for every outcome lies within ``LIVE_CLOSE_WINDOW`` before kick-off,
        the same definition the retail-EV builder uses for its close. Events already
        named in an alert inside the lookback are not reported again.
        """
        moment = now or datetime.now(UTC)
        if (
            not self.settings.betfair.enabled
            or BETFAIR_CLOSE_JOB not in self.settings.scheduler.bootstrap_jobs
        ):
            return []
        if BETFAIR_CLOSE_SPORT not in resolve_target_sports(
            BETFAIR_CLOSE_JOB, self.settings.data_collection.sports
        ):
            return []
        if (
            await latest_snapshot_time(
                self.session, BETFAIR_CLOSE_SPORT, SnapshotSource.BETFAIR_API
            )
            is None
        ):
            logger.info("betfair_close_no_history", sport=BETFAIR_CLOSE_SPORT)
            return []
        lookback = timedelta(hours=self.settings.alerts.betfair_close_lookback_hours)
        events = (
            (
                await self.session.execute(
                    select(Event)
                    .where(
                        Event.sport_key == BETFAIR_CLOSE_SPORT,
                        Event.commence_time <= moment,
                        Event.commence_time > moment - lookback,
                        Event.status.notin_((EventStatus.POSTPONED, EventStatus.CANCELLED)),  # type: ignore[attr-defined]
                    )
                    .order_by(Event.commence_time)
                )
            )
            .scalars()
            .all()
        )
        already_alerted = await self._event_ids_alerted(BETFAIR_CLOSE_ALERT_TYPE, moment - lookback)
        tag = OddsSnapshot.raw_data["source"].as_string()  # type: ignore[index]
        missing: list[MissingBetfairClose] = []
        for event in events:
            if event.id in already_alerted:
                continue
            kickoff = event.commence_time
            betfair_before = (
                await self.session.execute(
                    select(OddsSnapshot.id)
                    .where(
                        OddsSnapshot.event_id == event.id,
                        tag == SnapshotSource.BETFAIR_API.value,
                        OddsSnapshot.snapshot_time >= kickoff - BETFAIR_HISTORY_WINDOW,
                        OddsSnapshot.snapshot_time <= kickoff,
                    )
                    .limit(1)
                )
            ).first()
            if betfair_before is None:
                continue
            closes = (
                await self.session.execute(
                    select(OddsSnapshot.raw_data).where(
                        OddsSnapshot.event_id == event.id,
                        tag == SnapshotSource.BETFAIR_API.value,
                        OddsSnapshot.snapshot_time >= kickoff - LIVE_CLOSE_WINDOW,
                        OddsSnapshot.snapshot_time <= kickoff,
                    )
                )
            ).scalars()
            names = outcome_names(event, THREE_WAY_MARKET)
            if any(
                not is_in_play(raw_data)
                and complete_exchange_prices(
                    raw_data, SnapshotKind.BETFAIR_LIVE, THREE_WAY_MARKET, names
                )
                is not None
                for raw_data in closes
            ):
                continue
            missing.append(
                MissingBetfairClose(
                    event_id=event.id,
                    home_team=event.home_team,
                    away_team=event.away_team,
                    commence_time=kickoff,
                )
            )
        return missing

    async def _event_ids_alerted(self, alert_type: str, since: datetime) -> set[str]:
        contexts = (
            await self.session.execute(
                select(AlertHistory.context).where(
                    AlertHistory.alert_type == alert_type, AlertHistory.sent_at >= since
                )
            )
        ).scalars()
        return {
            event_id for context in contexts for event_id in (context or {}).get("event_ids", [])
        }

    async def check_api_quota(self) -> tuple[bool, int | None, float | None]:
        """
        Check API quota levels.

        Returns:
            (is_healthy, quota_remaining, quota_fraction)
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)
        stats = await reader.get_database_stats()

        quota_remaining = stats.get("api_quota_remaining")

        if quota_remaining is None:
            # No quota data available yet
            return True, None, None

        quota_fraction = quota_remaining / self.settings.api.quota
        warning_threshold = self.settings.alerts.quota_warning_threshold

        is_healthy = quota_fraction >= warning_threshold

        return is_healthy, quota_remaining, quota_fraction

    async def check_data_quality(self) -> tuple[bool, int]:
        """
        Check data quality error rate.

        Returns:
            (is_healthy, error_count_24h)
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)

        # Get error/critical quality logs from last 24h
        start_time = datetime.now(UTC) - timedelta(hours=24)
        error_logs = await reader.get_data_quality_logs(
            severity="error",
            start_time=start_time,
            limit=1000,
        )

        critical_logs = await reader.get_data_quality_logs(
            severity="critical",
            start_time=start_time,
            limit=1000,
        )

        total_errors = len(error_logs) + len(critical_logs)

        # Check against configurable threshold (default: 10 errors per 24h)
        threshold = self.settings.alerts.data_quality_error_threshold
        is_healthy = total_errors < threshold

        return is_healthy, total_errors

    async def check_event_identity(self) -> list[str]:
        """
        Check that each match is one event.

        Each issue names its sport and is alerted separately; none merges anything.
        Run ``odds events merge-twins --sport <sport> --apply`` for twins.

        Returns:
            Issue descriptions, empty when healthy.
        """
        merger = EventMerger(self.session)
        issues: list[str] = []

        twins = Counter(group[0].sport_key for group in await merger.find_twin_groups())
        for sport, count in sorted(twins.items()):
            issues.append(f"{count} {sport} matches stored as more than one event")
            await self._send_alert(
                alert_type=f"duplicate_events:{sport}",
                severity="warning",
                message=f"⚠️ {count} {sport} matches stored as more than one event; "
                f"run `odds events merge-twins --sport {sport} --apply`",
                context={"sport_key": sport, "twin_matches": count},
            )

        for sport in LEAGUE_SPORTS:
            repeated = len(await merger.find_repeated_fixtures(sport))
            if repeated:
                issues.append(f"{repeated} {sport} fixtures stored twice in one season")
                await self._send_alert(
                    alert_type=f"repeated_fixtures:{sport}",
                    severity="warning",
                    message=f"⚠️ {repeated} {sport} fixtures stored twice in one season "
                    "(a rescheduled match under a second id)",
                    context={"sport_key": sport, "repeated_fixtures": repeated},
                )

        return issues

    async def check_unsettled_events(self, now: datetime) -> list[str]:
        """
        Check that no league event stays pending long after kick-off.

        Alerts per league sport, at most once a day, naming the events of the
        current season still SCHEDULED or LIVE more than ``UNSETTLED_AGE``
        after kick-off.

        Returns:
            Issue descriptions, empty when healthy.
        """
        issues: list[str] = []
        for sport in LEAGUE_SPORTS:
            events = await find_unsettled_events(self.session, sport, UNSETTLED_AGE, now)
            if not events:
                continue
            ids = [event.id for event in events]
            shown = ", ".join(ids[:UNSETTLED_IDS_SHOWN])
            more = (
                f" and {len(ids) - UNSETTLED_IDS_SHOWN} more"
                if len(ids) > UNSETTLED_IDS_SHOWN
                else ""
            )
            issue = f"{len(ids)} {sport} events unsettled over 24 h after kick-off: {shown}{more}"
            issues.append(issue)
            await self._send_alert(
                alert_type=f"unsettled_events:{sport}",
                severity="warning",
                message=f"⚠️ {issue}",
                context={"sport_key": sport, "event_ids": ids},
                rate_limit_minutes=UNSETTLED_ALERT_MINUTES,
            )
        return issues

    async def collect_metrics(self) -> HealthMetrics:
        """
        Collect all health metrics.

        Returns:
            HealthMetrics with current system state
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)
        stats = await reader.get_database_stats()

        # Calculate hours since last data arrived (any source)
        hours_since_last_fetch = await self._get_hours_since_latest_snapshot()

        # Get data quality errors
        _, data_quality_errors = await self.check_data_quality()

        return HealthMetrics(
            hours_since_last_fetch=hours_since_last_fetch,
            api_quota_remaining=stats.get("api_quota_remaining"),
            data_quality_errors_24h=data_quality_errors,
            events_scheduled=stats.get("events_by_status", {}).get("scheduled", 0),
            events_live=stats.get("events_by_status", {}).get("live", 0),
            events_final=stats.get("events_by_status", {}).get("final", 0),
        )

    async def check_job_heartbeats(self) -> list[str]:
        """Check that active jobs have reported heartbeats recently.

        Expectations are read from ``settings.alerts.heartbeat_expectations``.

        Returns:
            List of issue descriptions for jobs that missed their window.
        """
        expectations = self.settings.alerts.heartbeat_expectations
        issues: list[str] = []
        now = datetime.now(UTC)

        for job_name, max_hours in expectations.items():
            cutoff = now - timedelta(hours=max_hours)
            query = select(func.max(AlertHistory.sent_at)).where(
                AlertHistory.alert_type == f"heartbeat:{job_name}"
            )
            result = await self.session.execute(query)
            last_beat = result.scalar_one_or_none()

            if last_beat is None:
                # No heartbeat history — job hasn't run since monitoring was
                # deployed.  Log but don't alert to avoid noise on first deploy.
                logger.info("heartbeat_no_history", job=job_name)
                continue

            if last_beat < cutoff:
                hours_ago = f"{(now - last_beat).total_seconds() / 3600:.1f}h ago"
                issue = (
                    f"Job {job_name} has not completed "
                    f"(last: {hours_ago}, expected every {max_hours:.0f}h)"
                )
                issues.append(issue)

                await self._send_alert(
                    alert_type=f"missing_heartbeat:{job_name}",
                    severity="warning",
                    message=f"⚠️ {issue}",
                )

        return issues

    async def check_schedule_chains(self, now: datetime | None = None) -> list[DeadChain]:
        """Bootstrapped self-scheduling jobs whose chain has ended or stalled."""
        if self.schedules is None:
            return []
        now = now or datetime.now(UTC)
        expected = expected_compound_job_names(
            self.settings.scheduler.bootstrap_jobs, self.settings.data_collection.sports
        )
        next_runs = {
            job.job_name: job.next_run_time for job in await self.schedules.get_scheduled_jobs()
        }
        outstanding = await self.schedules.get_outstanding_job_names()

        dead: list[DeadChain] = []
        for job_name in sorted(expected):
            base_name, _ = resolve_job_name(job_name)
            if get_job_cron(base_name) is not None or job_name in outstanding:
                continue
            enabled = _JOB_ENABLED.get(base_name)
            if enabled is not None and not enabled(self.settings):
                continue
            next_run = next_runs.get(job_name)
            if next_run is None or next_run < now - SCHEDULE_OVERDUE_GRACE:
                dead.append(DeadChain(job_name, next_run))
        return dead

    async def purge_old_heartbeats(self) -> int:
        """Delete heartbeat rows older than the configured retention period.

        Returns:
            Number of rows deleted.
        """
        from sqlalchemy import delete

        cutoff = datetime.now(UTC) - timedelta(days=self.settings.alerts.heartbeat_retention_days)
        result = await self.session.execute(
            delete(AlertHistory).where(
                AlertHistory.alert_type.startswith("heartbeat:"),
                AlertHistory.sent_at < cutoff,
            )
        )
        deleted = result.rowcount
        if deleted:
            await self.session.commit()
            logger.info("heartbeats_purged", deleted=deleted)
        return deleted

    async def check_system_health(self, now: datetime | None = None) -> HealthStatus:
        """
        Perform comprehensive system health check.

        Checks:
        1. Stale data per source (no rows within the source's cadence while fixtures are scheduled),
           and EPL events with no live Betfair close
        2. API quota levels
        3. Data quality issues
        4. Job heartbeats (missing completions)
        5. Event identity: twins, repeated fixtures, stale scheduled events
        6. Self-scheduling chains with no run booked or an overdue run

        Returns:
            HealthStatus with metrics and alerts sent
        """
        logger.info("health_check_started")
        moment = now or datetime.now(UTC)

        issues_detected = []
        alerts_sent = []

        try:
            # Collect metrics
            metrics = await self.collect_metrics()

            # Check 1: Stale data, per source
            alerts_config = self.settings.alerts
            for stale in await self.check_stale_sources(moment):
                issue = stale.describe()
                issues_detected.append(issue)

                if await self._send_alert(
                    alert_type=stale.alert_type,
                    severity="critical",
                    message=f"🚨 Stale data: {issue}",
                    rate_limit_minutes=alerts_config.stale_reminder_minutes,
                ):
                    alerts_sent.append(stale.alert_type)

            missing_closes = await self.check_missing_betfair_closes(moment)
            if missing_closes:
                listing = "; ".join(close.describe() for close in missing_closes)
                issue = (
                    f"{len(missing_closes)} {BETFAIR_CLOSE_SPORT} events have no live Betfair "
                    f"snapshot within {LIVE_CLOSE_WINDOW} before kick-off: {listing}"
                )
                issues_detected.append(issue)
                if await self._send_alert(
                    alert_type=BETFAIR_CLOSE_ALERT_TYPE,
                    severity="critical",
                    message=f"🚨 {issue}",
                    context={"event_ids": [close.event_id for close in missing_closes]},
                    rate_limit_minutes=alerts_config.alert_rate_limit_minutes,
                ):
                    alerts_sent.append(BETFAIR_CLOSE_ALERT_TYPE)

            # Check 2: API quota
            quota_healthy, quota_remaining, quota_fraction = await self.check_api_quota()

            if quota_remaining is not None and quota_fraction is not None:
                critical_threshold = self.settings.alerts.quota_critical_threshold
                warning_threshold = self.settings.alerts.quota_warning_threshold
                percentage_remaining = round(quota_fraction * 100, 1)

                if quota_fraction < critical_threshold:
                    issue = (
                        f"API quota critical: {quota_remaining} requests ({percentage_remaining}%)"
                    )
                    issues_detected.append(issue)

                    if await self._send_alert(
                        alert_type="quota_critical",
                        severity="critical",
                        message=f"🚨 {issue}",
                        context={
                            "quota_remaining": quota_remaining,
                            "quota_fraction": quota_fraction,
                        },
                    ):
                        alerts_sent.append("quota_critical")

                elif quota_fraction < warning_threshold:
                    issue = f"API quota low: {quota_remaining} requests ({percentage_remaining}%)"
                    issues_detected.append(issue)

                    if await self._send_alert(
                        alert_type="quota_low",
                        severity="warning",
                        message=f"⚠️ {issue}",
                        context={
                            "quota_remaining": quota_remaining,
                            "quota_fraction": quota_fraction,
                        },
                    ):
                        alerts_sent.append("quota_low")

            # Check 3: Data quality
            quality_healthy, error_count = await self.check_data_quality()
            if not quality_healthy:
                issue = f"High data quality error rate: {error_count} errors in 24h"
                issues_detected.append(issue)

                if await self._send_alert(
                    alert_type="data_quality_errors",
                    severity="warning",
                    message=f"⚠️ {issue}",
                    context={"error_count_24h": error_count},
                ):
                    alerts_sent.append("data_quality_errors")

            # Check 4: Job heartbeats
            heartbeat_issues = await self.check_job_heartbeats()
            issues_detected.extend(heartbeat_issues)

            # Check 5: One event per match
            issues_detected.extend(await self.check_event_identity())

            # Check 5b: League events settle after kick-off
            issues_detected.extend(await self.check_unsettled_events(moment))

            # Check 6: Self-scheduling chains
            for chain in await self.check_schedule_chains(moment):
                issue = chain.describe()
                issues_detected.append(issue)
                if await self._send_alert(
                    alert_type=chain.alert_type,
                    severity="critical",
                    message=f"🚨 Dead schedule chain: {issue}",
                ):
                    alerts_sent.append(chain.alert_type)

            # Housekeeping: purge old heartbeat rows
            await self.purge_old_heartbeats()

            overall_healthy = len(issues_detected) == 0

            logger.info(
                "health_check_completed",
                overall_healthy=overall_healthy,
                issues_count=len(issues_detected),
                alerts_sent_count=len(alerts_sent),
            )

            return HealthStatus(
                timestamp=datetime.now(UTC),
                overall_healthy=overall_healthy,
                metrics=metrics,
                alerts_sent=alerts_sent,
                issues_detected=issues_detected,
            )

        except Exception as e:
            logger.error("health_check_failed", error=str(e), exc_info=True)

            # Send critical alert about health check failure
            if self.settings.alerts.alert_enabled:
                try:
                    await self._send_alert(
                        alert_type="health_check_failure",
                        severity="critical",
                        message=f"🚨 Health check failed: {type(e).__name__}: {str(e)}",
                        context={"error_type": type(e).__name__, "error_message": str(e)},
                    )
                except Exception as alert_error:
                    logger.error(
                        "failed_to_send_health_check_failure_alert", error=str(alert_error)
                    )

            # Return unhealthy status
            return HealthStatus(
                timestamp=datetime.now(UTC),
                overall_healthy=False,
                metrics=HealthMetrics(
                    hours_since_last_fetch=None,
                    api_quota_remaining=None,
                    data_quality_errors_24h=0,
                    events_scheduled=0,
                    events_live=0,
                    events_final=0,
                ),
                alerts_sent=[],
                issues_detected=[f"Health check failed: {str(e)}"],
            )


async def check_system_health() -> HealthStatus:
    """
    Standalone health check function for job execution.

    This is the main entry point called by the scheduler.

    Returns:
        HealthStatus with results
    """
    settings = get_settings()
    schedules: ScheduleReader | None = None
    if settings.scheduler.backend == "local":
        from odds_lambda.scheduling.backends.local import LocalSchedulerBackend

        schedules = LocalSchedulerBackend()

    async with async_session_maker() as session:
        monitor = HealthMonitor(session, settings, schedules)
        return await monitor.check_system_health()
