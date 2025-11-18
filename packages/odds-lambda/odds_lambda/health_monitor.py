"""
System health monitoring with automated checks and alerting.

This module implements proactive health monitoring to detect:
- Stale data (no recent fetches)
- Consecutive failures
- Low API quota
- Data quality degradation
- Database connectivity issues

All checks use existing OddsReader methods to query database state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import Settings, get_settings
from odds_core.models import AlertHistory
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_core.database import get_session

logger = structlog.get_logger()


class HealthMetrics(BaseModel):
    """Health check metrics."""

    fetch_success_rate_24h: float
    hours_since_last_fetch: float | None
    api_quota_remaining: int | None
    consecutive_failures: int
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

    def __init__(self, session: AsyncSession, settings: Settings):
        """
        Initialize health monitor.

        Args:
            session: Database session
            settings: Application settings
        """
        self.session = session
        self.settings = settings

    async def _should_send_alert(self, alert_type: str) -> bool:
        """
        Check if alert should be sent based on rate limiting.

        Args:
            alert_type: Alert type identifier

        Returns:
            True if alert should be sent (no recent alert of this type)
        """
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

    async def _send_alert(self, alert_type: str, severity: str, message: str, context: dict | None = None):
        """
        Send alert if rate limiting allows.

        Args:
            alert_type: Alert type identifier
            severity: Alert severity level
            message: Alert message
            context: Additional context data

        Returns:
            True if alert was sent
        """
        if not self.settings.alerts.alert_enabled:
            logger.debug("alert_skipped_disabled", alert_type=alert_type)
            return False

        if not await self._should_send_alert(alert_type):
            logger.debug(
                "alert_rate_limited",
                alert_type=alert_type,
                rate_limit_minutes=self.settings.alerts.alert_rate_limit_minutes,
            )
            return False

        # Send via alert system
        from odds_cli.alerts.base import alert_manager

        await alert_manager.alert(message, severity)

        # Record in database
        await self._record_alert(alert_type, severity, message, context)

        return True

    async def check_stale_data(self) -> tuple[bool, str | None]:
        """
        Check for stale data (no recent fetches).

        Returns:
            (is_healthy, issue_description)
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)

        # Get most recent fetch
        fetch_logs = await reader.get_fetch_logs(limit=1)

        if not fetch_logs:
            return False, "No fetch logs found in database"

        last_fetch = fetch_logs[0]
        hours_since_fetch = (datetime.now(UTC) - last_fetch.fetch_time).total_seconds() / 3600

        threshold_hours = self.settings.alerts.stale_data_hours

        if hours_since_fetch > threshold_hours:
            return False, f"No data fetched in {hours_since_fetch:.1f} hours (threshold: {threshold_hours}h)"

        return True, None

    async def check_consecutive_failures(self) -> tuple[bool, int]:
        """
        Check for consecutive fetch failures.

        Returns:
            (is_healthy, consecutive_failure_count)
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)

        # Get recent fetch logs
        fetch_logs = await reader.get_fetch_logs(limit=10)

        if not fetch_logs:
            return True, 0

        # Count consecutive failures from most recent
        consecutive_failures = 0
        for log in fetch_logs:
            if log.success:
                break
            consecutive_failures += 1

        threshold = self.settings.alerts.consecutive_failures_threshold

        is_healthy = consecutive_failures < threshold

        return is_healthy, consecutive_failures

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
        critical_threshold = self.settings.alerts.quota_critical_threshold
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

    async def collect_metrics(self) -> HealthMetrics:
        """
        Collect all health metrics.

        Returns:
            HealthMetrics with current system state
        """
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(self.session)
        stats = await reader.get_database_stats()

        # Calculate hours since last fetch
        fetch_logs = await reader.get_fetch_logs(limit=1)
        hours_since_last_fetch = None
        if fetch_logs:
            hours_since_last_fetch = (
                datetime.now(UTC) - fetch_logs[0].fetch_time
            ).total_seconds() / 3600

        # Get consecutive failures
        _, consecutive_failures = await self.check_consecutive_failures()

        # Get data quality errors
        _, data_quality_errors = await self.check_data_quality()

        return HealthMetrics(
            fetch_success_rate_24h=stats.get("fetch_success_rate_24h", 0.0),
            hours_since_last_fetch=hours_since_last_fetch,
            api_quota_remaining=stats.get("api_quota_remaining"),
            consecutive_failures=consecutive_failures,
            data_quality_errors_24h=data_quality_errors,
            events_scheduled=stats.get("events_by_status", {}).get("scheduled", 0),
            events_live=stats.get("events_by_status", {}).get("live", 0),
            events_final=stats.get("events_by_status", {}).get("final", 0),
        )

    async def check_system_health(self) -> HealthStatus:
        """
        Perform comprehensive system health check.

        Checks:
        1. Stale data (no recent fetches)
        2. Consecutive failures
        3. API quota levels
        4. Data quality issues

        Returns:
            HealthStatus with metrics and alerts sent
        """
        logger.info("health_check_started")

        issues_detected = []
        alerts_sent = []

        try:
            # Collect metrics
            metrics = await self.collect_metrics()

            # Check 1: Stale data
            stale_healthy, stale_issue = await self.check_stale_data()
            if not stale_healthy and stale_issue:
                issues_detected.append(stale_issue)

                if await self._send_alert(
                    alert_type="stale_data",
                    severity="warning",
                    message=f"⚠️ Stale data detected: {stale_issue}",
                ):
                    alerts_sent.append("stale_data")

            # Check 2: Consecutive failures
            failures_healthy, failure_count = await self.check_consecutive_failures()
            if not failures_healthy:
                issue = f"{failure_count} consecutive fetch failures (threshold: {self.settings.alerts.consecutive_failures_threshold})"
                issues_detected.append(issue)

                if await self._send_alert(
                    alert_type="consecutive_failures",
                    severity="error",
                    message=f"🚨 Multiple fetch failures: {issue}",
                    context={"failure_count": failure_count},
                ):
                    alerts_sent.append("consecutive_failures")

            # Check 3: API quota
            quota_healthy, quota_remaining, quota_fraction = await self.check_api_quota()

            if quota_remaining is not None and quota_fraction is not None:
                critical_threshold = self.settings.alerts.quota_critical_threshold
                warning_threshold = self.settings.alerts.quota_warning_threshold
                percentage_remaining = round(quota_fraction * 100, 1)

                if quota_fraction < critical_threshold:
                    issue = f"API quota critical: {quota_remaining} requests ({percentage_remaining}%)"
                    issues_detected.append(issue)

                    if await self._send_alert(
                        alert_type="quota_critical",
                        severity="critical",
                        message=f"🚨 {issue}",
                        context={"quota_remaining": quota_remaining, "quota_fraction": quota_fraction},
                    ):
                        alerts_sent.append("quota_critical")

                elif quota_fraction < warning_threshold:
                    issue = f"API quota low: {quota_remaining} requests ({percentage_remaining}%)"
                    issues_detected.append(issue)

                    if await self._send_alert(
                        alert_type="quota_low",
                        severity="warning",
                        message=f"⚠️ {issue}",
                        context={"quota_remaining": quota_remaining, "quota_fraction": quota_fraction},
                    ):
                        alerts_sent.append("quota_low")

            # Check 4: Data quality
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
                    logger.error("failed_to_send_health_check_failure_alert", error=str(alert_error))

            # Return unhealthy status
            return HealthStatus(
                timestamp=datetime.now(UTC),
                overall_healthy=False,
                metrics=HealthMetrics(
                    fetch_success_rate_24h=0.0,
                    hours_since_last_fetch=None,
                    api_quota_remaining=None,
                    consecutive_failures=0,
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

    async with get_session() as session:
        monitor = HealthMonitor(session, settings)
        return await monitor.check_system_health()
