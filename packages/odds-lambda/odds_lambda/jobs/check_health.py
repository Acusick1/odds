"""
Health check job - standalone executable or Lambda handler.

This job:
1. Checks system health metrics (stale data, failures, quota, quality)
2. Sends alerts when thresholds are breached
3. Self-schedules next execution (60-minute intervals)
"""

import asyncio

import structlog
from odds_core.config import get_settings

from odds_lambda.health_monitor import check_system_health
from odds_lambda.scheduling.backends import get_scheduler_backend

logger = structlog.get_logger()


async def main():
    """
    Main job execution flow.

    Flow:
    1. Run health checks
    2. Log results
    3. Schedule next execution (60 minutes)
    """
    app_settings = get_settings()

    logger.info("health_check_job_started", backend=app_settings.scheduler.backend)

    try:
        # Execute health check
        health_status = await check_system_health()

        # Log results
        logger.info(
            "health_check_results",
            overall_healthy=health_status.overall_healthy,
            issues_detected=len(health_status.issues_detected),
            alerts_sent=len(health_status.alerts_sent),
            success_rate_24h=health_status.metrics.fetch_success_rate_24h,
            hours_since_last_fetch=health_status.metrics.hours_since_last_fetch,
            quota_remaining=health_status.metrics.api_quota_remaining,
            consecutive_failures=health_status.metrics.consecutive_failures,
        )

        if not health_status.overall_healthy:
            logger.warning(
                "health_check_issues_detected",
                issues=health_status.issues_detected,
                alerts_sent=health_status.alerts_sent,
            )

    except Exception as e:
        logger.error("health_check_job_failed", error=str(e), exc_info=True)

        # Send critical alert
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            await send_critical(f"🚨 Health check job failed: {type(e).__name__}: {str(e)}")

        # Don't schedule next run if we failed - let manual intervention happen
        raise

    # Self-schedule next execution (60 minutes from now)
    try:
        from datetime import UTC, datetime, timedelta

        next_execution = datetime.now(UTC) + timedelta(minutes=60)

        backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
        await backend.schedule_next_execution(job_name="check-health", next_time=next_execution)

        logger.info(
            "health_check_next_scheduled",
            next_time=next_execution.isoformat(),
            backend=backend.get_backend_name(),
        )

    except Exception as e:
        logger.error("health_check_scheduling_failed", error=str(e), exc_info=True)

        # Send error alert
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_error

            await send_error(f"Health check scheduling failed: {type(e).__name__}: {str(e)}")

        # Don't fail the job if scheduling fails - the health check itself succeeded


if __name__ == "__main__":
    asyncio.run(main())
