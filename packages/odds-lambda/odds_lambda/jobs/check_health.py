"""
Health check job - standalone executable or Lambda handler.

Checks system health metrics (stale data, failures, quota, quality)
and sends alerts when thresholds are breached.
"""

import asyncio

import structlog

from odds_lambda.health_monitor import check_system_health
from odds_lambda.scheduling.jobs import JobContext

logger = structlog.get_logger()


async def main(ctx: JobContext) -> None:
    """Run health checks and log results."""
    from odds_core.alerts import job_alert_context

    logger.info("health_check_job_started")

    async with job_alert_context("check-health"):
        health_status = await check_system_health()

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


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
