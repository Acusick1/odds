"""Railway cron scheduler backend (static schedules)."""

from datetime import datetime

import structlog

from odds_lambda.scheduling.backends.base import (
    BackendHealth,
    RetryConfig,
    ScheduledJob,
    SchedulerBackend,
    ValidationResult,
)

logger = structlog.get_logger()


class RailwayBackend(SchedulerBackend):
    """
    Railway cron implementation.

    Railway uses static cron schedules configured via railway.json or UI.
    This backend is mainly for compatibility - scheduling operations are no-ops
    since Railway doesn't support dynamic schedule modification.

    Jobs will still execute self-gating logic to skip execution when appropriate,
    but they cannot dynamically adjust the cron schedule itself.

    Configuration:
    Configure static cron schedules in railway.json:
    {
      "cron": {
        "jobs": [
          {
            "name": "fetch-odds",
            "schedule": "*/30 * * * *",
            "command": "python jobs/fetch_odds.py"
          }
        ]
      }
    }

    Features:
    - Static cron schedules via Railway config
    - Smart gating logic in jobs prevents wasteful execution
    - Health checks verify configuration
    - Dry-run mode supported
    """

    def __init__(self, dry_run: bool = False, retry_config: RetryConfig | None = None):
        """
        Initialize Railway backend.

        Args:
            dry_run: If True, log operations without executing them
            retry_config: Not used for Railway but maintained for interface compatibility
        """
        super().__init__(dry_run=dry_run, retry_config=retry_config)

        logger.info("railway_backend_initialized", dry_run=self.dry_run)

    def validate_configuration(self) -> ValidationResult:
        """Validate Railway backend configuration."""
        from odds_core.scheduling.health_check import ValidationBuilder

        builder = ValidationBuilder()

        # Railway backend has minimal requirements
        # Configuration is managed externally via railway.json
        builder.add_warning(
            "Railway uses static cron schedules - configure in railway.json or Railway UI"
        )
        builder.add_warning(
            "Dynamic scheduling not supported - jobs use smart gating logic instead"
        )

        # Always valid - no runtime requirements
        return builder.build()

    async def health_check(self) -> BackendHealth:
        """Perform health check of Railway backend."""
        import os

        from odds_core.scheduling.health_check import HealthCheckBuilder

        builder = HealthCheckBuilder(self.get_backend_name())

        # Railway backend is always healthy since it has no external dependencies
        builder.pass_check("Railway backend operational")
        builder.pass_check("Static cron schedules managed externally")

        # Check if running in Railway environment
        railway_env = os.getenv("RAILWAY_ENVIRONMENT")
        builder.check_condition(
            railway_env is not None,
            "Running in Railway environment",
            "Not running in Railway environment",
        )

        if railway_env:
            builder.add_detail("railway_environment", railway_env)

        return builder.build()

    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        """
        Get list of scheduled jobs.

        Note: Railway doesn't provide API to query cron schedules programmatically.

        Raises:
            BackendUnavailableError: Always, as Railway doesn't support job listing
        """
        from odds_core.scheduling.exceptions import BackendUnavailableError

        raise BackendUnavailableError(
            "Railway backend does not support programmatic job listing. "
            "View schedules in Railway dashboard or railway.json"
        )

    async def get_job_status(self, job_name: str) -> ScheduledJob | None:
        """
        Get status of specific job.

        Note: Railway doesn't provide API to query job status.

        Raises:
            BackendUnavailableError: Always, as Railway doesn't support job status queries
        """
        from odds_core.scheduling.exceptions import BackendUnavailableError

        raise BackendUnavailableError(
            f"Railway backend does not support job status queries for '{job_name}'. "
            "View schedules in Railway dashboard or railway.json"
        )

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        No-op: Railway uses static cron schedules.

        Logs the intended next execution time for observability,
        but does not modify Railway's cron configuration.

        Args:
            job_name: Job identifier
            next_time: Intended next execution time (logged only)
        """
        if self.dry_run:
            logger.info(
                "dry_run_schedule",
                job=job_name,
                next_time=next_time.isoformat(),
                backend="railway_cron",
            )
            return

        logger.info(
            "railway_scheduling_noop",
            job=job_name,
            intended_next_time=next_time.isoformat(),
            note="Railway uses static cron - configure in railway.json",
        )

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        No-op: Railway cron cannot be dynamically cancelled.

        Args:
            job_name: Job identifier
        """
        if self.dry_run:
            logger.info("dry_run_cancel", job=job_name, backend="railway_cron")
            return

        logger.info(
            "railway_cancel_noop",
            job=job_name,
            note="Railway cron cannot be dynamically cancelled",
        )

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "railway_cron"
