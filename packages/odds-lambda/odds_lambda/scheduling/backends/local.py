"""Local APScheduler backend for testing."""

import asyncio
from datetime import datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from odds_lambda.scheduling.backends.base import (
    BackendHealth,
    JobStatus,
    RetryConfig,
    ScheduledJob,
    SchedulerBackend,
    ValidationResult,
)
from odds_lambda.scheduling.exceptions import (
    BackendUnavailableError,
    CancellationFailedError,
    JobNotFoundError,
    SchedulingFailedError,
)

logger = structlog.get_logger()


class LocalSchedulerBackend(SchedulerBackend):
    """
    Local testing implementation using APScheduler.

    Simulates AWS EventBridge behavior but runs everything locally.
    Jobs self-schedule their next execution just like in AWS.

    Features:
    - Simulates dynamic scheduling locally
    - Full testing without AWS account
    - Identical job behavior to production
    - APScheduler date triggers for one-time execution
    - Context manager support for clean lifecycle management
    - Health checks and status queries
    - Dry-run mode for testing

    Usage:
    Set SCHEDULER_BACKEND=local in .env and run:
        python -m cli.main scheduler start

    Or use as async context manager:
        async with LocalSchedulerBackend() as backend:
            # Bootstrap jobs
            await some_job.main()
            # Keep running
            await asyncio.sleep(float('inf'))
    """

    def __init__(self, dry_run: bool = False, retry_config: RetryConfig | None = None):
        """
        Initialize APScheduler.

        Args:
            dry_run: If True, log operations without executing them
            retry_config: Retry configuration (uses defaults if None)
        """
        super().__init__(dry_run=dry_run, retry_config=retry_config)

        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._started = False

        logger.info("local_scheduler_initialized", dry_run=self.dry_run)

    def validate_configuration(self) -> ValidationResult:
        """Validate local backend configuration."""
        from odds_core.scheduling.health_check import ValidationBuilder

        builder = ValidationBuilder()

        # Check APScheduler availability
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: F401
        except ImportError:
            builder.add_error("APScheduler not installed (pip install apscheduler)")

        # Check that job registry can be loaded
        try:
            from odds_core.scheduling.jobs import get_job_registry

            registry = get_job_registry()
            if not registry:
                builder.add_warning("Job registry is empty")
        except ImportError as e:
            builder.add_warning(f"Job registry import warning: {e}")

        return builder.build()

    async def health_check(self) -> BackendHealth:
        """
        Perform comprehensive health check of local backend.

        Note: Backend is considered healthy if configuration is valid,
        regardless of whether scheduler is started. "Not started" is a
        valid state, not a failure.
        """
        from odds_core.scheduling.health_check import HealthCheckBuilder

        builder = HealthCheckBuilder(self.get_backend_name())

        # Check configuration
        validation = self.validate_configuration()
        builder.check_condition(
            validation.is_valid,
            "Configuration valid",
            f"Configuration invalid: {', '.join(validation.errors)}",
        )

        # Check scheduler state (informational, not a failure)
        if self._started:
            builder.pass_check("Scheduler running")
            builder.add_detail("scheduler_state", "running")

            # Check number of scheduled jobs
            jobs = self.scheduler.get_jobs()
            builder.add_detail("scheduled_jobs_count", len(jobs))
            builder.pass_check(f"{len(jobs)} jobs scheduled")
        else:
            # Not started is OK - just report the state
            builder.pass_check("Scheduler initialized (not started)")
            builder.add_detail("scheduler_state", "stopped")
            builder.add_detail("note", "Call start() or use as context manager to begin scheduling")

        # Check event loop availability (informational)
        try:
            asyncio.get_running_loop()
            builder.pass_check("Event loop available")
            builder.add_detail("event_loop_running", True)
        except RuntimeError:
            # No event loop is OK if scheduler isn't started
            builder.add_detail("event_loop_running", False)
            if not self._started:
                builder.pass_check("No event loop (expected when stopped)")
            else:
                # This would be weird - scheduler running but no loop
                builder.fail_check("Scheduler started but no event loop detected")

        return builder.build()

    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        """Get list of all currently scheduled jobs."""
        if self.dry_run:
            logger.info("dry_run_get_scheduled_jobs")
            return []

        if not self._started:
            logger.warning("local_scheduler_not_started")
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            # Extract job name from APScheduler job ID
            job_name = job.id

            # Get next run time
            next_run = job.next_run_time

            jobs.append(
                ScheduledJob(
                    job_name=job_name,
                    next_run_time=next_run,
                    status=JobStatus.SCHEDULED,
                )
            )

        return jobs

    async def get_job_status(self, job_name: str) -> ScheduledJob | None:
        """Get status of a specific job."""
        if self.dry_run:
            logger.info("dry_run_get_job_status", job=job_name)
            return None

        if not self._started:
            logger.warning("local_scheduler_not_started")
            return None

        job = self.scheduler.get_job(job_name)
        if not job:
            return None

        return ScheduledJob(
            job_name=job_name,
            next_run_time=job.next_run_time,
            status=JobStatus.SCHEDULED,
        )

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Schedule job using APScheduler date trigger.

        Args:
            job_name: Job identifier (e.g., 'fetch-odds')
            next_time: UTC datetime for next execution

        Raises:
            SchedulingFailedError: If scheduling fails
        """
        if self.dry_run:
            logger.info(
                "dry_run_schedule",
                job=job_name,
                next_time=next_time.isoformat(),
                backend="local_apscheduler",
            )
            return

        self._ensure_started()

        try:
            # Get job function from centralized registry
            from odds_core.scheduling.jobs import get_job_function

            job_func = get_job_function(job_name)

            # Remove existing job if present (replace with new schedule)
            if self.scheduler.get_job(job_name):
                self.scheduler.remove_job(job_name)
                logger.debug("local_job_removed", job=job_name)

            # Add new scheduled job with date trigger (one-time execution)
            self.scheduler.add_job(
                func=job_func,
                trigger="date",
                run_date=next_time,
                id=job_name,
                name=f"Odds {job_name}",
                replace_existing=True,
            )

            logger.info(
                "local_job_scheduled",
                job=job_name,
                next_time=next_time.isoformat(),
            )

        except KeyError as e:
            # Job not found in registry
            raise SchedulingFailedError(str(e)) from e
        except Exception as e:
            logger.error(
                "local_scheduling_failed",
                job=job_name,
                next_time=next_time.isoformat(),
                error=str(e),
                exc_info=True,
            )
            raise SchedulingFailedError(f"Failed to schedule {job_name}: {e}") from e

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Cancel APScheduler job.

        Args:
            job_name: Job identifier to cancel

        Raises:
            JobNotFoundError: If job doesn't exist
            CancellationFailedError: If cancellation fails
        """
        if self.dry_run:
            logger.info("dry_run_cancel", job=job_name, backend="local_apscheduler")
            return

        try:
            if self.scheduler.get_job(job_name):
                self.scheduler.remove_job(job_name)
                logger.info("local_job_cancelled", job=job_name)
            else:
                raise JobNotFoundError(f"Job {job_name} not found")

        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error(
                "local_cancel_failed",
                job=job_name,
                error=str(e),
                exc_info=True,
            )
            raise CancellationFailedError(f"Failed to cancel {job_name}: {e}") from e

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "local_apscheduler"

    def _ensure_started(self):
        """Start scheduler if not already running."""
        if not self._started:
            try:
                # Try to get running loop (will fail if not in async context)
                asyncio.get_running_loop()
                self.scheduler.start()
                self._started = True
                logger.info("local_scheduler_started")
            except RuntimeError as e:
                # No event loop running
                raise BackendUnavailableError(
                    "Cannot start scheduler: No event loop running. "
                    "Use LocalSchedulerBackend as async context manager or ensure running in async context."
                ) from e

    async def __aenter__(self):
        """Start scheduler when entering context."""
        self._ensure_started()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Shutdown scheduler when exiting context."""
        if self._started:
            self.scheduler.shutdown()
            self._started = False
            logger.info("local_scheduler_shutdown")
        return False  # Don't suppress exceptions
