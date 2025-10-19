"""Local APScheduler backend for testing."""

import asyncio
from datetime import datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.scheduling.backends.base import SchedulerBackend

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

    Usage:
    Set SCHEDULER_BACKEND=local in .env and run:
        python -m cli.main scheduler start

    This will start a local scheduler that keeps running and executes
    jobs at their scheduled times, just like AWS Lambda.
    """

    def __init__(self):
        """Initialize APScheduler."""
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._started = False
        logger.info("local_scheduler_initialized")

    def _ensure_started(self):
        """Start scheduler if not already running."""
        if not self._started:
            try:
                # Try to get running loop (will fail if not in async context)
                asyncio.get_running_loop()
                self.scheduler.start()
                self._started = True
            except RuntimeError:
                # No event loop running - scheduler will start when keep_alive() is called
                pass

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Schedule job using APScheduler date trigger.

        Args:
            job_name: Job identifier (e.g., 'fetch-odds')
            next_time: UTC datetime for next execution

        Raises:
            Exception: If scheduling fails
        """
        self._ensure_started()

        try:
            # Import job modules
            from jobs import fetch_odds, fetch_scores, update_status

            job_map = {
                "fetch-odds": fetch_odds.main,
                "fetch-scores": fetch_scores.main,
                "update-status": update_status.main,
            }

            if job_name not in job_map:
                raise ValueError(f"Unknown job: {job_name}")

            # Remove existing job if present (replace with new schedule)
            if self.scheduler.get_job(job_name):
                self.scheduler.remove_job(job_name)
                logger.debug("local_job_removed", job=job_name)

            # Add new scheduled job with date trigger (one-time execution)
            self.scheduler.add_job(
                func=job_map[job_name],
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

        except Exception as e:
            logger.error(
                "local_scheduling_failed",
                job=job_name,
                next_time=next_time.isoformat(),
                error=str(e),
                exc_info=True,
            )
            raise

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Cancel APScheduler job.

        Args:
            job_name: Job identifier to cancel
        """
        try:
            if self.scheduler.get_job(job_name):
                self.scheduler.remove_job(job_name)
                logger.info("local_job_cancelled", job=job_name)
            else:
                logger.warning("local_job_not_found", job=job_name)

        except Exception as e:
            logger.error(
                "local_cancel_failed",
                job=job_name,
                error=str(e),
                exc_info=True,
            )
            raise

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "local_apscheduler"

    def keep_alive(self):
        """
        Block to keep scheduler running (for local testing).

        Call this from CLI to start the scheduler and keep it running
        until interrupted with Ctrl+C.
        """
        # Create new event loop if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Ensure scheduler is started within the event loop context
        if not self._started:
            # Set the event loop for APScheduler before starting
            self.scheduler._eventloop = loop
            self.scheduler.start()
            self._started = True

        try:
            logger.info("local_scheduler_running", message="Press Ctrl+C to stop")
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("local_scheduler_shutdown")
            self.scheduler.shutdown()
            loop.close()
