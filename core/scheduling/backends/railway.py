"""Railway cron scheduler backend (static schedules)."""

from datetime import datetime

import structlog

from core.scheduling.backends.base import SchedulerBackend

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
    """

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        No-op: Railway uses static cron schedules.

        Logs the intended next execution time for observability,
        but does not modify Railway's cron configuration.

        Args:
            job_name: Job identifier
            next_time: Intended next execution time (logged only)
        """
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
        logger.info(
            "railway_cancel_noop",
            job=job_name,
            note="Railway cron cannot be dynamically cancelled",
        )

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "railway_cron"
