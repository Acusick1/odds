"""Abstract base class for scheduler backend implementations."""

from abc import ABC, abstractmethod
from datetime import datetime


class SchedulerBackend(ABC):
    """
    Abstract interface for scheduling backends.

    All scheduler implementations (AWS EventBridge, Railway cron, local APScheduler)
    must implement this interface to provide a consistent API for job scheduling.
    """

    @abstractmethod
    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Schedule next execution of a job at a specific time.

        Args:
            job_name: Identifier for the job (e.g., 'fetch-odds', 'fetch-scores')
            next_time: UTC datetime when job should execute next

        Raises:
            Exception: If scheduling fails
        """
        pass

    @abstractmethod
    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Cancel upcoming scheduled execution of a job.

        Args:
            job_name: Identifier for the job to cancel

        Raises:
            Exception: If cancellation fails
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Return backend identifier for logging.

        Returns:
            Backend name string (e.g., 'aws_eventbridge', 'local_apscheduler')
        """
        pass
