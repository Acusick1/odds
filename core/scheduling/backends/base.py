"""Abstract base class for scheduler backend implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Status of a scheduled job."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class ScheduledJob:
    """Information about a scheduled job."""

    job_name: str
    next_run_time: datetime | None
    status: JobStatus = JobStatus.UNKNOWN
    error_message: str | None = None


@dataclass
class BackendHealth:
    """Health status of scheduler backend."""

    is_healthy: bool
    backend_name: str
    checks_passed: list[str]
    checks_failed: list[str]
    details: dict | None = None


@dataclass
class RetryConfig:
    """Retry configuration for scheduling operations."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class SchedulerBackend(ABC):
    """
    Abstract interface for scheduling backends.

    All scheduler implementations (AWS EventBridge, Railway cron, local APScheduler)
    must implement this interface to provide a consistent API for job scheduling.

    Features:
    - Dynamic job scheduling and cancellation
    - Health monitoring and status queries
    - Configuration validation
    - Dry-run mode for testing
    - Retry configuration for resilience
    """

    def __init__(self, dry_run: bool = False, retry_config: RetryConfig | None = None):
        """
        Initialize scheduler backend.

        Args:
            dry_run: If True, log operations without executing them
            retry_config: Configuration for retry behavior (uses defaults if None)
        """
        self.dry_run = dry_run
        self.retry_config = retry_config or RetryConfig()

    @abstractmethod
    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Schedule next execution of a job at a specific time.

        Args:
            job_name: Identifier for the job (e.g., 'fetch-odds', 'fetch-scores')
            next_time: UTC datetime when job should execute next

        Raises:
            SchedulingFailedError: If scheduling fails after retries
            BackendUnavailableError: If backend service is unavailable
        """
        ...

    @abstractmethod
    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Cancel upcoming scheduled execution of a job.

        Args:
            job_name: Identifier for the job to cancel

        Raises:
            CancellationFailedError: If cancellation fails
            JobNotFoundError: If job doesn't exist
        """
        ...

    @abstractmethod
    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        """
        Get list of all currently scheduled jobs.

        Returns:
            List of scheduled job information

        Raises:
            BackendUnavailableError: If backend service is unavailable
        """
        ...

    @abstractmethod
    async def get_job_status(self, job_name: str) -> ScheduledJob | None:
        """
        Get status of a specific job.

        Args:
            job_name: Job identifier to query

        Returns:
            Job information if found, None otherwise

        Raises:
            BackendUnavailableError: If backend service is unavailable
        """
        ...

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """
        Perform comprehensive health check of backend.

        Checks:
        - Backend service availability
        - Configuration validity
        - Authentication/permissions
        - Required resources accessible

        Returns:
            Health status with details of checks performed
        """
        ...

    @abstractmethod
    def validate_configuration(self) -> ValidationResult:
        """
        Validate backend configuration.

        Checks:
        - Required environment variables present
        - Configuration values valid
        - Credentials/permissions available

        Returns:
            Validation result with errors and warnings
        """
        ...

    @abstractmethod
    def get_backend_name(self) -> str:
        """
        Return backend identifier for logging.

        Returns:
            Backend name string (e.g., 'aws_eventbridge', 'local_apscheduler')
        """
        ...
