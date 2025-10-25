"""Scheduler backend implementations."""

from core.scheduling.backends.base import (
    BackendHealth,
    JobStatus,
    RetryConfig,
    ScheduledJob,
    SchedulerBackend,
    ValidationResult,
)
from core.scheduling.exceptions import (
    BackendUnavailableError,
    CancellationFailedError,
    ConfigurationError,
    JobNotFoundError,
    SchedulerBackendError,
    SchedulingFailedError,
)


def get_scheduler_backend(
    backend_type: str | None = None,
    dry_run: bool = False,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> SchedulerBackend:
    """
    Create scheduler backend based on configuration.

    Args:
        backend_type: Backend type ('aws', 'railway', 'local').
                     If None, reads from settings.scheduler_backend
        dry_run: Enable dry-run mode (log operations without executing)
        retry_config: Custom retry configuration (uses defaults if None)
        **kwargs: Backend-specific configuration options

    Returns:
        Configured scheduler backend instance

    Raises:
        ConfigurationError: If unknown backend specified or configuration invalid

    Supported backends:
    - 'aws': AWS Lambda + EventBridge (production)
    - 'railway': Railway cron (alternative production)
    - 'local': APScheduler (local testing)

    Examples:
        # Use configured backend from settings
        backend = get_scheduler_backend()

        # Explicit backend with dry-run
        backend = get_scheduler_backend(backend_type='aws', dry_run=True)

        # Custom retry configuration
        retry_config = RetryConfig(max_attempts=5, initial_delay_seconds=2.0)
        backend = get_scheduler_backend(retry_config=retry_config)

        # AWS backend with custom region
        backend = get_scheduler_backend(
            backend_type='aws',
            aws_region='us-west-2',
            lambda_arn='arn:aws:lambda:...'
        )
    """
    # Import settings here to avoid circular imports
    from core.config import settings

    backend = backend_type or settings.scheduler_backend.lower()

    if backend == "aws":
        from core.scheduling.backends.aws import AWSEventBridgeBackend

        return AWSEventBridgeBackend(
            dry_run=dry_run,
            retry_config=retry_config,
            aws_region=kwargs.get("aws_region"),
            lambda_arn=kwargs.get("lambda_arn"),
        )

    elif backend == "railway":
        from core.scheduling.backends.railway import RailwayBackend

        return RailwayBackend(
            dry_run=dry_run,
            retry_config=retry_config,
        )

    elif backend == "local":
        from core.scheduling.backends.local import LocalSchedulerBackend

        return LocalSchedulerBackend(
            dry_run=dry_run,
            retry_config=retry_config,
        )

    else:
        raise ConfigurationError(
            f"Unknown scheduler backend: {backend}. Supported: 'aws', 'railway', 'local'"
        )


__all__ = [
    # Base classes and data models
    "SchedulerBackend",
    "ScheduledJob",
    "JobStatus",
    "BackendHealth",
    "ValidationResult",
    "RetryConfig",
    # Exceptions
    "SchedulerBackendError",
    "SchedulingFailedError",
    "CancellationFailedError",
    "BackendUnavailableError",
    "ConfigurationError",
    "JobNotFoundError",
    # Factory function
    "get_scheduler_backend",
]
