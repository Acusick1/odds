"""Scheduler backend implementations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

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
    ConfigurationError,
    JobNotFoundError,
    SchedulerBackendError,
    SchedulingFailedError,
)

# ---------------------------------------------------------------------------
# Self-scheduling policy (the ``self_schedule`` seam)
# ---------------------------------------------------------------------------
#
# Process-local switch that forces every backend constructed via
# ``get_scheduler_backend`` into dry-run, so no rows are written to the
# schedule store regardless of the ``dry_run`` a caller passes. Default is
# ``True`` (scheduling permitted). The scheduler ``smoke`` command flips it off
# so a validation run exercises full job bodies — including their self-schedule
# call — while leaving the live schedule store untouched. Enforcing it at the
# single backend-construction chokepoint means no per-job knowledge of which
# jobs self-schedule is needed. Blast radius stays on the local scheduler: the
# Lambda/EventBridge dispatch path never enters this context.
_scheduling_enabled: ContextVar[bool] = ContextVar("odds_scheduling_enabled", default=True)


def scheduling_enabled() -> bool:
    """Whether self-scheduling (writes to the schedule store) is permitted."""
    return _scheduling_enabled.get()


@contextmanager
def suppress_scheduling(suppressed: bool = True) -> Iterator[None]:
    """Scope self-scheduling off for the duration of the ``with`` block.

    Used by ``scheduler smoke`` so job bodies run without persisting any
    schedule. Restores the previous value on exit.
    """
    token = _scheduling_enabled.set(not suppressed)
    try:
        yield
    finally:
        _scheduling_enabled.reset(token)


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
                     If None, reads from settings.scheduler.backend
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
    from odds_core.config import get_settings

    settings = get_settings()
    backend = backend_type or settings.scheduler.backend.lower()

    # A ``suppress_scheduling`` scope (set by ``scheduler smoke``) forces dry-run
    # at this single chokepoint so no job can write to the schedule store.
    dry_run = dry_run or not scheduling_enabled()

    if backend == "aws":
        from odds_lambda.scheduling.backends.aws import AWSEventBridgeBackend

        return AWSEventBridgeBackend(
            dry_run=dry_run,
            retry_config=retry_config,
            aws_region=kwargs.get("aws_region") or settings.aws.region,
            lambda_arn=kwargs.get("lambda_arn") or settings.aws.lambda_arn,
            rule_prefix=kwargs.get("rule_prefix") or settings.aws.rule_prefix,
        )

    elif backend == "railway":
        from odds_lambda.scheduling.backends.railway import RailwayBackend

        return RailwayBackend(
            dry_run=dry_run,
            retry_config=retry_config,
        )

    elif backend == "local":
        from odds_lambda.scheduling.backends.local import LocalSchedulerBackend

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
    # Self-scheduling policy seam
    "scheduling_enabled",
    "suppress_scheduling",
]
