"""Scheduler backend implementations."""

from core.scheduling.backends.base import SchedulerBackend


def get_scheduler_backend() -> SchedulerBackend:
    """
    Create scheduler backend based on SCHEDULER_BACKEND env var.

    Supported backends:
    - 'aws': AWS Lambda + EventBridge (production)
    - 'railway': Railway cron (alternative production)
    - 'local': APScheduler (local testing)

    Returns:
        Configured scheduler backend instance

    Raises:
        ValueError: If unknown backend specified
    """
    from core.config import settings

    backend = settings.scheduler_backend.lower()

    if backend == "aws":
        from core.scheduling.backends.aws import AWSEventBridgeBackend

        return AWSEventBridgeBackend()

    elif backend == "railway":
        from core.scheduling.backends.railway import RailwayBackend

        return RailwayBackend()

    elif backend == "local":
        from core.scheduling.backends.local import LocalSchedulerBackend

        return LocalSchedulerBackend()

    else:
        raise ValueError(
            f"Unknown scheduler backend: {backend}. " f"Supported: 'aws', 'railway', 'local'"
        )


__all__ = ["SchedulerBackend", "get_scheduler_backend"]
