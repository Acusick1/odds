"""Centralized job registry for scheduler backends."""

from collections.abc import Awaitable, Callable

# Lazy-loaded registry to avoid circular imports
_JOB_REGISTRY: dict[str, Callable[[], Awaitable[None]]] | None = None


def get_job_registry() -> dict[str, Callable[[], Awaitable[None]]]:
    """
    Get mapping of job names to job functions.

    Registry is lazy-loaded on first access to avoid importing
    job modules during backend initialization.

    Returns:
        Dictionary mapping job names (e.g., 'fetch-odds') to async job functions
    """
    global _JOB_REGISTRY

    if _JOB_REGISTRY is None:
        # Import job modules only when needed
        from odds_lambda.jobs import fetch_odds, fetch_scores, update_status

        _JOB_REGISTRY = {
            "fetch-odds": fetch_odds.main,
            "fetch-scores": fetch_scores.main,
            "update-status": update_status.main,
        }

    return _JOB_REGISTRY


def get_job_function(job_name: str) -> Callable[[], Awaitable[None]]:
    """
    Get job function by name.

    Args:
        job_name: Job identifier (e.g., 'fetch-odds')

    Returns:
        Async job function

    Raises:
        KeyError: If job name not found in registry
    """
    registry = get_job_registry()
    if job_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown job '{job_name}'. Available jobs: {available}")
    return registry[job_name]


def list_available_jobs() -> list[str]:
    """
    List all registered job names.

    Returns:
        Sorted list of job names
    """
    return sorted(get_job_registry().keys())
