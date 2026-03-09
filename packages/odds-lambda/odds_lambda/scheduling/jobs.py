"""Centralized job registry for scheduler backends."""

from collections.abc import Awaitable, Callable
from importlib import import_module
from typing import Any

# Maps job name to (module_path, function_name).
# Modules are imported lazily when the job is first requested, so a Lambda
# that only runs scraper jobs never imports score_predictions (which needs
# odds_analytics / xgboost).
_JOB_MODULE_MAP: dict[str, tuple[str, str]] = {
    "fetch-odds": ("odds_lambda.jobs.fetch_odds", "main"),
    "fetch-scores": ("odds_lambda.jobs.fetch_scores", "main"),
    "update-status": ("odds_lambda.jobs.update_status", "main"),
    "check-health": ("odds_lambda.jobs.check_health", "main"),
    "fetch-polymarket": ("odds_lambda.jobs.fetch_polymarket", "main"),
    "backfill-polymarket": ("odds_lambda.jobs.backfill_polymarket", "main"),
    "fetch-oddsportal": ("odds_lambda.jobs.fetch_oddsportal", "main"),
    "fetch-oddsportal-results": ("odds_lambda.jobs.fetch_oddsportal_results", "main"),
    "score-predictions": ("odds_lambda.jobs.score_predictions", "main"),
    "daily-digest": ("odds_lambda.jobs.daily_digest", "main"),
}

# Cache of already-imported job functions.
_loaded_jobs: dict[str, Callable[..., Awaitable[Any]]] = {}


def get_job_function(job_name: str) -> Callable[..., Awaitable[Any]]:
    """Get job function by name, importing its module on first access.

    Args:
        job_name: Job identifier (e.g., 'fetch-odds')

    Returns:
        Async job function

    Raises:
        KeyError: If job name not found in registry
    """
    if job_name in _loaded_jobs:
        return _loaded_jobs[job_name]

    if job_name not in _JOB_MODULE_MAP:
        available = ", ".join(sorted(_JOB_MODULE_MAP))
        raise KeyError(f"Unknown job '{job_name}'. Available jobs: {available}")

    module_path, func_name = _JOB_MODULE_MAP[job_name]
    module = import_module(module_path)
    fn = getattr(module, func_name)
    _loaded_jobs[job_name] = fn
    return fn


def list_available_jobs() -> list[str]:
    """List all registered job names."""
    return sorted(_JOB_MODULE_MAP)
