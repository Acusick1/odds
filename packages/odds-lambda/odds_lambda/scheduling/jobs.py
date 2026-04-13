"""Centralized job registry for scheduler backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from importlib import import_module

import structlog

logger = structlog.get_logger()


@dataclass
class JobContext:
    """Typed context passed to every job ``main()`` function.

    Scheduler backends construct this from the event payload. Jobs that
    require specific fields (e.g. ``sport``) assert them at the top of
    ``main()``. Adding new fields here is backward-compatible — all
    fields have defaults.
    """

    sport: str | None = None
    retry_count: int = 0

    # backfill-polymarket manual invocation params
    include_spreads: bool = False
    include_totals: bool = False
    dry_run: bool = False

    # daily-digest configuration
    lookback_hours: float = 24
    lookahead_hours: float = 48

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> JobContext:
        """Construct a ``JobContext`` from a raw event/scheduler payload.

        Known keys are mapped to dataclass fields; unknown keys are
        logged and ignored.
        """
        known_fields = {f.name for f in fields(cls)}
        known: dict[str, object] = {}
        for k, v in payload.items():
            if k in known_fields:
                known[k] = v
            else:
                logger.debug("job_context_unknown_key", key=k, value=v)
        return cls(**known)  # type: ignore[arg-type]


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

# Maps sport suffix to sport_key for per-sport job routing.
# e.g. "fetch-odds-epl" resolves to ("fetch-odds", "soccer_epl")
_SPORT_SUFFIX_MAP: dict[str, str] = {
    "epl": "soccer_epl",
    "mlb": "baseball_mlb",
}

# Jobs that accept a sport parameter. When invoked with a sport suffix
# (e.g. "fetch-odds-epl"), the sport_key is extracted and passed as a kwarg.
_PER_SPORT_JOBS: frozenset[str] = frozenset(
    {
        "fetch-odds",
        "fetch-scores",
        "fetch-oddsportal",
        "fetch-oddsportal-results",
        "score-predictions",
        "daily-digest",
    }
)

# Cache of already-imported job functions.
_loaded_jobs: dict[str, Callable[[JobContext], Awaitable[None]]] = {}


def sport_key_to_suffix(sport_key: str) -> str | None:
    """Map a sport key to its scheduling suffix.

    Reverse of ``_SPORT_SUFFIX_MAP``: e.g. ``"soccer_epl"`` -> ``"epl"``.

    Returns:
        The suffix string, or None if the sport key has no mapping.
    """
    for suffix, key in _SPORT_SUFFIX_MAP.items():
        if key == sport_key:
            return suffix
    return None


def make_compound_job_name(base_job_name: str, sport: str | None) -> str:
    """Build the compound job name used for self-scheduling.

    When a sport is provided and the job supports per-sport routing,
    appends the sport suffix so the EventBridge rule name matches
    Terraform (e.g. ``"fetch-odds"`` + ``"soccer_epl"`` -> ``"fetch-odds-epl"``).
    """
    if sport and base_job_name in _PER_SPORT_JOBS:
        suffix = sport_key_to_suffix(sport)
        if suffix:
            return f"{base_job_name}-{suffix}"
    return base_job_name


def resolve_job_name(compound_name: str) -> tuple[str, str | None]:
    """Resolve a potentially sport-suffixed job name.

    Args:
        compound_name: Job name, optionally with sport suffix (e.g. "fetch-odds-epl")

    Returns:
        (base_job_name, sport_key or None)
    """
    # Try direct match first (global jobs and bare per-sport jobs)
    if compound_name in _JOB_MODULE_MAP:
        return compound_name, None

    # Try stripping sport suffix: "fetch-odds-epl" -> ("fetch-odds", "epl")
    for suffix, sport_key in _SPORT_SUFFIX_MAP.items():
        if compound_name.endswith(f"-{suffix}"):
            base_name = compound_name[: -(len(suffix) + 1)]
            if base_name in _JOB_MODULE_MAP and base_name in _PER_SPORT_JOBS:
                return base_name, sport_key

    return compound_name, None


def get_job_function(job_name: str) -> Callable[[JobContext], Awaitable[None]]:
    """Get job function by name, importing its module on first access.

    Uses the base job name (without sport suffix) for module lookup.

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
