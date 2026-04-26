"""Centralized job registry for scheduler backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from importlib import import_module

import structlog
from odds_core.sports import SportKey

from odds_lambda.betfair.constants import SPORT_CONFIG as BETFAIR_SPORT_CONFIG

logger = structlog.get_logger()


@dataclass
class JobContext:
    """Typed context passed to every job ``main()`` function.

    Scheduler backends construct this from the event payload. Jobs that
    require specific fields (e.g. ``sport``) assert them at the top of
    ``main()``. Adding new fields here is backward-compatible — all
    fields have defaults.
    """

    sport: SportKey | None = None

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
    "agent-run": ("odds_lambda.jobs.agent_run", "main"),
    "fetch-espn-fixtures": ("odds_lambda.jobs.fetch_espn_fixtures", "main"),
    "fetch-betfair-exchange": ("odds_lambda.jobs.fetch_betfair_exchange", "main"),
    "fetch-mlb-probables": ("odds_lambda.jobs.fetch_mlb_probables", "main"),
}

# Bootstrap entry-point overrides: jobs listed here use a different function
# for bootstrap (scheduler start) than for normal execution. The function
# receives a single ``sport: str`` argument for per-sport jobs, or no args
# for global jobs.  Jobs not listed here use their regular ``main(JobContext)``
# for bootstrap.
_JOB_BOOTSTRAP_MAP: dict[str, tuple[str, str]] = {
    "agent-run": ("odds_lambda.jobs.agent_run", "schedule_next"),
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
        "agent-run",
        "fetch-espn-fixtures",
        "fetch-betfair-exchange",
        "fetch-mlb-probables",
    }
)

# Jobs with fixed cron schedules when run locally. Mirrors terraform's
# ``fixed_schedule_expressions`` for the jobs that have moved out of
# Lambda and into the local scheduler. Keys are base job names (no sport
# suffix); the scheduler CLI expands per-sport jobs into one schedule per
# allowed sport, each with its own compound job name.
#
# Values are ``(cron_expression, allowed_sports_or_None)``. When the second
# element is ``None`` the job runs for every sport in
# ``data_collection.sports``; otherwise it runs only for the listed sport
# keys. Use an allowlist for jobs whose ``main()`` raises on unsupported
# sports (e.g. ``fetch-espn-fixtures`` is EPL-only) — otherwise the daily
# cron fire on an unsupported sport would raise inside ``job_alert_context``
# and spam critical Discord alerts.
_JOB_CRON_MAP: dict[str, tuple[str, tuple[SportKey, ...] | None]] = {
    # daily-digest is gated to EPL because the digest formatter and
    # heartbeat_expectations entry are EPL-only today.
    "daily-digest": ("0 8 * * *", ("soccer_epl",)),
    "fetch-espn-fixtures": ("0 6 * * *", ("soccer_epl",)),
    # Betfair Exchange direct ingestion. The job self-schedules a tighter
    # cadence near kickoff; this cron is the safety floor that catches
    # cold-start gaps and backfills if self-scheduling stalls. Allowlist
    # is derived from the BFE adapter's supported-sports map so adding a
    # new sport requires only a SPORT_CONFIG entry.
    "fetch-betfair-exchange": ("*/30 * * * *", tuple(BETFAIR_SPORT_CONFIG)),
    # MLB probable pitchers — once daily at 06:00 UTC. Backstop only:
    # the MCP ``get_probable_pitchers`` tool writes through every call.
    "fetch-mlb-probables": ("0 6 * * *", ("baseball_mlb",)),
}


def get_job_cron(job_name: str) -> str | None:
    """Return the cron expression for a job, or None if it is event-driven."""
    entry = _JOB_CRON_MAP.get(job_name)
    if entry is None:
        return None
    cron_expr, _ = entry
    return cron_expr


def get_job_cron_sports(job_name: str) -> tuple[SportKey, ...] | None:
    """Return the allowlist of sports for a cron job, or ``None`` for all sports.

    Returns:
        A tuple of ``SportKey`` values when the job's cron schedule is
        restricted, ``None`` when it runs for every configured sport, and
        ``None`` for event-driven (non-cron) jobs.
    """
    entry = _JOB_CRON_MAP.get(job_name)
    if entry is None:
        return None
    _, sports = entry
    return sports


assert set(_JOB_CRON_MAP) <= set(_JOB_MODULE_MAP), (
    f"Unknown jobs in _JOB_CRON_MAP: {set(_JOB_CRON_MAP) - set(_JOB_MODULE_MAP)}"
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


def is_per_sport_job(job_name: str) -> bool:
    """Check whether a job accepts a sport parameter."""
    return job_name in _PER_SPORT_JOBS


def get_bootstrap_function(job_name: str) -> Callable[[JobContext], Awaitable[None]]:
    """Get the bootstrap entry point for a job.

    Returns a callable with the same ``(JobContext) -> Awaitable[None]``
    signature as regular job functions. Jobs with a custom bootstrap
    (e.g. ``agent-run`` uses ``schedule_next(sport)`` instead of ``main``)
    are wrapped so the caller doesn't need to know about the difference.

    Raises:
        KeyError: If job name not found in registry
    """
    if job_name in _JOB_BOOTSTRAP_MAP:
        module_path, func_name = _JOB_BOOTSTRAP_MAP[job_name]
        module = import_module(module_path)
        raw_fn = getattr(module, func_name)

        # Wrap custom bootstrap functions to accept JobContext uniformly.
        # Per-sport custom bootstraps expect (sport: str); global ones expect ().
        if job_name in _PER_SPORT_JOBS:

            async def _per_sport_wrapper(ctx: JobContext) -> None:
                assert ctx.sport, f"Job '{job_name}' requires a sport in JobContext"
                await raw_fn(ctx.sport)

            return _per_sport_wrapper
        else:

            async def _global_wrapper(ctx: JobContext) -> None:
                await raw_fn()

            return _global_wrapper

    return get_job_function(job_name)


def list_available_jobs() -> list[str]:
    """List all registered job names."""
    return sorted(_JOB_MODULE_MAP)
