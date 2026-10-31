"""Centralized job registry for scheduler backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from importlib import import_module

import structlog
from odds_core.sports import SportKey

from odds_lambda.betfair.constants import SPORT_CONFIG as BETFAIR_SPORT_CONFIG
from odds_lambda.polymarket_sports import MATCH_ODDS_SPORTS

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
# Modules are imported lazily when the job is first requested.
_JOB_MODULE_MAP: dict[str, tuple[str, str]] = {
    "fetch-odds": ("odds_lambda.jobs.fetch_odds", "main"),
    "fetch-scores": ("odds_lambda.jobs.fetch_scores", "main"),
    "update-status": ("odds_lambda.jobs.update_status", "main"),
    "check-health": ("odds_lambda.jobs.check_health", "main"),
    "fetch-oddsportal": ("odds_lambda.jobs.fetch_oddsportal", "main"),
    "fetch-oddsportal-results": ("odds_lambda.jobs.fetch_oddsportal_results", "main"),
    "fetch-mlb-results": ("odds_lambda.jobs.fetch_mlb_results", "main"),
    "agent-run": ("odds_lambda.jobs.agent_run", "main"),
    "fetch-espn-fixtures": ("odds_lambda.jobs.fetch_espn_fixtures", "main"),
    "fetch-betfair-exchange": ("odds_lambda.jobs.fetch_betfair_exchange", "main"),
    "fetch-polymarket-odds": ("odds_lambda.jobs.fetch_polymarket_odds", "main"),
    "fetch-mlb-probables": ("odds_lambda.jobs.fetch_mlb_probables", "main"),
    "run-candidates": ("odds_lambda.jobs.run_candidates", "main"),
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
_SPORT_SUFFIX_MAP: dict[str, SportKey] = {
    "epl": "soccer_epl",
    "mlb": "baseball_mlb",
    "nba": "basketball_nba",
}

# Jobs that must never run during a deploy smoke check, even when listed in
# ``bootstrap_jobs``. ``agent-run`` places paper bets, so executing it as part of
# a pre-restart validation would pollute the live portfolio. The smoke command
# skips it by default and only runs it under an explicit opt-in.
_SMOKE_UNSAFE_JOBS: frozenset[str] = frozenset({"agent-run"})

# ``run-candidates`` also places paper bets, so smoke skips it too, under its own opt-in.
_CANDIDATE_JOBS: frozenset[str] = frozenset({"run-candidates"})

# Jobs whose runs spend Odds API quota units. The smoke command skips these by
# default because every deploy would otherwise burn units off-schedule.
_PAID_API_JOBS: frozenset[str] = frozenset({"fetch-odds", "fetch-scores"})

# Jobs that accept a sport parameter. When invoked with a sport suffix
# (e.g. "fetch-odds-epl"), the sport_key is extracted and passed as a kwarg.
_PER_SPORT_JOBS: frozenset[str] = frozenset(
    {
        "fetch-odds",
        "fetch-scores",
        "fetch-oddsportal",
        "fetch-oddsportal-results",
        "agent-run",
        "fetch-espn-fixtures",
        "fetch-betfair-exchange",
        "fetch-polymarket-odds",
        "fetch-mlb-probables",
        "run-candidates",
    }
)

# Jobs with fixed cron schedules in the local scheduler. Keys are base job
# names (no sport suffix); the scheduler CLI expands per-sport jobs into one
# schedule per target sport, each with its own compound job name. A job that
# self-schedules must not be listed: bootstrap re-applies every cron, which
# would replace the job's pending run under the shared schedule id.
_JOB_CRON_MAP: dict[str, str] = {
    "fetch-espn-fixtures": "0 6 * * *",
    # MLB probable pitchers — once daily at 06:00 UTC. Backstop only:
    # the MCP ``get_probable_pitchers`` tool writes through every call.
    "fetch-mlb-probables": "0 6 * * *",
}

# Sport allowlists for per-sport jobs, cron or event-driven. A job absent
# here runs for every sport in ``data_collection.sports``. Use an allowlist
# for jobs whose ``main()`` raises on unsupported sports (a fire on one would
# raise inside ``job_alert_context`` and spam critical Discord alerts) or
# that should not spend resources on every configured sport.
_JOB_SPORTS: dict[str, tuple[SportKey, ...]] = {
    "fetch-espn-fixtures": ("soccer_epl",),
    # Allowlist derived from the BFE adapter's supported-sports map so adding
    # a new sport requires only a SPORT_CONFIG entry.
    "fetch-betfair-exchange": tuple(BETFAIR_SPORT_CONFIG),
    "fetch-polymarket-odds": MATCH_ODDS_SPORTS,
    "fetch-mlb-probables": ("baseball_mlb",),
    # MLB is settled from the MLB Stats API by fetch-mlb-results.
    "fetch-oddsportal-results": ("soccer_epl",),
    # Odds API quota is spent on EPL only.
    "fetch-odds": ("soccer_epl",),
    "fetch-scores": ("soccer_epl",),
}


def get_job_cron(job_name: str) -> str | None:
    """Return the cron expression for a job, or None if it is event-driven."""
    return _JOB_CRON_MAP.get(job_name)


def get_job_sports(job_name: str) -> tuple[SportKey, ...] | None:
    """Return the sport allowlist for a job, or ``None`` for all configured sports."""
    return _JOB_SPORTS.get(job_name)


assert set(_JOB_CRON_MAP) <= set(_JOB_MODULE_MAP), (
    f"Unknown jobs in _JOB_CRON_MAP: {set(_JOB_CRON_MAP) - set(_JOB_MODULE_MAP)}"
)
assert set(_JOB_SPORTS) <= _PER_SPORT_JOBS, (
    f"Non-per-sport jobs in _JOB_SPORTS: {set(_JOB_SPORTS) - _PER_SPORT_JOBS}"
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

    A per-sport job appends the sport suffix so the schedule id matches the
    rule name (e.g. ``"fetch-odds"`` + ``"soccer_epl"`` -> ``"fetch-odds-epl"``).

    Raises:
        ValueError: If a per-sport job is given no sport, since the bare name
            matches no schedule and would start an unmanaged chain.
    """
    if base_job_name in _PER_SPORT_JOBS:
        if not sport:
            raise ValueError(f"Job '{base_job_name}' requires a sport")
        suffix = sport_key_to_suffix(sport)
        if suffix:
            return f"{base_job_name}-{suffix}"
    return base_job_name


def require_sport(ctx: JobContext, job_name: str) -> SportKey:
    """Return ``ctx.sport`` for a per-sport job, raising if it is unset."""
    if ctx.sport is None:
        raise ValueError(f"Job '{job_name}' requires a sport")
    return ctx.sport


def resolve_job_name(compound_name: str) -> tuple[str, SportKey | None]:
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


def spends_api_quota(job_name: str) -> bool:
    """Whether a run of the job can spend Odds API quota units.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    return base_name in _PAID_API_JOBS


def places_candidate_trades(job_name: str) -> bool:
    """Whether a run of the job places the registry candidates' paper trades.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    return base_name in _CANDIDATE_JOBS


def is_smoke_unsafe(job_name: str) -> bool:
    """Whether a job must never run during a deploy smoke check.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    return base_name in _SMOKE_UNSAFE_JOBS


def resolve_target_sports(job_name: str, configured_sports: Sequence[SportKey]) -> list[SportKey]:
    """Sports a per-sport job actually schedules for, honoring any allowlist.

    A job may restrict itself to a subset of sports via ``_JOB_SPORTS``;
    unrestricted jobs run for every configured sport.
    """
    allowed = get_job_sports(job_name)
    return [s for s in configured_sports if allowed is None or s in allowed]


def expected_compound_job_names(
    bootstrap_jobs: Iterable[str], configured_sports: Sequence[SportKey]
) -> set[str]:
    """Compute the full set of compound job names a config is expected to schedule.

    Mirrors the bootstrap loop's naming so callers (e.g. stale-schedule pruning)
    can compare against what is actually in the data store.
    """
    expected: set[str] = set()
    for job_name in bootstrap_jobs:
        if is_per_sport_job(job_name):
            for sport_key in resolve_target_sports(job_name, configured_sports):
                expected.add(make_compound_job_name(job_name, sport_key))
        else:
            expected.add(job_name)
    return expected


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
                await raw_fn(require_sport(ctx, job_name))

            return _per_sport_wrapper
        else:

            async def _global_wrapper(ctx: JobContext) -> None:
                await raw_fn()

            return _global_wrapper

    return get_job_function(job_name)


def list_available_jobs() -> list[str]:
    """List all registered job names."""
    return sorted(_JOB_MODULE_MAP)
