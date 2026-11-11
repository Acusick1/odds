"""Centralized job registry for scheduler backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from importlib import import_module

import structlog
from odds_core.job_specs import JOB_SPECS, SportMap
from odds_core.sports import SportKey

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


# Maps sport suffix to sport_key for per-sport job routing.
# e.g. "fetch-odds-epl" resolves to ("fetch-odds", "soccer_epl")
_SPORT_SUFFIX_MAP: dict[str, SportKey] = {
    "epl": "soccer_epl",
    "mlb": "baseball_mlb",
    "nba": "basketball_nba",
    "championship": "soccer_efl_champ",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_one",
}


def get_job_cron(job_name: str) -> str | None:
    spec = JOB_SPECS.get(job_name)
    return spec.cron if spec else None


def get_job_sports(job_name: str) -> tuple[SportKey, ...] | None:
    spec = JOB_SPECS.get(job_name)
    sports = spec.sports if spec else None
    if isinstance(sports, SportMap):
        return tuple(getattr(import_module(sports.module), sports.attribute))
    return sports


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
    if is_per_sport_job(base_job_name):
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
    if compound_name in JOB_SPECS:
        return compound_name, None

    # Try stripping sport suffix: "fetch-odds-epl" -> ("fetch-odds", "epl")
    for suffix, sport_key in _SPORT_SUFFIX_MAP.items():
        if compound_name.endswith(f"-{suffix}"):
            base_name = compound_name[: -(len(suffix) + 1)]
            if base_name in JOB_SPECS and is_per_sport_job(base_name):
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

    if job_name not in JOB_SPECS:
        available = ", ".join(sorted(JOB_SPECS))
        raise KeyError(f"Unknown job '{job_name}'. Available jobs: {available}")

    spec = JOB_SPECS[job_name]
    module_path, func_name = spec.module, spec.function
    module = import_module(module_path)
    fn = getattr(module, func_name)
    _loaded_jobs[job_name] = fn
    return fn


def is_per_sport_job(job_name: str) -> bool:
    """Check whether a job accepts a sport parameter."""
    spec = JOB_SPECS.get(job_name)
    return spec.per_sport if spec else False


def spends_api_quota(job_name: str) -> bool:
    """Whether a run of the job can spend Odds API quota units.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    spec = JOB_SPECS.get(base_name)
    return spec.paid_api if spec else False


def places_candidate_trades(job_name: str) -> bool:
    """Whether a run of the job places the registry candidates' paper trades.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    spec = JOB_SPECS.get(base_name)
    return spec.candidate_trades if spec else False


def is_smoke_unsafe(job_name: str) -> bool:
    """Whether a job must never run during a deploy smoke check.

    Accepts a compound (sport-suffixed) or base job name.
    """
    base_name, _ = resolve_job_name(job_name)
    spec = JOB_SPECS.get(base_name)
    return spec.agent_trades if spec else False


def resolve_target_sports(job_name: str, configured_sports: Sequence[SportKey]) -> list[SportKey]:
    """Sports a per-sport job actually schedules for, honoring any allowlist.

    A job may restrict itself to a subset of sports via its ``JobSpec``;
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
    spec = JOB_SPECS.get(job_name)
    if spec is not None and spec.bootstrap_function is not None:
        module_path, func_name = spec.module, spec.bootstrap_function
        module = import_module(module_path)
        raw_fn = getattr(module, func_name)

        # Wrap custom bootstrap functions to accept JobContext uniformly.
        # Per-sport custom bootstraps expect (sport: str); global ones expect ().
        if is_per_sport_job(job_name):

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
    return sorted(JOB_SPECS)
