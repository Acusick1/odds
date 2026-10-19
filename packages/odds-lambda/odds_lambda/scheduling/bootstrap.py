"""Install one job's schedule when the local scheduler starts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from odds_core.sports import SportKey

from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.jobs import (
    JobContext,
    get_bootstrap_function,
    get_job_cron,
    make_compound_job_name,
)


class BootstrapAction(StrEnum):
    CRON_APPLIED = "cron_applied"
    PENDING_KEPT = "pending_kept"
    BOOTSTRAPPED = "bootstrapped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    job_name: str
    action: BootstrapAction
    pending_run: datetime | None
    cron_expr: str | None = None
    error: str | None = None


async def bootstrap_job(
    backend: LocalSchedulerBackend, base_name: str, sport: SportKey | None
) -> BootstrapResult:
    """Install the schedule for a job, per sport when ``sport`` is set, on scheduler start.

    A cron job has its cron re-applied, so an edit to ``_JOB_CRON_MAP`` takes
    effect. A self-scheduling job keeps any pending run, which it computed
    itself; only without one does its bootstrap function run.
    """
    job_name = make_compound_job_name(base_name, sport)
    cron_expr = get_job_cron(base_name)
    pending = await backend.get_job_status(job_name)
    pending_run = pending.next_run_time if pending is not None else None
    try:
        if cron_expr is not None:
            await backend.schedule_cron(job_name, cron_expr)
            return BootstrapResult(
                job_name, BootstrapAction.CRON_APPLIED, pending_run, cron_expr=cron_expr
            )
        if pending_run is not None:
            return BootstrapResult(job_name, BootstrapAction.PENDING_KEPT, pending_run)
        await get_bootstrap_function(base_name)(JobContext(sport=sport))
        return BootstrapResult(job_name, BootstrapAction.BOOTSTRAPPED, pending_run)
    except Exception as e:
        return BootstrapResult(
            job_name, BootstrapAction.FAILED, pending_run, cron_expr=cron_expr, error=str(e)
        )
