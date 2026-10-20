"""Run a scheduled job so that no failure or hang can end its self-scheduling chain."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import structlog

from odds_lambda.scheduling import jobs
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.jobs import JobContext, get_job_cron, resolve_job_name

logger = structlog.get_logger()

JOB_TIMEOUT = timedelta(minutes=20)
CHAIN_RETRY_DELAY = timedelta(minutes=15)


class JobTimeoutError(TimeoutError):
    def __init__(self, job_name: str, timeout: timedelta) -> None:
        super().__init__(f"{job_name} exceeded {timeout.total_seconds():g}s and was cancelled")


async def run_chain_job(
    backend: LocalSchedulerBackend,
    job_name: str,
    run: Callable[[], Awaitable[None]],
    *,
    timeout: timedelta = JOB_TIMEOUT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Run one execution of ``job_name`` under a hard timeout.

    A run that raises or times out, before or after it booked anything, leaves
    the chain with a run booked ``CHAIN_RETRY_DELAY`` from now unless the job
    had already booked one. A stored run that is not in the future is the one
    being executed, which the scheduler has not yet marked finished. The
    failure still propagates.
    """
    try:
        async with asyncio.timeout(timeout.total_seconds()) as deadline:
            await run()
    except Exception as e:
        failure: Exception = e
        if deadline.expired() and isinstance(e, TimeoutError):
            failure = JobTimeoutError(job_name, timeout)
            logger.error("job_timed_out", job=job_name, timeout_s=timeout.total_seconds())
            from odds_core.alerts import send_job_failure

            await send_job_failure(job_name, failure)
        await _keep_chain_alive(backend, job_name, now())
        if failure is not e:
            raise failure from e
        raise


async def _keep_chain_alive(backend: LocalSchedulerBackend, job_name: str, now: datetime) -> None:
    base_name, _ = resolve_job_name(job_name)
    if get_job_cron(base_name) is not None:
        return
    try:
        booked = await backend.get_one_off_run(job_name)
        if booked is not None and booked > now:
            return
        retry_at = now + CHAIN_RETRY_DELAY
        await backend.schedule_next_execution(job_name, retry_at)
        logger.warning("job_chain_rebooked", job=job_name, next_time=retry_at.isoformat())
    except Exception:
        logger.error("job_chain_rebook_failed", job=job_name, exc_info=True)


async def run_scheduled_job(job_name: str, ctx: JobContext) -> None:
    """Task function behind every local schedule; resolves the job at run time."""
    base_name, _ = resolve_job_name(job_name)
    job = jobs.get_job_function(base_name)
    await run_chain_job(LocalSchedulerBackend(), job_name, lambda: job(ctx), timeout=JOB_TIMEOUT)
