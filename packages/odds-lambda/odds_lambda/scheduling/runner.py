"""Run a scheduled job so that no failure or hang can end its self-scheduling chain."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import send_job_failure, send_job_warning
from odds_core.config import get_settings

from odds_lambda.scheduling import jobs
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.jobs import JobContext, get_job_cron, resolve_job_name

logger = structlog.get_logger()

JOB_TIMEOUT = timedelta(minutes=20)
CHAIN_RETRY_DELAY = timedelta(minutes=15)
RECOVERY_TIMEOUT = timedelta(seconds=30)


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
    being executed, which the scheduler has not yet marked finished. The chain
    is re-booked before any alert is sent, and each step of that recovery is
    bounded by ``RECOVERY_TIMEOUT``. The failure still propagates.
    """
    try:
        async with asyncio.timeout(timeout.total_seconds()) as deadline:
            await run()
    except Exception as e:
        timed_out = deadline.expired() and isinstance(e, TimeoutError)
        failure: Exception = JobTimeoutError(job_name, timeout) if timed_out else e
        if timed_out:
            logger.error("job_timed_out", job=job_name, timeout_s=timeout.total_seconds())
        try:
            await _keep_chain_alive(backend, job_name, now())
        finally:
            if timed_out:
                await _bounded(send_job_failure(job_name, failure), "job_timeout_alert_failed")
        if failure is not e:
            raise failure from e
        raise


async def _bounded[T](step: Awaitable[T], failure_event: str) -> T | None:
    """Await a recovery step for at most ``RECOVERY_TIMEOUT``; a failure is logged, not raised."""
    try:
        async with asyncio.timeout(RECOVERY_TIMEOUT.total_seconds()):
            return await step
    except Exception:
        logger.error(failure_event, exc_info=True)
        return None


async def _keep_chain_alive(backend: LocalSchedulerBackend, job_name: str, now: datetime) -> None:
    base_name, _ = resolve_job_name(job_name)
    if get_job_cron(base_name) is not None:
        return
    booked = await _bounded(backend.get_one_off_run(job_name), "job_chain_lookup_failed")
    if booked is not None and booked > now:
        return
    retry_at = now + CHAIN_RETRY_DELAY

    async def book() -> bool:
        await backend.schedule_next_execution(job_name, retry_at)
        return True

    if await _bounded(book(), "job_chain_rebook_failed"):
        logger.warning("job_chain_rebooked", job=job_name, next_time=retry_at.isoformat())
        await _bounded(
            send_job_warning(
                f"job_chain_rebooked:{job_name}",
                f"⚠️ {job_name} failed and was re-booked for {retry_at:%H:%M} UTC",
            ),
            "job_chain_rebook_alert_failed",
        )


async def run_scheduled_job(job_name: str, ctx: JobContext) -> None:
    """Task function behind every local schedule; resolves the job at run time."""
    base_name, _ = resolve_job_name(job_name)
    job = jobs.get_job_function(base_name)
    await run_chain_job(
        LocalSchedulerBackend(dry_run=get_settings().scheduler.dry_run),
        job_name,
        lambda: job(ctx),
        timeout=JOB_TIMEOUT,
    )
