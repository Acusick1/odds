"""Scheduler start: bootstrap keeps self-scheduled runs, and UTC crons fire on UTC time."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from apscheduler import AsyncScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend, utc_cron_trigger
from odds_lambda.scheduling.bootstrap import BootstrapAction, bootstrap_job
from odds_lambda.scheduling.exceptions import SchedulingFailedError
from odds_lambda.scheduling.jobs import JobContext

_BUILD_PATCH = "odds_lambda.scheduling.backends.local.build_scheduler"
_JOB_FUNCTION_PATCH = "odds_lambda.scheduling.jobs.get_job_function"

_bootstrap_calls: list[JobContext] = []


async def _recording_job(ctx: JobContext) -> None:
    _bootstrap_calls.append(ctx)


def _in_memory_scheduler(**kwargs: object) -> AsyncScheduler:
    return AsyncScheduler()


@pytest.fixture
def local_timezone_bst(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run as the mini-PC does, on Europe/London local time."""
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def recorded_bootstraps() -> Iterator[list[JobContext]]:
    _bootstrap_calls.clear()
    with (
        patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
        patch(_JOB_FUNCTION_PATCH, return_value=_recording_job),
    ):
        yield _bootstrap_calls


SELF_SCHEDULING_JOBS = [
    "fetch-betfair-exchange",
    "fetch-polymarket-odds",
    "fetch-oddsportal",
    "fetch-odds",
]


class TestBootstrapSelfSchedulingJob:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("job", SELF_SCHEDULING_JOBS)
    async def test_restart_keeps_pending_run(
        self, job: str, recorded_bootstraps: list[JobContext]
    ) -> None:
        """2026-10-17: a restart replaced the 13:15Z closing-tier run with a cron firing at 14:30Z."""
        pending = datetime.now(UTC) + timedelta(minutes=14)
        async with LocalSchedulerBackend() as backend:
            await backend.schedule_next_execution(f"{job}-epl", pending)

            result = await bootstrap_job(backend, job, "soccer_epl")

            assert result.action == BootstrapAction.PENDING_KEPT
            status = await backend.get_job_status(f"{job}-epl")
            assert status is not None
            assert status.next_run_time == pending
            assert backend._scheduler is not None
            schedule = await backend._scheduler.get_schedule(f"{job}-epl")
            assert isinstance(schedule.trigger, DateTrigger)
        assert recorded_bootstraps == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job", SELF_SCHEDULING_JOBS)
    async def test_without_pending_run_job_computes_its_own(
        self, job: str, recorded_bootstraps: list[JobContext]
    ) -> None:
        async with LocalSchedulerBackend() as backend:
            result = await bootstrap_job(backend, job, "soccer_epl")

        assert result.action == BootstrapAction.BOOTSTRAPPED
        assert recorded_bootstraps == [JobContext(sport="soccer_epl")]

    @pytest.mark.asyncio
    async def test_pending_run_stored_for_the_job_function_moves_to_the_runner(
        self, recorded_bootstraps: list[JobContext]
    ) -> None:
        """Schedules written before the runner existed ran the job function as their own task."""
        pending = datetime.now(UTC) + timedelta(minutes=14)
        ctx = JobContext(sport="soccer_epl")
        async with LocalSchedulerBackend() as backend:
            assert backend._scheduler is not None
            await backend._scheduler.add_schedule(
                _recording_job,
                trigger=DateTrigger(run_time=pending),
                id="fetch-betfair-exchange-epl",
                args=[ctx],
            )

            result = await bootstrap_job(backend, "fetch-betfair-exchange", "soccer_epl")

            assert result.action == BootstrapAction.PENDING_KEPT
            schedule = await backend._scheduler.get_schedule("fetch-betfair-exchange-epl")
            assert schedule.task_id == "fetch-betfair-exchange-epl"
            assert schedule.args == ("fetch-betfair-exchange-epl", ctx)
            assert schedule.next_fire_time == pending
            task = await backend._scheduler.data_store.get_task("fetch-betfair-exchange-epl")
            assert task.func == "odds_lambda.scheduling.runner:run_scheduled_job"

    @pytest.mark.asyncio
    async def test_leftover_cron_is_replaced_by_bootstrap(
        self, recorded_bootstraps: list[JobContext]
    ) -> None:
        """A cron stored before the job self-scheduled is not a pending run of the job's own."""
        async with LocalSchedulerBackend() as backend:
            await backend.schedule_cron("fetch-betfair-exchange-epl", "*/30 * * * *")

            result = await bootstrap_job(backend, "fetch-betfair-exchange", "soccer_epl")

            assert result.action == BootstrapAction.BOOTSTRAPPED
            assert result.existing_run is not None
            assert await backend.get_job_status("fetch-betfair-exchange-epl") is None
        assert recorded_bootstraps == [JobContext(sport="soccer_epl")]

    @pytest.mark.asyncio
    async def test_bootstrap_failure_is_reported(self) -> None:
        async def failing(ctx: JobContext) -> None:
            raise RuntimeError("betfair login failed")

        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=failing),
        ):
            async with LocalSchedulerBackend() as backend:
                result = await bootstrap_job(backend, "fetch-betfair-exchange", "soccer_epl")

        assert result.action == BootstrapAction.FAILED
        assert result.error == "betfair login failed"


class TestBootstrapCronJob:
    @pytest.mark.asyncio
    async def test_cron_is_reapplied_over_existing_schedule(
        self, recorded_bootstraps: list[JobContext]
    ) -> None:
        async with LocalSchedulerBackend() as backend:
            await backend.schedule_cron("fetch-espn-fixtures-epl", "0 9 * * *")

            result = await bootstrap_job(backend, "fetch-espn-fixtures", "soccer_epl")

            assert result.action == BootstrapAction.CRON_APPLIED
            assert result.cron_expr == "0 6 * * *"
            assert backend._scheduler is not None
            schedule = await backend._scheduler.get_schedule("fetch-espn-fixtures-epl")
            assert isinstance(schedule.trigger, CronTrigger)
            assert schedule.next_fire_time is not None
            assert schedule.next_fire_time.astimezone(UTC).hour == 6
        assert recorded_bootstraps == []

    @pytest.mark.asyncio
    async def test_cron_job_cannot_self_schedule(
        self, recorded_bootstraps: list[JobContext]
    ) -> None:
        async with LocalSchedulerBackend() as backend:
            with pytest.raises(SchedulingFailedError, match="cron job"):
                await backend.schedule_next_execution(
                    "fetch-espn-fixtures-epl", datetime.now(UTC) + timedelta(hours=1)
                )


@pytest.mark.usefixtures("local_timezone_bst")
class TestUtcCronTrigger:
    def test_first_fire_is_next_utc_match(self) -> None:
        """At 13:01:41Z on a BST host, ``*/30`` fires at 13:30Z, not 14:30Z."""
        trigger = utc_cron_trigger(
            "*/30 * * * *", now=datetime(2026, 10, 17, 13, 1, 41, tzinfo=UTC)
        )

        assert trigger.next() == datetime(2026, 10, 17, 13, 30, tzinfo=UTC)

    def test_non_utc_now_is_converted(self) -> None:
        bst = timezone(timedelta(hours=1))
        trigger = utc_cron_trigger(
            "*/30 * * * *", now=datetime(2026, 10, 17, 14, 1, 41, tzinfo=bst)
        )

        assert trigger.next() == datetime(2026, 10, 17, 13, 30, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_schedule_cron_starts_from_utc_now(
        self, recorded_bootstraps: list[JobContext]
    ) -> None:
        async with LocalSchedulerBackend() as backend:
            await backend.schedule_cron("fetch-espn-fixtures-epl", "*/30 * * * *")

            assert backend._scheduler is not None
            schedule = await backend._scheduler.get_schedule("fetch-espn-fixtures-epl")
            assert schedule.next_fire_time is not None
            assert schedule.next_fire_time - datetime.now(UTC) <= timedelta(minutes=30)
