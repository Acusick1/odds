"""A hung or failed run cannot end its job's self-scheduling chain (#485)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from apscheduler import AsyncScheduler
from odds_lambda.scheduling import runner
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.jobs import JobContext

_BUILD_PATCH = "odds_lambda.scheduling.backends.local.build_scheduler"
_JOB_FUNCTION_PATCH = "odds_lambda.scheduling.jobs.get_job_function"
_FAILURE_ALERT_PATCH = "odds_lambda.scheduling.runner.send_job_failure"
_WARNING_PATCH = "odds_lambda.scheduling.runner.send_outage_warning"

EPL_JOB = "fetch-oddsportal-epl"
JobFunc = Callable[[JobContext], Awaitable[None]]


@dataclass
class Probe:
    """What the stub jobs observed."""

    runs: list[str | None] = field(default_factory=list)
    other_chain_ran: asyncio.Event = field(default_factory=asyncio.Event)
    second_run_done: asyncio.Event = field(default_factory=asyncio.Event)
    booked: datetime | None = None
    booking_delay: timedelta = timedelta(milliseconds=200)

    def hang_for_epl_only(self) -> JobFunc:
        async def job(ctx: JobContext) -> None:
            if ctx.sport == "soccer_epl":
                await asyncio.sleep(3600)
            self.other_chain_ran.set()

        return job

    def book_then_hang_once(self) -> JobFunc:
        async def job(ctx: JobContext) -> None:
            self.runs.append(ctx.sport)
            if len(self.runs) == 1:
                await asyncio.sleep(0.05)
                self.booked = datetime.now(UTC) + self.booking_delay
                await LocalSchedulerBackend().schedule_next_execution(EPL_JOB, self.booked)
                await asyncio.sleep(3600)
            self.second_run_done.set()

        return job


async def _fail_before_booking(ctx: JobContext) -> None:
    await asyncio.sleep(0.05)
    raise RuntimeError("DNS failure")


async def _succeed_without_booking(ctx: JobContext) -> None:
    return None


def _in_memory_scheduler(**kwargs: object) -> AsyncScheduler:
    return AsyncScheduler()


def _soon() -> datetime:
    return datetime.now(UTC) + timedelta(milliseconds=100)


@pytest.fixture
def probe() -> Probe:
    return Probe()


@pytest.fixture
def short_timeout() -> Iterator[None]:
    with patch.object(runner, "JOB_TIMEOUT", timedelta(milliseconds=300)):
        yield


@pytest.fixture
def failure_alert() -> Iterator[AsyncMock]:
    with patch(_FAILURE_ALERT_PATCH, new=AsyncMock()) as alert:
        yield alert


@pytest.fixture
def warning() -> Iterator[AsyncMock]:
    with patch(_WARNING_PATCH, new=AsyncMock()) as alert:
        yield alert


@pytest.fixture
def in_memory_store() -> Iterator[None]:
    with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
        yield


def _job(func: JobFunc):
    return patch(_JOB_FUNCTION_PATCH, return_value=func)


@pytest.mark.usefixtures("in_memory_store", "failure_alert", "warning")
class TestHungRun:
    @pytest.mark.asyncio
    async def test_hung_run_does_not_block_another_sport_of_the_same_job(
        self, probe: Probe
    ) -> None:
        with (
            patch.object(runner, "JOB_TIMEOUT", timedelta(seconds=30)),
            _job(probe.hang_for_epl_only()),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await backend.schedule_next_execution("fetch-oddsportal-mlb", _soon())

                await asyncio.wait_for(probe.other_chain_ran.wait(), timeout=3.0)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("short_timeout")
    async def test_run_queued_behind_a_hung_run_starts_once_it_times_out(
        self, probe: Probe
    ) -> None:
        with _job(probe.book_then_hang_once()):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())

                await asyncio.wait_for(probe.second_run_done.wait(), timeout=3.0)

        assert len(probe.runs) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("short_timeout")
    async def test_timeout_alerts_and_keeps_the_run_the_job_booked(
        self, probe: Probe, failure_alert: AsyncMock
    ) -> None:
        probe.booking_delay = timedelta(hours=1)
        with _job(probe.book_then_hang_once()):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await asyncio.sleep(0.8)

                assert await backend.get_one_off_run(EPL_JOB) == probe.booked

        job_name, error = failure_alert.await_args.args
        assert job_name == EPL_JOB
        assert isinstance(error, runner.JobTimeoutError)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("short_timeout")
    async def test_failing_timeout_alert_does_not_stop_the_chain_rebooking(
        self, probe: Probe, failure_alert: AsyncMock
    ) -> None:
        failure_alert.side_effect = RuntimeError("database stalled")
        with _job(probe.book_then_hang_once()):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await asyncio.wait_for(probe.second_run_done.wait(), timeout=3.0)

    @pytest.mark.asyncio
    async def test_hanging_alert_is_abandoned_after_the_recovery_timeout(self) -> None:
        async def hangs(*args: object) -> None:
            await asyncio.sleep(3600)

        async def hung_run() -> None:
            await asyncio.sleep(3600)

        with (
            patch.object(runner, "RECOVERY_TIMEOUT", timedelta(milliseconds=100)),
            patch(_FAILURE_ALERT_PATCH, new=hangs),
            patch.object(runner, "_keep_chain_alive", new=AsyncMock()) as keep_alive,
        ):
            with pytest.raises(runner.JobTimeoutError):
                await asyncio.wait_for(
                    runner.run_chain_job(
                        LocalSchedulerBackend(),
                        EPL_JOB,
                        hung_run,
                        timeout=timedelta(milliseconds=100),
                    ),
                    timeout=2.0,
                )

        keep_alive.assert_awaited_once()


@pytest.mark.usefixtures("in_memory_store", "failure_alert")
class TestFailedRun:
    @pytest.mark.asyncio
    async def test_run_that_raises_before_booking_rebooks_the_chain(
        self, warning: AsyncMock
    ) -> None:
        with _job(_fail_before_booking):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await asyncio.sleep(0.4)

                rebooked = await backend.get_one_off_run(EPL_JOB)

        assert rebooked is not None
        expected = datetime.now(UTC) + runner.CHAIN_RETRY_DELAY
        assert abs(rebooked - expected) < timedelta(seconds=10)
        assert warning.await_args.args[1] == f"job_chain_rebooked:{EPL_JOB}"

    @pytest.mark.asyncio
    async def test_failing_rebook_warning_is_swallowed(self, warning: AsyncMock) -> None:
        warning.side_effect = RuntimeError("database stalled")
        with _job(_fail_before_booking):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await asyncio.sleep(0.4)

                assert await backend.get_one_off_run(EPL_JOB) is not None

    @pytest.mark.asyncio
    async def test_run_that_returns_without_booking_is_left_alone(self, warning: AsyncMock) -> None:
        with _job(_succeed_without_booking):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(EPL_JOB, _soon())
                await asyncio.sleep(0.4)

                assert await backend.get_one_off_run(EPL_JOB) is None
        warning.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_cron_run_does_not_add_a_one_off_run(self, warning: AsyncMock) -> None:
        with _job(_fail_before_booking):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_cron("daily-digest-epl", "0 8 * * *")

                with pytest.raises(RuntimeError, match="DNS failure"):
                    await runner.run_chain_job(
                        backend, "daily-digest-epl", lambda: _fail_before_booking(JobContext())
                    )

                assert await backend.get_one_off_run("daily-digest-epl") is None
                assert await backend.get_job_status("daily-digest-epl") is not None
        warning.assert_not_awaited()
