"""A hung or failed run cannot end its job's self-scheduling chain (#485)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from apscheduler import AsyncScheduler
from odds_lambda.scheduling import runner
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.jobs import JobContext

_BUILD_PATCH = "odds_lambda.scheduling.backends.local.build_scheduler"
_JOB_FUNCTION_PATCH = "odds_lambda.scheduling.jobs.get_job_function"
_FAILURE_ALERT_PATCH = "odds_core.alerts.send_job_failure"

_runs: list[tuple[str | None, int]] = []
_other_chain_ran: asyncio.Event | None = None
_second_run_done: asyncio.Event | None = None
_next_run_booked: datetime | None = None
_booking_delay = timedelta(milliseconds=200)


def _in_memory_scheduler(**kwargs: object) -> AsyncScheduler:
    return AsyncScheduler()


async def _hang_for_epl_only(ctx: JobContext) -> None:
    assert _other_chain_ran is not None
    if ctx.sport == "soccer_epl":
        await asyncio.sleep(3600)
    _other_chain_ran.set()


async def _book_then_hang_once(ctx: JobContext) -> None:
    global _next_run_booked
    assert _second_run_done is not None
    _runs.append((ctx.sport, len(_runs)))
    if len(_runs) == 1:
        await asyncio.sleep(0.05)
        _next_run_booked = datetime.now(UTC) + _booking_delay
        await LocalSchedulerBackend().schedule_next_execution(
            "fetch-oddsportal-epl", _next_run_booked
        )
        await asyncio.sleep(3600)
    _second_run_done.set()


async def _fail_before_booking(ctx: JobContext) -> None:
    await asyncio.sleep(0.05)
    raise RuntimeError("DNS failure")


async def _succeed_without_booking(ctx: JobContext) -> None:
    return None


@pytest.fixture
def short_timeout() -> Iterator[None]:
    with patch.object(runner, "JOB_TIMEOUT", timedelta(milliseconds=300)):
        yield


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    global _other_chain_ran, _second_run_done, _next_run_booked, _booking_delay
    _runs.clear()
    _other_chain_ran = asyncio.Event()
    _second_run_done = asyncio.Event()
    _next_run_booked = None
    _booking_delay = timedelta(milliseconds=200)
    yield


def _soon() -> datetime:
    return datetime.now(UTC) + timedelta(milliseconds=100)


class TestHungRun:
    @pytest.mark.asyncio
    async def test_hung_run_does_not_block_another_sport_of_the_same_job(self) -> None:
        assert _other_chain_ran is not None
        with (
            patch.object(runner, "JOB_TIMEOUT", timedelta(seconds=30)),
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_hang_for_epl_only),
            patch(_FAILURE_ALERT_PATCH, new=AsyncMock()),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution("fetch-oddsportal-epl", _soon())
                await backend.schedule_next_execution("fetch-oddsportal-mlb", _soon())

                await asyncio.wait_for(_other_chain_ran.wait(), timeout=3.0)

    @pytest.mark.asyncio
    async def test_run_queued_behind_a_hung_run_starts_once_it_times_out(
        self, short_timeout: None
    ) -> None:
        assert _second_run_done is not None
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_book_then_hang_once),
            patch(_FAILURE_ALERT_PATCH, new=AsyncMock()),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution("fetch-oddsportal-epl", _soon())

                await asyncio.wait_for(_second_run_done.wait(), timeout=3.0)

        assert len(_runs) == 2

    @pytest.mark.asyncio
    async def test_timeout_alerts_and_keeps_the_run_the_job_booked(
        self, short_timeout: None
    ) -> None:
        global _booking_delay
        _booking_delay = timedelta(hours=1)
        alert = AsyncMock()
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_book_then_hang_once),
            patch(_FAILURE_ALERT_PATCH, new=alert),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution("fetch-oddsportal-epl", _soon())
                await asyncio.sleep(0.8)

                assert await backend.get_one_off_run("fetch-oddsportal-epl") == _next_run_booked

        job_name, error = alert.await_args.args
        assert job_name == "fetch-oddsportal-epl"
        assert isinstance(error, runner.JobTimeoutError)


class TestFailedRun:
    @pytest.mark.asyncio
    async def test_run_that_raises_before_booking_rebooks_the_chain(self) -> None:
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_fail_before_booking),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution("fetch-oddsportal-epl", _soon())
                await asyncio.sleep(0.4)

                rebooked = await backend.get_one_off_run("fetch-oddsportal-epl")

        assert rebooked is not None
        expected = datetime.now(UTC) + runner.CHAIN_RETRY_DELAY
        assert abs(rebooked - expected) < timedelta(seconds=10)

    @pytest.mark.asyncio
    async def test_run_that_returns_without_booking_is_left_alone(self) -> None:
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_succeed_without_booking),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution("fetch-oddsportal-epl", _soon())
                await asyncio.sleep(0.4)

                assert await backend.get_one_off_run("fetch-oddsportal-epl") is None

    @pytest.mark.asyncio
    async def test_failed_cron_run_does_not_add_a_one_off_run(self) -> None:
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(_JOB_FUNCTION_PATCH, return_value=_fail_before_booking),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_cron("daily-digest-epl", "0 8 * * *")

                with pytest.raises(RuntimeError, match="DNS failure"):
                    await runner.run_chain_job(
                        backend, "daily-digest-epl", lambda: _fail_before_booking(JobContext())
                    )

                assert await backend.get_one_off_run("daily-digest-epl") is None
                assert await backend.get_job_status("daily-digest-epl") is not None
