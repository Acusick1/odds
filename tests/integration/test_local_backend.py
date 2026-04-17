"""Integration tests for local scheduler backend (APScheduler 4)."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from apscheduler import AsyncScheduler
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.exceptions import (
    JobNotFoundError,
    SchedulingFailedError,
)
from odds_lambda.scheduling.jobs import JobContext


def _in_memory_scheduler(**kwargs) -> AsyncScheduler:
    """Build an in-memory AsyncScheduler for tests (no Postgres needed)."""
    return AsyncScheduler(**kwargs)


_BUILD_PATCH = "odds_lambda.scheduling.backends.local.build_scheduler"

# Module-level state for execution tracking (APScheduler 4 requires
# serializable references — nested functions are rejected). Events
# are recreated per test to avoid cross-event-loop binding.
_execution_event: asyncio.Event | None = None
_captured_contexts: list[JobContext] = []
_job1_event: asyncio.Event | None = None
_job2_event: asyncio.Event | None = None


def _reset_events() -> None:
    """Recreate events on the current event loop."""
    global _execution_event, _job1_event, _job2_event
    _execution_event = asyncio.Event()
    _captured_contexts.clear()
    _job1_event = asyncio.Event()
    _job2_event = asyncio.Event()


async def _noop_job(ctx: JobContext) -> None:
    """Stub job function for scheduling tests."""


async def _tracking_job(ctx: JobContext) -> None:
    """Job that signals execution."""
    assert _execution_event is not None
    _execution_event.set()


async def _capturing_job(ctx: JobContext) -> None:
    """Job that captures context and signals execution."""
    assert _execution_event is not None
    _captured_contexts.append(ctx)
    _execution_event.set()


async def _multi_job1(ctx: JobContext) -> None:
    """First job in multi-execution test."""
    assert _job1_event is not None
    _job1_event.set()


async def _multi_job2(ctx: JobContext) -> None:
    """Second job in multi-execution test."""
    assert _job2_event is not None
    _job2_event.set()


class TestLocalSchedulerBackend:
    """Tests for local APScheduler 4 backend."""

    def test_validate_configuration_success(self):
        backend = LocalSchedulerBackend()
        validation = backend.validate_configuration()

        assert validation.is_valid is True
        assert len(validation.errors) == 0

    def test_get_backend_name(self):
        backend = LocalSchedulerBackend()
        assert backend.get_backend_name() == "local_apscheduler"

    @pytest.mark.asyncio
    async def test_health_check_not_started(self):
        backend = LocalSchedulerBackend()
        health = await backend.health_check()

        assert health.is_healthy is True
        assert health.backend_name == "local_apscheduler"
        assert "Configuration valid" in health.checks_passed
        assert "initialized" in " ".join(health.checks_passed).lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
            async with LocalSchedulerBackend() as backend:
                health = await backend.health_check()

                assert health.is_healthy is True
                assert "Scheduler running" in health.checks_passed
                assert health.details is not None
                assert health.details["scheduler_state"] == "running"

    @pytest.mark.asyncio
    async def test_context_manager_starts_scheduler(self):
        backend = LocalSchedulerBackend()

        assert backend._scheduler is None

        with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
            async with backend:
                assert backend._scheduler is not None

        assert backend._scheduler is None

    @pytest.mark.asyncio
    async def test_schedule_next_execution(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_noop_job),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "test-job"
                assert jobs[0].next_run_time is not None

    @pytest.mark.asyncio
    async def test_schedule_replaces_existing(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_noop_job),
        ):
            async with LocalSchedulerBackend() as backend:
                time1 = datetime.now(UTC) + timedelta(hours=1)
                time2 = datetime.now(UTC) + timedelta(hours=2)

                await backend.schedule_next_execution(job_name="test-job", next_time=time1)
                await backend.schedule_next_execution(job_name="test-job", next_time=time2)

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "test-job"

    @pytest.mark.asyncio
    async def test_cancel_scheduled_execution(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_noop_job),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1

                await backend.cancel_scheduled_execution(job_name="test-job")

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job_raises(self):
        with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
            async with LocalSchedulerBackend() as backend:
                with pytest.raises(JobNotFoundError):
                    await backend.cancel_scheduled_execution(job_name="nonexistent-job")

    @pytest.mark.asyncio
    async def test_get_job_status_exists(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_noop_job),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                status = await backend.get_job_status(job_name="test-job")

                assert status is not None
                assert status.job_name == "test-job"
                assert status.next_run_time is not None

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self):
        with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
            async with LocalSchedulerBackend() as backend:
                status = await backend.get_job_status(job_name="nonexistent")
                assert status is None

    @pytest.mark.asyncio
    async def test_get_scheduled_jobs_empty(self):
        with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
            async with LocalSchedulerBackend() as backend:
                jobs = await backend.get_scheduled_jobs()
                assert jobs == []

    @pytest.mark.asyncio
    async def test_get_scheduled_jobs_multiple(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_noop_job),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(
                    job_name="job1", next_time=datetime.now(UTC) + timedelta(hours=1)
                )
                await backend.schedule_next_execution(
                    job_name="job2", next_time=datetime.now(UTC) + timedelta(hours=2)
                )
                await backend.schedule_next_execution(
                    job_name="job3", next_time=datetime.now(UTC) + timedelta(hours=3)
                )

                jobs = await backend.get_scheduled_jobs()

                assert len(jobs) == 3
                job_names = {job.job_name for job in jobs}
                assert job_names == {"job1", "job2", "job3"}

    @pytest.mark.asyncio
    async def test_dry_run_mode_schedule(self):
        backend = LocalSchedulerBackend(dry_run=True)

        next_time = datetime.now(UTC) + timedelta(hours=1)
        await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

    @pytest.mark.asyncio
    async def test_dry_run_mode_cancel(self):
        backend = LocalSchedulerBackend(dry_run=True)
        await backend.cancel_scheduled_execution(job_name="test-job")

    @pytest.mark.asyncio
    async def test_dry_run_mode_get_jobs(self):
        backend = LocalSchedulerBackend(dry_run=True)

        jobs = await backend.get_scheduled_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_schedule_unknown_job_raises(self):
        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(
                "odds_lambda.scheduling.jobs.get_job_function",
                side_effect=KeyError("Unknown job 'invalid-job'"),
            ),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                with pytest.raises(SchedulingFailedError):
                    await backend.schedule_next_execution(
                        job_name="invalid-job", next_time=next_time
                    )

    @pytest.mark.asyncio
    async def test_job_execution_happens(self):
        _reset_events()

        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", return_value=_tracking_job),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(milliseconds=100)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                try:
                    await asyncio.wait_for(_execution_event.wait(), timeout=3.0)
                    assert _execution_event.is_set()
                except TimeoutError:
                    pytest.fail("Job did not execute within timeout")

    @pytest.mark.asyncio
    async def test_schedule_compound_job_name(self):
        _reset_events()

        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(
                "odds_lambda.scheduling.jobs.get_job_function",
                return_value=_capturing_job,
            ) as mock_get_job,
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(milliseconds=100)

                await backend.schedule_next_execution(
                    job_name="fetch-oddsportal-epl", next_time=next_time
                )

                mock_get_job.assert_called_once_with("fetch-oddsportal")

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "fetch-oddsportal-epl"

                try:
                    await asyncio.wait_for(_execution_event.wait(), timeout=3.0)
                except TimeoutError:
                    pytest.fail("Compound job did not execute within timeout")

                assert len(_captured_contexts) == 1
                assert _captured_contexts[0].sport == "soccer_epl"

    @pytest.mark.asyncio
    async def test_schedule_non_compound_job_name(self):
        _reset_events()

        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch(
                "odds_lambda.scheduling.jobs.get_job_function",
                return_value=_capturing_job,
            ) as mock_get_job,
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(milliseconds=100)

                await backend.schedule_next_execution(job_name="check-health", next_time=next_time)

                mock_get_job.assert_called_once_with("check-health")

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "check-health"

                try:
                    await asyncio.wait_for(_execution_event.wait(), timeout=3.0)
                except TimeoutError:
                    pytest.fail("Non-compound job did not execute within timeout")

                assert len(_captured_contexts) == 1
                assert _captured_contexts[0].sport is None

    @pytest.mark.asyncio
    async def test_multiple_jobs_execute_independently(self):
        _reset_events()

        def get_job_func(job_name: str):
            return _multi_job1 if job_name == "job1" else _multi_job2

        with (
            patch(_BUILD_PATCH, side_effect=_in_memory_scheduler),
            patch("odds_lambda.scheduling.jobs.get_job_function", side_effect=get_job_func),
        ):
            async with LocalSchedulerBackend() as backend:
                await backend.schedule_next_execution(
                    job_name="job1", next_time=datetime.now(UTC) + timedelta(milliseconds=100)
                )
                await backend.schedule_next_execution(
                    job_name="job2", next_time=datetime.now(UTC) + timedelta(milliseconds=150)
                )

                try:
                    await asyncio.wait_for(
                        asyncio.gather(_job1_event.wait(), _job2_event.wait()),
                        timeout=3.0,
                    )
                    assert _job1_event.is_set()
                    assert _job2_event.is_set()
                except TimeoutError:
                    pytest.fail("Jobs did not execute within timeout")
