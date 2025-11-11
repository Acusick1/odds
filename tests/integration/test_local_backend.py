"""Integration tests for local scheduler backend."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
from odds_lambda.scheduling.exceptions import (
    JobNotFoundError,
    SchedulingFailedError,
)


class TestLocalSchedulerBackend:
    """Tests for local APScheduler backend."""

    def test_validate_configuration_success(self):
        """Test that configuration validation passes with all requirements."""
        backend = LocalSchedulerBackend()
        validation = backend.validate_configuration()

        assert validation.is_valid is True
        assert len(validation.errors) == 0

    def test_get_backend_name(self):
        """Test backend name identifier."""
        backend = LocalSchedulerBackend()
        assert backend.get_backend_name() == "local_apscheduler"

    @pytest.mark.asyncio
    async def test_health_check_not_started(self):
        """Test health check when scheduler is not started."""
        backend = LocalSchedulerBackend()
        health = await backend.health_check()

        assert health.is_healthy is True
        assert health.backend_name == "local_apscheduler"
        assert "Configuration valid" in health.checks_passed
        assert "initialized" in " ".join(health.checks_passed).lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """Test health check when scheduler is running."""
        async with LocalSchedulerBackend() as backend:
            health = await backend.health_check()

            assert health.is_healthy is True
            assert "Scheduler running" in health.checks_passed
            assert health.details is not None
            assert health.details["scheduler_state"] == "running"

    @pytest.mark.asyncio
    async def test_context_manager_starts_scheduler(self):
        """Test that async context manager starts the scheduler."""
        backend = LocalSchedulerBackend()

        assert backend._started is False

        async with backend:
            assert backend._started is True

        # Scheduler should be stopped after exit
        assert backend._started is False

    @pytest.mark.asyncio
    async def test_schedule_next_execution(self):
        """Test scheduling a job at a specific time."""
        # Create mock job function
        mock_job = AsyncMock()

        # Patch job registry to return our mock
        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=mock_job):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                # Verify job was scheduled
                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "test-job"
                assert jobs[0].next_run_time is not None

    @pytest.mark.asyncio
    async def test_schedule_replaces_existing(self):
        """Test that scheduling replaces existing job with same name."""
        mock_job = AsyncMock()

        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=mock_job):
            async with LocalSchedulerBackend() as backend:
                time1 = datetime.now(UTC) + timedelta(hours=1)
                time2 = datetime.now(UTC) + timedelta(hours=2)

                # Schedule first time
                await backend.schedule_next_execution(job_name="test-job", next_time=time1)

                # Schedule again with different time (should replace)
                await backend.schedule_next_execution(job_name="test-job", next_time=time2)

                # Should only have one job
                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "test-job"

    @pytest.mark.asyncio
    async def test_cancel_scheduled_execution(self):
        """Test cancelling a scheduled job."""
        mock_job = AsyncMock()

        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=mock_job):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                # Schedule job
                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                # Verify it's scheduled
                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1

                # Cancel it
                await backend.cancel_scheduled_execution(job_name="test-job")

                # Should be gone
                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job_raises(self):
        """Test that cancelling non-existent job raises error."""
        async with LocalSchedulerBackend() as backend:
            with pytest.raises(JobNotFoundError):
                await backend.cancel_scheduled_execution(job_name="nonexistent-job")

    @pytest.mark.asyncio
    async def test_get_job_status_exists(self):
        """Test getting status of an existing job."""
        mock_job = AsyncMock()

        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=mock_job):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                # Get status
                status = await backend.get_job_status(job_name="test-job")

                assert status is not None
                assert status.job_name == "test-job"
                assert status.next_run_time is not None

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self):
        """Test getting status of non-existent job returns None."""
        async with LocalSchedulerBackend() as backend:
            status = await backend.get_job_status(job_name="nonexistent")
            assert status is None

    @pytest.mark.asyncio
    async def test_get_scheduled_jobs_empty(self):
        """Test getting jobs when none are scheduled."""
        async with LocalSchedulerBackend() as backend:
            jobs = await backend.get_scheduled_jobs()
            assert jobs == []

    @pytest.mark.asyncio
    async def test_get_scheduled_jobs_multiple(self):
        """Test getting multiple scheduled jobs."""
        mock_job = AsyncMock()

        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=mock_job):
            async with LocalSchedulerBackend() as backend:
                # Schedule multiple jobs
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
        """Test that dry-run mode logs but doesn't actually schedule."""
        backend = LocalSchedulerBackend(dry_run=True)

        # Don't need to start backend in dry-run
        next_time = datetime.now(UTC) + timedelta(hours=1)

        # Should not raise error and should log
        await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

        # Verify nothing was actually scheduled (can't check jobs since not started)
        # Just verify no errors occurred

    @pytest.mark.asyncio
    async def test_dry_run_mode_cancel(self):
        """Test that dry-run mode logs cancel but doesn't actually cancel."""
        backend = LocalSchedulerBackend(dry_run=True)

        # Should not raise error
        await backend.cancel_scheduled_execution(job_name="test-job")

    @pytest.mark.asyncio
    async def test_dry_run_mode_get_jobs(self):
        """Test that dry-run mode returns empty list for get_jobs."""
        backend = LocalSchedulerBackend(dry_run=True)

        jobs = await backend.get_scheduled_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_schedule_unknown_job_raises(self):
        """Test that scheduling unknown job raises SchedulingFailedError."""
        # Patch to raise KeyError (job not in registry)
        with patch(
            "odds_lambda.scheduling.jobs.get_job_function",
            side_effect=KeyError("Unknown job 'invalid-job'"),
        ):
            async with LocalSchedulerBackend() as backend:
                next_time = datetime.now(UTC) + timedelta(hours=1)

                with pytest.raises(SchedulingFailedError):
                    await backend.schedule_next_execution(
                        job_name="invalid-job", next_time=next_time
                    )

    @pytest.mark.asyncio
    async def test_job_execution_happens(self):
        """Test that scheduled job actually executes at the right time."""
        # Create a flag to track execution
        executed = asyncio.Event()

        async def test_job():
            executed.set()

        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=test_job):
            async with LocalSchedulerBackend() as backend:
                # Schedule job to run very soon (100ms)
                next_time = datetime.now(UTC) + timedelta(milliseconds=100)

                await backend.schedule_next_execution(job_name="test-job", next_time=next_time)

                # Wait for job to execute (with timeout)
                try:
                    await asyncio.wait_for(executed.wait(), timeout=2.0)
                    assert executed.is_set()
                except TimeoutError:
                    pytest.fail("Job did not execute within timeout")

    @pytest.mark.asyncio
    async def test_multiple_jobs_execute_independently(self):
        """Test that multiple jobs execute independently."""
        job1_executed = asyncio.Event()
        job2_executed = asyncio.Event()

        async def job1():
            job1_executed.set()

        async def job2():
            job2_executed.set()

        def get_job_func(job_name):
            return job1 if job_name == "job1" else job2

        with patch("odds_lambda.scheduling.jobs.get_job_function", side_effect=get_job_func):
            async with LocalSchedulerBackend() as backend:
                # Schedule both jobs
                await backend.schedule_next_execution(
                    job_name="job1", next_time=datetime.now(UTC) + timedelta(milliseconds=100)
                )
                await backend.schedule_next_execution(
                    job_name="job2", next_time=datetime.now(UTC) + timedelta(milliseconds=150)
                )

                # Wait for both to execute
                try:
                    await asyncio.wait_for(
                        asyncio.gather(job1_executed.wait(), job2_executed.wait()),
                        timeout=2.0,
                    )
                    assert job1_executed.is_set()
                    assert job2_executed.is_set()
                except TimeoutError:
                    pytest.fail("Jobs did not execute within timeout")
