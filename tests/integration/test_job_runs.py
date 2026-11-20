"""Every scheduled job run records its start, finish and outcome, and /perf charts the durations
of each job's recent runs against the time between them."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import get_settings
from odds_core.job_run_models import JobOutcome, JobRun
from odds_core.models import AlertHistory
from odds_lambda.health_monitor import JOB_RUN_RETENTION, HealthMonitor
from odds_lambda.scheduling import runner
from odds_lambda.scheduling.jobs import JobContext
from odds_web.app import create_app
from odds_web.db import WebDatabase
from odds_web.perf import JobDurations, RunPoint, load_job_durations
from odds_web.status import ChipState
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import as_odds_web_login, client_of

JOB = "fetch-oddsportal-epl"
NOW = datetime.now(UTC).replace(microsecond=0)


@pytest.fixture
def run_session(pglite_async_session: AsyncSession) -> Iterator[AsyncSession]:
    """The test session as the runner's session, with chain recovery and alerts stubbed out."""

    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return pglite_async_session

        async def __aexit__(self, *args: object) -> bool:
            return False

    with (
        patch.object(runner, "async_session_maker", _Maker()),
        patch.object(runner, "_keep_chain_alive", new=AsyncMock()),
        patch.object(runner, "send_job_failure", new=AsyncMock()),
    ):
        yield pglite_async_session


async def _runs(session: AsyncSession) -> list[JobRun]:
    return list((await session.scalars(select(JobRun).order_by(JobRun.started_at))).all())


async def _run(job: object) -> None:
    with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=job):
        await runner.run_scheduled_job(JOB, JobContext(sport="soccer_epl"))


class TestRecordingARun:
    async def test_a_run_that_succeeds_records_its_start_finish_and_outcome(
        self, run_session: AsyncSession
    ) -> None:
        async def job(ctx: JobContext) -> None:
            await asyncio.sleep(0.05)

        before = datetime.now(UTC)
        await _run(job)

        [run] = await _runs(run_session)
        assert run.job_name == JOB and run.outcome is JobOutcome.SUCCEEDED
        assert before <= run.started_at
        assert run.finished_at - run.started_at >= timedelta(milliseconds=50)

    async def test_a_run_that_fails_records_a_failure_and_still_raises(
        self, run_session: AsyncSession
    ) -> None:
        async def job(ctx: JobContext) -> None:
            raise RuntimeError("DNS failure")

        with pytest.raises(RuntimeError, match="DNS failure"):
            await _run(job)

        [run] = await _runs(run_session)
        assert run.outcome is JobOutcome.FAILED
        assert run.finished_at >= run.started_at

    async def test_a_run_cancelled_at_the_timeout_records_a_timeout(
        self, run_session: AsyncSession
    ) -> None:
        async def job(ctx: JobContext) -> None:
            await asyncio.sleep(3600)

        with (
            patch.object(runner, "JOB_TIMEOUT", timedelta(milliseconds=100)),
            pytest.raises(runner.JobTimeoutError),
        ):
            await _run(job)

        [run] = await _runs(run_session)
        assert run.outcome is JobOutcome.TIMED_OUT
        assert run.finished_at - run.started_at >= timedelta(milliseconds=100)

    async def test_a_failing_record_does_not_mask_the_runs_failure(
        self, run_session: AsyncSession
    ) -> None:
        async def job(ctx: JobContext) -> None:
            raise RuntimeError("DNS failure")

        with (
            patch.object(runner, "record_job_run", side_effect=RuntimeError("database down")),
            pytest.raises(RuntimeError, match="DNS failure"),
        ):
            await _run(job)

    async def test_a_dry_run_records_nothing(
        self, run_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def job(ctx: JobContext) -> None:
            return None

        monkeypatch.setattr(get_settings().scheduler, "dry_run", True)
        await _run(job)

        assert await _runs(run_session) == []

    async def test_a_run_writes_no_heartbeat_of_its_own(self, run_session: AsyncSession) -> None:
        async def job(ctx: JobContext) -> None:
            return None

        await _run(job)

        assert await run_session.scalar(select(func.count()).select_from(AlertHistory)) == 0


class TestPurge:
    async def test_runs_older_than_the_retention_are_deleted(
        self, pglite_async_session: AsyncSession
    ) -> None:
        settings = get_settings()
        for started in (
            NOW - JOB_RUN_RETENTION - timedelta(hours=1),
            NOW - timedelta(hours=1),
        ):
            pglite_async_session.add(
                JobRun(
                    job_name=JOB,
                    started_at=started,
                    finished_at=started + timedelta(seconds=5),
                    outcome=JobOutcome.SUCCEEDED,
                )
            )
        await pglite_async_session.commit()

        deleted = await HealthMonitor(pglite_async_session, settings).purge_old_job_runs()

        assert deleted == 1
        [kept] = await _runs(pglite_async_session)
        assert kept.started_at == NOW - timedelta(hours=1)


def _durations(*seconds: float, every: timedelta = timedelta(minutes=10)) -> JobDurations:
    return JobDurations(
        JOB,
        [RunPoint(NOW + i * every, s * 1000, JobOutcome.SUCCEEDED) for i, s in enumerate(seconds)],
    )


class TestJobDurations:
    def test_percentiles_latest_and_interval(self) -> None:
        durations = _durations(*range(1, 21))

        assert durations.p50_ms == 10_500
        assert durations.p95_ms == pytest.approx(19_050)
        assert durations.latest_ms == 20_000
        assert durations.interval_ms == 600_000

    @pytest.mark.parametrize(
        ("seconds", "state"),
        [
            (60, ChipState.OK),
            (300, ChipState.LATE),
            (480, ChipState.STALE),
        ],
    )
    def test_a_p95_near_the_interval_is_late_then_stale(
        self, seconds: float, state: ChipState
    ) -> None:
        assert _durations(seconds, seconds, seconds).state is state

    def test_a_single_run_has_no_interval(self) -> None:
        durations = _durations(5)

        assert durations.interval_ms is None
        assert durations.state is ChipState.UNKNOWN


async def _store(
    session: AsyncSession, job: str, seconds: list[float], outcome: JobOutcome
) -> None:
    for i, s in enumerate(seconds):
        started = NOW - timedelta(minutes=10 * (len(seconds) - i))
        session.add(
            JobRun(
                job_name=job,
                started_at=started,
                finished_at=started + timedelta(seconds=s),
                outcome=outcome,
            )
        )


class TestLoadJobDurations:
    async def test_keeps_each_jobs_newest_runs_oldest_first(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await _store(pglite_async_session, JOB, [1, 2, 3, 4], JobOutcome.SUCCEEDED)
        await _store(pglite_async_session, "check-health", [7], JobOutcome.FAILED)
        await pglite_async_session.commit()

        loaded = await load_job_durations(pglite_async_session, limit=3)

        assert [d.job_name for d in loaded] == ["check-health", JOB]
        health, epl = loaded
        assert [r.duration_ms for r in epl.runs] == [2000, 3000, 4000]
        assert health.runs[0].outcome is JobOutcome.FAILED


class TestPerfPageSection:
    async def test_lists_each_job_with_its_durations_state_and_failed_runs(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        # Runs ten minutes apart: the EPL fetch takes six minutes (late), check-health seconds.
        await _store(odds_web_session, JOB, [360, 360, 360], JobOutcome.SUCCEEDED)
        await _store(odds_web_session, "check-health", [2, 2], JobOutcome.SUCCEEDED)
        await _store(odds_web_session, "check-health", [3], JobOutcome.FAILED)
        await odds_web_session.commit()
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                page = (await client.get("/perf")).text
                fragment = (await client.get("/fragments/job-runs")).text

        assert "Job durations" in page and 'id="job-runs"' in page
        assert 'class="unavailable"' not in page
        epl_row = next(row for row in fragment.split("<tr>") if JOB in row)
        assert 'class="dot late"' in epl_row and 'class="spark late"' in epl_row
        assert "6.0 min" in epl_row and "10.0 min" in epl_row
        health_row = next(row for row in fragment.split("<tr>") if "check-health" in row)
        assert 'class="dot ok"' in health_row
        assert health_row.count('class="failed"') == 1 and "failed after 3.0 s" in health_row
        assert "<html" not in fragment

    async def test_no_runs_says_so(self, odds_web_session: AsyncSession, tmp_path: Path) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                fragment = (await client.get("/fragments/job-runs")).text

        assert "no job runs recorded" in fragment

    async def test_a_job_with_one_run_shows_a_single_point(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _store(odds_web_session, JOB, [30], JobOutcome.SUCCEEDED)
        await odds_web_session.commit()
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                fragment = (await client.get("/fragments/job-runs")).text

        assert 'class="dot unknown"' in fragment
        assert 'class="only"' in fragment
        assert "<polyline" not in fragment

    async def test_a_job_whose_one_run_failed_shows_only_the_red_point(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _store(odds_web_session, JOB, [30], JobOutcome.FAILED)
        await odds_web_session.commit()
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                fragment = (await client.get("/fragments/job-runs")).text

        assert fragment.count('class="failed"') == 1
        assert 'class="only"' not in fragment
        assert "<polyline" not in fragment
