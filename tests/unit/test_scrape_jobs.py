"""Tests for scrape job queue: MCP tools (refresh_scrape, get_scrape_status) and worker claiming."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from odds_core.scrape_job_models import ScrapeJob, ScrapeJobStatus
from sqlalchemy import select


def _patch_session(session):
    """Patch async_session_maker to yield the test session instead of creating a new one."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_maker():
        yield session

    return patch("odds_mcp.server.async_session_maker", _fake_session_maker)


def _patch_worker_session(session):
    """Patch async_session_maker for worker module."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_maker():
        yield session

    return patch("odds_cli.commands.worker.async_session_maker", _fake_session_maker)


class TestRefreshScrape:
    """Tests for the refresh_scrape MCP tool."""

    @pytest.mark.asyncio
    async def test_creates_pending_job(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        with _patch_session(test_session):
            result = await refresh_scrape(league="england-premier-league", market="1x2")

        assert "error" not in result
        assert result["job_id"] is not None
        assert result["status"] == "pending"
        assert result["league"] == "england-premier-league"
        assert result["market"] == "1x2"
        assert "created_at" in result

        # Verify persisted in DB
        row = await test_session.execute(select(ScrapeJob).where(ScrapeJob.id == result["job_id"]))
        job = row.scalar_one()
        assert job.status == ScrapeJobStatus.PENDING
        assert job.league == "england-premier-league"
        assert job.market == "1x2"

    @pytest.mark.asyncio
    async def test_invalid_league_returns_error(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        with _patch_session(test_session):
            result = await refresh_scrape(league="nonexistent-league", market="1x2")

        assert "error" in result
        assert "Unknown league" in result["error"]
        assert "nonexistent-league" in result["error"]

    @pytest.mark.asyncio
    async def test_deduplicates_pending_job(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        with _patch_session(test_session):
            first = await refresh_scrape(league="england-premier-league", market="1x2")
            second = await refresh_scrape(league="england-premier-league", market="1x2")

        assert first["job_id"] == second["job_id"]
        assert second["message"] == "Existing job found for this league+market"

    @pytest.mark.asyncio
    async def test_deduplicates_running_job(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        # Create a running job directly
        job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        test_session.add(job)
        await test_session.flush()

        with _patch_session(test_session):
            result = await refresh_scrape(league="england-premier-league", market="1x2")

        assert result["job_id"] == job.id
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_does_not_dedup_completed_job(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        # Create a completed job — should NOT be deduped
        old_job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.COMPLETED,
            completed_at=datetime.now(UTC),
        )
        test_session.add(old_job)
        await test_session.flush()

        with _patch_session(test_session):
            result = await refresh_scrape(league="england-premier-league", market="1x2")

        assert result["job_id"] != old_job.id
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_different_market_creates_separate_job(self, test_session) -> None:
        from odds_mcp.server import refresh_scrape

        with _patch_session(test_session):
            first = await refresh_scrape(league="england-premier-league", market="1x2")
            second = await refresh_scrape(league="england-premier-league", market="over_under_2_5")

        assert first["job_id"] != second["job_id"]


class TestGetScrapeStatus:
    """Tests for the get_scrape_status MCP tool."""

    @pytest.mark.asyncio
    async def test_pending_job_status(self, test_session) -> None:
        from odds_mcp.server import get_scrape_status

        job = ScrapeJob(league="england-premier-league", market="1x2")
        test_session.add(job)
        await test_session.flush()

        with _patch_session(test_session):
            result = await get_scrape_status(job_id=job.id)

        assert result["status"] == "pending"
        assert result["league"] == "england-premier-league"
        assert result["started_at"] is None
        assert result["completed_at"] is None
        assert "results" not in result

    @pytest.mark.asyncio
    async def test_completed_job_includes_results(self, test_session) -> None:
        from odds_mcp.server import get_scrape_status

        job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.COMPLETED,
            started_at=datetime(2026, 4, 14, 10, 0, tzinfo=UTC),
            completed_at=datetime(2026, 4, 14, 10, 5, tzinfo=UTC),
            matches_scraped=10,
            matches_converted=8,
            events_matched=7,
            events_created=1,
            snapshots_stored=7,
        )
        test_session.add(job)
        await test_session.flush()

        with _patch_session(test_session):
            result = await get_scrape_status(job_id=job.id)

        assert result["status"] == "completed"
        assert result["started_at"] is not None
        assert result["completed_at"] is not None
        assert result["results"]["matches_scraped"] == 10
        assert result["results"]["snapshots_stored"] == 7

    @pytest.mark.asyncio
    async def test_failed_job_includes_error(self, test_session) -> None:
        from odds_mcp.server import get_scrape_status

        job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.FAILED,
            completed_at=datetime(2026, 4, 14, 10, 5, tzinfo=UTC),
            error_message="Playwright timeout",
        )
        test_session.add(job)
        await test_session.flush()

        with _patch_session(test_session):
            result = await get_scrape_status(job_id=job.id)

        assert result["status"] == "failed"
        assert result["error_message"] == "Playwright timeout"
        assert "results" not in result

    @pytest.mark.asyncio
    async def test_nonexistent_job_returns_error(self, test_session) -> None:
        from odds_mcp.server import get_scrape_status

        with _patch_session(test_session):
            result = await get_scrape_status(job_id=99999)

        assert "error" in result
        assert "99999" in result["error"]


class TestWorkerClaiming:
    """Tests for worker job claiming logic."""

    @pytest.mark.asyncio
    async def test_claims_oldest_pending_job(self, test_session) -> None:
        """Worker should claim the oldest pending job."""

        job1 = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            created_at=datetime(2026, 4, 14, 10, 0, tzinfo=UTC),
        )
        job2 = ScrapeJob(
            league="england-premier-league",
            market="over_under_2_5",
            created_at=datetime(2026, 4, 14, 10, 5, tzinfo=UTC),
        )
        test_session.add_all([job1, job2])
        await test_session.flush()

        # Simulate the claiming query from the worker
        with _patch_worker_session(test_session):
            find_query = (
                select(ScrapeJob.id)
                .where(ScrapeJob.status == ScrapeJobStatus.PENDING)
                .order_by(ScrapeJob.created_at.asc())
                .limit(1)
            )
            result = await test_session.execute(find_query)
            claimed_id = result.scalar_one_or_none()

        assert claimed_id == job1.id

    @pytest.mark.asyncio
    async def test_expire_stale_running_jobs(self, test_session) -> None:
        """Running jobs older than stale timeout should be marked failed."""

        stale_job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.RUNNING,
            started_at=datetime(2026, 4, 14, 8, 0, tzinfo=UTC),  # Well past stale timeout
        )
        test_session.add(stale_job)
        await test_session.flush()
        stale_id = stale_job.id

        with _patch_worker_session(test_session):
            from odds_cli.commands.worker import _expire_stale_jobs

            await _expire_stale_jobs()

        # Re-fetch to check status
        result = await test_session.execute(select(ScrapeJob).where(ScrapeJob.id == stale_id))
        job = result.scalar_one()
        assert job.status == ScrapeJobStatus.FAILED
        assert "Expired" in job.error_message

    @pytest.mark.asyncio
    async def test_recent_running_job_not_expired(self, test_session) -> None:
        """Running jobs within stale timeout should not be expired."""
        recent_job = ScrapeJob(
            league="england-premier-league",
            market="1x2",
            status=ScrapeJobStatus.RUNNING,
            started_at=datetime.now(UTC),  # Just started
        )
        test_session.add(recent_job)
        await test_session.flush()
        job_id = recent_job.id

        with _patch_worker_session(test_session):
            from odds_cli.commands.worker import _expire_stale_jobs

            await _expire_stale_jobs()

        result = await test_session.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one()
        assert job.status == ScrapeJobStatus.RUNNING
