"""Tests for agent wakeup scheduling MCP tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.agent_wakeup_models import AgentWakeup
from sqlalchemy import select


def _patch_session(session):
    """Patch async_session_maker to yield the test session."""

    @asynccontextmanager
    async def _fake_session_maker():
        yield session

    return patch("odds_mcp.server.async_session_maker", _fake_session_maker)


def _patch_build_scheduler(*, schedules=None, error=None):
    """Patch build_scheduler to return a mock AsyncScheduler context manager."""
    mock_scheduler = AsyncMock()
    mock_scheduler.__aenter__ = AsyncMock(return_value=mock_scheduler)
    mock_scheduler.__aexit__ = AsyncMock(return_value=False)
    if error is not None:
        mock_scheduler.get_schedules = AsyncMock(side_effect=error)
    else:
        mock_scheduler.get_schedules = AsyncMock(return_value=list(schedules or []))

    return patch(
        "odds_lambda.scheduling.backends.local.build_scheduler",
        MagicMock(return_value=mock_scheduler),
    )


def _make_schedule(job_id: str, next_fire_time: datetime | None):
    return SimpleNamespace(id=job_id, next_fire_time=next_fire_time)


class TestScheduleNextWakeup:
    """Tests for the schedule_next_wakeup MCP tool."""

    @pytest.mark.asyncio
    async def test_inserts_new_wakeup(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        with _patch_session(test_session):
            result = await schedule_next_wakeup(
                sport="soccer_epl", delay_hours=6.0, reason="Pre-match context"
            )

        assert result["sport"] == "soccer_epl"
        assert result["delay_hours"] == 6.0
        assert result["reason"] == "Pre-match context"
        assert "requested_time" in result

        # Verify persisted
        row = await test_session.execute(
            select(AgentWakeup).where(AgentWakeup.sport_key == "soccer_epl")
        )
        wakeup = row.scalar_one()
        assert wakeup.reason == "Pre-match context"
        assert wakeup.consumed_at is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_same_sport(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        with _patch_session(test_session):
            await schedule_next_wakeup(sport="soccer_epl", delay_hours=6.0, reason="First reason")
            second = await schedule_next_wakeup(
                sport="soccer_epl", delay_hours=12.0, reason="Updated reason"
            )

        assert second["delay_hours"] == 12.0
        assert second["reason"] == "Updated reason"

        # Only one row should exist for this sport
        rows = await test_session.execute(
            select(AgentWakeup).where(AgentWakeup.sport_key == "soccer_epl")
        )
        wakeups = list(rows.scalars().all())
        assert len(wakeups) == 1
        assert wakeups[0].reason == "Updated reason"

    @pytest.mark.asyncio
    async def test_clamps_delay_below_minimum(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        with _patch_session(test_session):
            result = await schedule_next_wakeup(
                sport="soccer_epl", delay_hours=0.1, reason="Too soon"
            )

        assert result["delay_hours"] == 0.5

    @pytest.mark.asyncio
    async def test_clamps_delay_above_maximum(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        with _patch_session(test_session):
            result = await schedule_next_wakeup(
                sport="soccer_epl", delay_hours=500.0, reason="Too far"
            )

        assert result["delay_hours"] == 168.0

    @pytest.mark.asyncio
    async def test_different_sports_coexist(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        with _patch_session(test_session):
            await schedule_next_wakeup(sport="soccer_epl", delay_hours=6.0, reason="EPL")
            await schedule_next_wakeup(sport="baseball_mlb", delay_hours=12.0, reason="MLB")

        rows = await test_session.execute(select(AgentWakeup))
        wakeups = list(rows.scalars().all())
        assert len(wakeups) == 2
        sports = {w.sport_key for w in wakeups}
        assert sports == {"soccer_epl", "baseball_mlb"}

    @pytest.mark.asyncio
    async def test_upsert_resets_consumed_at(self, test_session) -> None:
        from odds_mcp.server import schedule_next_wakeup

        # 1. Insert a wakeup
        with _patch_session(test_session):
            await schedule_next_wakeup(sport="soccer_epl", delay_hours=6.0, reason="Initial")

        # 2. Simulate the consumer marking it consumed
        row = await test_session.execute(
            select(AgentWakeup).where(AgentWakeup.sport_key == "soccer_epl")
        )
        wakeup = row.scalar_one()
        wakeup.consumed_at = datetime.now(UTC)
        await test_session.commit()

        # 3. Upsert again for the same sport
        with _patch_session(test_session):
            await schedule_next_wakeup(sport="soccer_epl", delay_hours=12.0, reason="Rescheduled")

        # 4. Assert consumed_at was reset to None
        test_session.expire_all()
        row = await test_session.execute(
            select(AgentWakeup).where(AgentWakeup.sport_key == "soccer_epl")
        )
        wakeup = row.scalar_one()
        assert wakeup.consumed_at is None
        assert wakeup.reason == "Rescheduled"


class TestGetScheduledJobs:
    """Tests for the get_scheduled_jobs MCP tool."""

    @pytest.mark.asyncio
    async def test_scheduler_unavailable_returns_message(self) -> None:
        from odds_mcp.server import get_scheduled_jobs

        with _patch_build_scheduler(error=RuntimeError("Scheduler not running")):
            result = await get_scheduled_jobs()

        assert result["jobs"] == []
        assert "Scheduler not running" in result["message"]
        assert "Could not query scheduler" in result["message"]

    @pytest.mark.asyncio
    async def test_sport_filtering(self) -> None:
        from odds_mcp.server import get_scheduled_jobs

        now = datetime.now(UTC)
        schedules = [
            _make_schedule("fetch-odds-epl", now),
            _make_schedule("fetch-odds-mlb", now),
            _make_schedule("agent-run-epl", now),
        ]

        with _patch_build_scheduler(schedules=schedules):
            result = await get_scheduled_jobs(sport="soccer_epl")

        assert result["job_count"] == 2
        job_names = [j["job_name"] for j in result["jobs"]]
        assert "fetch-odds-epl" in job_names
        assert "agent-run-epl" in job_names
        assert "fetch-odds-mlb" not in job_names

    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self) -> None:
        from odds_mcp.server import get_scheduled_jobs

        now = datetime.now(UTC)
        schedules = [
            _make_schedule("fetch-odds-epl", now),
            _make_schedule("fetch-odds-mlb", now),
        ]

        with _patch_build_scheduler(schedules=schedules):
            result = await get_scheduled_jobs()

        assert result["job_count"] == 2

    @pytest.mark.asyncio
    async def test_generic_exception_returns_message(self) -> None:
        from odds_mcp.server import get_scheduled_jobs

        with _patch_build_scheduler(error=RuntimeError("Connection refused")):
            result = await get_scheduled_jobs()

        assert result["jobs"] == []
        assert "Connection refused" in result["message"]
