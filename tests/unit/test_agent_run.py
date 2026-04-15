"""Tests for the agent-run job module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.jobs.agent_run import (
    FALLBACK_INTERVAL_HOURS,
    OVERNIGHT_WINDOWS,
    main,
)
from odds_lambda.scheduling.jobs import JobContext


class TestMainNoSport:
    """Test that main() exits early without a sport."""

    @pytest.mark.asyncio
    async def test_no_sport_returns_early(self) -> None:
        ctx = JobContext(sport=None)
        with patch("odds_lambda.jobs.agent_run._run_claude_agent") as mock_run:
            await main(ctx)
            mock_run.assert_not_called()


class TestMainOrchestration:
    """Test the main() orchestration logic."""

    @pytest.mark.asyncio
    async def test_fallback_scheduled_before_agent_run(self) -> None:
        """Verify fallback pre-scheduling happens before agent subprocess."""
        call_order: list[str] = []

        async def mock_schedule(**kwargs: object) -> None:
            call_order.append("schedule")

        async def mock_run(sport: str) -> int:
            call_order.append("run")
            return 0

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=mock_schedule),
            patch("odds_lambda.jobs.agent_run._run_claude_agent", side_effect=mock_run),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        assert call_order == ["schedule", "run"]

    @pytest.mark.asyncio
    async def test_fallback_reason_is_crash_protection(self) -> None:
        """Fallback pre-schedule has the expected reason."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        assert len(schedule_calls) >= 1
        assert schedule_calls[0]["reason"] == "fallback (crash protection)"
        assert schedule_calls[0]["interval_hours"] == FALLBACK_INTERVAL_HOURS

    @pytest.mark.asyncio
    async def test_nonzero_exit_keeps_fallback(self) -> None:
        """If agent exits non-zero, only fallback schedule should exist."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
            ) as mock_wakeup,
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        # Agent wakeup should not be checked on non-zero exit
        mock_wakeup.assert_not_called()
        # Only the fallback schedule
        assert len(schedule_calls) == 1

    @pytest.mark.asyncio
    async def test_agent_wakeup_overrides_fallback(self) -> None:
        """Agent-requested wakeup triggers a second schedule call."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        override_time = datetime.now(UTC) + timedelta(hours=2)

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=override_time,
            ),
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        # Two schedule calls: fallback + override
        assert len(schedule_calls) == 2
        assert schedule_calls[0]["reason"] == "fallback (crash protection)"
        assert schedule_calls[1]["reason"] == "agent-requested"

    @pytest.mark.asyncio
    async def test_agent_wakeup_in_past_ignored(self) -> None:
        """Agent-requested wakeup in the past should be ignored."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        past_time = datetime.now(UTC) - timedelta(hours=1)

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=past_time,
            ),
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        # Only the fallback call, no override
        assert len(schedule_calls) == 1
        assert schedule_calls[0]["reason"] == "fallback (crash protection)"

    @pytest.mark.asyncio
    async def test_no_wakeup_requested_keeps_fallback(self) -> None:
        """When agent does not request a wakeup, only fallback remains."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
        ):
            await main(JobContext(sport="soccer_epl"))

        assert len(schedule_calls) == 1


class TestOvernightWindowPerSport:
    """Tests for sport-aware overnight suppression windows."""

    @pytest.mark.asyncio
    async def test_unknown_sport_raises_valueerror(self) -> None:
        ctx = JobContext(sport="basketball_nba")
        with (
            patch("odds_core.config.get_settings"),
            pytest.raises(ValueError, match="No overnight window configured for basketball_nba"),
        ):
            await main(ctx)

    @pytest.mark.asyncio
    async def test_mlb_uses_mlb_overnight_window(self) -> None:
        """MLB fallback landing at 08:00 UTC (inside 06:00-14:00) should be pushed to 14:00."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Set now so that now + 12h lands at 08:00 UTC (inside MLB window 06-14)
        fake_now = datetime(2026, 7, 14, 20, 0, tzinfo=UTC)

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
            patch("odds_lambda.jobs.agent_run.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await main(JobContext(sport="baseball_mlb"))

        assert len(schedule_calls) == 1
        scheduled_time = schedule_calls[0]["next_time"]
        # Should be pushed to 14:00 UTC
        assert scheduled_time.hour == 14

    @pytest.mark.asyncio
    async def test_epl_uses_epl_overnight_window(self) -> None:
        """EPL fallback landing at 23:00 UTC (inside 22:00-06:00) should be pushed to 06:00."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Set now so that now + 12h lands at 23:00 UTC (inside EPL window 22-06)
        fake_now = datetime(2026, 4, 18, 11, 0, tzinfo=UTC)

        with (
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run._run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
            patch("odds_lambda.jobs.agent_run.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await main(JobContext(sport="soccer_epl"))

        assert len(schedule_calls) == 1
        scheduled_time = schedule_calls[0]["next_time"]
        # Should be pushed to 06:00 UTC next day
        assert scheduled_time.hour == 6

    def test_overnight_windows_has_no_default(self) -> None:
        """OVERNIGHT_WINDOWS should be an explicit dict with no fallback."""
        assert isinstance(OVERNIGHT_WINDOWS, dict)
        assert "soccer_epl" in OVERNIGHT_WINDOWS
        assert "baseball_mlb" in OVERNIGHT_WINDOWS
