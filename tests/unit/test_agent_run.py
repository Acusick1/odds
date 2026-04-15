"""Tests for the agent-run job module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.jobs.agent_run import (
    OVERNIGHT_RESUME_UTC,
    OVERNIGHT_START_UTC,
    SKIP_THRESHOLD_HOURS,
    TIER_ACTIVE_HOURS,
    TIER_FAR_HOURS,
    TIER_LINEUP_HOURS,
    TIER_NO_FIXTURES_HOURS,
    TIER_RESEARCH_HOURS,
    TIER_TOO_CLOSE_HOURS,
    _apply_overnight_skip,
    _compute_wake_interval,
    _should_skip_run,
    main,
)
from odds_lambda.scheduling.jobs import JobContext


class TestComputeWakeInterval:
    """Tests for fixture-proximity wake interval tiers."""

    def test_no_fixtures_returns_no_fixtures_tier(self) -> None:
        assert _compute_wake_interval(None) == TIER_NO_FIXTURES_HOURS

    def test_far_out_over_48h(self) -> None:
        assert _compute_wake_interval(72.0) == TIER_FAR_HOURS

    def test_boundary_exactly_48h(self) -> None:
        # > 48 is false at exactly 48, falls to next tier
        assert _compute_wake_interval(48.0) == TIER_RESEARCH_HOURS

    def test_research_window_24_to_48h(self) -> None:
        assert _compute_wake_interval(36.0) == TIER_RESEARCH_HOURS

    def test_boundary_exactly_24h(self) -> None:
        assert _compute_wake_interval(24.0) == TIER_ACTIVE_HOURS

    def test_active_research_6_to_24h(self) -> None:
        assert _compute_wake_interval(12.0) == TIER_ACTIVE_HOURS

    def test_boundary_exactly_6h(self) -> None:
        assert _compute_wake_interval(6.0) == TIER_LINEUP_HOURS

    def test_lineup_window_1_5_to_6h(self) -> None:
        assert _compute_wake_interval(3.0) == TIER_LINEUP_HOURS

    def test_too_close_returns_post_match_tier(self) -> None:
        assert _compute_wake_interval(1.0) == TIER_TOO_CLOSE_HOURS

    def test_boundary_exactly_1_5h(self) -> None:
        # <= 1.5 is the skip zone, returns post-match check-in tier
        assert _compute_wake_interval(SKIP_THRESHOLD_HOURS) == TIER_TOO_CLOSE_HOURS


class TestShouldSkipRun:
    """Tests for skip-run logic."""

    def test_no_fixtures_does_not_skip(self) -> None:
        assert _should_skip_run(None) is False

    def test_far_fixture_does_not_skip(self) -> None:
        assert _should_skip_run(10.0) is False

    def test_close_fixture_skips(self) -> None:
        assert _should_skip_run(1.0) is True

    def test_boundary_at_threshold_skips(self) -> None:
        assert _should_skip_run(SKIP_THRESHOLD_HOURS) is True

    def test_just_above_threshold_does_not_skip(self) -> None:
        assert _should_skip_run(SKIP_THRESHOLD_HOURS + 0.01) is False


class TestApplyOvernightSkip:
    """Tests for overnight suppression."""

    def test_daytime_unchanged(self) -> None:
        dt = datetime(2026, 4, 15, 14, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == dt

    def test_late_night_pushed_to_next_morning(self) -> None:
        dt = datetime(2026, 4, 15, 23, 30, tzinfo=UTC)
        expected = datetime(2026, 4, 16, OVERNIGHT_RESUME_UTC, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == expected

    def test_early_morning_pushed_to_same_day_resume(self) -> None:
        dt = datetime(2026, 4, 15, 3, 0, tzinfo=UTC)
        expected = datetime(2026, 4, 15, OVERNIGHT_RESUME_UTC, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == expected

    def test_exactly_at_overnight_start_pushed(self) -> None:
        dt = datetime(2026, 4, 15, OVERNIGHT_START_UTC, 0, tzinfo=UTC)
        expected = datetime(2026, 4, 16, OVERNIGHT_RESUME_UTC, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == expected

    def test_exactly_at_resume_pushed(self) -> None:
        # hour < OVERNIGHT_RESUME_UTC is overnight, but hour == OVERNIGHT_RESUME_UTC is not
        dt = datetime(2026, 4, 15, OVERNIGHT_RESUME_UTC, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == dt


class TestMainNoSport:
    """Test that main() exits early without a sport."""

    @pytest.mark.asyncio
    async def test_no_sport_returns_early(self) -> None:
        ctx = JobContext(sport=None)
        # Should not raise, just log error and return
        with patch("odds_lambda.jobs.agent_run._get_next_kickoff") as mock_kickoff:
            await main(ctx)
            mock_kickoff.assert_not_called()


class TestMainOrchestration:
    """Test the main() orchestration logic."""

    @pytest.mark.asyncio
    async def test_preschedule_before_agent_run(self) -> None:
        """Verify pre-scheduling happens before agent subprocess."""
        call_order: list[str] = []

        async def mock_schedule(**kwargs: object) -> None:
            call_order.append("schedule")

        async def mock_run(sport: str) -> int:
            call_order.append("run")
            return 0

        with (
            patch("odds_lambda.jobs.agent_run._get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run._self_schedule", side_effect=mock_schedule),
            patch("odds_lambda.jobs.agent_run._run_claude_agent", side_effect=mock_run),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        assert call_order == ["schedule", "run"]

    @pytest.mark.asyncio
    async def test_skip_run_when_too_close(self) -> None:
        """Agent subprocess should not run when too close to kickoff."""
        with (
            patch("odds_lambda.jobs.agent_run._get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run._self_schedule", new_callable=AsyncMock),
            patch("odds_lambda.jobs.agent_run._run_claude_agent", new_callable=AsyncMock) as run,
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = datetime.now(UTC) + timedelta(hours=0.5)
            await main(JobContext(sport="soccer_epl"))
            run.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_override_reschedules(self) -> None:
        """Agent-requested wakeup that is sooner than default triggers reschedule."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        override_time = datetime.now(UTC) + timedelta(hours=2)

        with (
            patch("odds_lambda.jobs.agent_run._get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run._self_schedule", side_effect=track_schedule),
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
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        # Two schedule calls: pre-schedule + override
        assert len(schedule_calls) == 2
        assert schedule_calls[1]["reason"] == "agent-requested override"

    @pytest.mark.asyncio
    async def test_agent_override_in_past_ignored(self) -> None:
        """Agent-requested wakeup in the past should be ignored."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        past_time = datetime.now(UTC) - timedelta(hours=1)

        with (
            patch("odds_lambda.jobs.agent_run._get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run._self_schedule", side_effect=track_schedule),
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
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        # Only the pre-schedule call, no override
        assert len(schedule_calls) == 1
        assert schedule_calls[0]["reason"] == "pre-schedule (default tier)"

    @pytest.mark.asyncio
    async def test_agent_override_later_than_default_ignored(self) -> None:
        """Agent-requested wakeup later than default should not trigger reschedule."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Default for 30h away is TIER_RESEARCH_HOURS (12h), so override at +20h is later
        override_time = datetime.now(UTC) + timedelta(hours=20)

        with (
            patch("odds_lambda.jobs.agent_run._get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run._self_schedule", side_effect=track_schedule),
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
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        # Only the pre-schedule call, no override
        assert len(schedule_calls) == 1
