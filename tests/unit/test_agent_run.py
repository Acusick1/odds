"""Tests for the agent-run job module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.jobs.agent_run import (
    OVERNIGHT_WINDOWS,
    TIER_ACTIVE_HOURS,
    TIER_FAR_HOURS,
    TIER_LINEUP_HOURS,
    TIER_NO_FIXTURES_HOURS,
    TIER_RESEARCH_HOURS,
    ScheduleResult,
    _compute_wake_interval,
    _log_stream_message,
    _preview_tool_input,
    main,
    schedule_next,
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

    def test_lineup_window_under_6h(self) -> None:
        assert _compute_wake_interval(3.0) == TIER_LINEUP_HOURS

    def test_very_close_to_ko_still_runs(self) -> None:
        assert _compute_wake_interval(0.5) == TIER_LINEUP_HOURS


class TestMainNoSport:
    """Test that main() exits early without a sport."""

    @pytest.mark.asyncio
    async def test_no_sport_returns_early(self) -> None:
        ctx = JobContext(sport=None)
        # Should not raise, just log error and return
        with patch("odds_lambda.jobs.agent_run.get_next_kickoff") as mock_kickoff:
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
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=mock_schedule),
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
    async def test_agent_override_reschedules(self) -> None:
        """Agent-requested wakeup that is sooner than default triggers reschedule."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        override_time = datetime.now(UTC) + timedelta(hours=2)

        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
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
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
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
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        # Only the pre-schedule call, no override
        assert len(schedule_calls) == 1
        assert schedule_calls[0]["reason"] == "pre-schedule (default tier)"

    @pytest.mark.asyncio
    async def test_agent_override_later_than_default_applied(self) -> None:
        """Agent-requested wakeup later than default should still be applied."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Default for 30h away is TIER_RESEARCH_HOURS (12h), so override at +20h is later
        override_time = datetime.now(UTC) + timedelta(hours=20)

        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
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
            mk.return_value = datetime.now(UTC) + timedelta(hours=30)
            await main(JobContext(sport="soccer_epl"))

        # Pre-schedule + agent override
        assert len(schedule_calls) == 2
        assert schedule_calls[1]["reason"] == "agent-requested override"
        assert schedule_calls[1]["next_time"] == override_time


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
        """MLB at 08:00 UTC (inside 06:00-14:00 window) should be pushed to 14:00."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Set now to a time where default interval lands at 08:00 UTC
        fake_now = datetime(2026, 7, 15, 4, 0, tzinfo=UTC)
        # 4h active tier -> wake at 08:00 UTC, inside MLB window (06-14)
        next_ko = fake_now + timedelta(hours=10)

        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
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
            mk.return_value = next_ko
            await main(JobContext(sport="baseball_mlb"))

        assert len(schedule_calls) == 1
        scheduled_time = schedule_calls[0]["next_time"]
        # Should be pushed to 14:00 UTC
        assert scheduled_time.hour == 14

    @pytest.mark.asyncio
    async def test_epl_uses_epl_overnight_window(self) -> None:
        """EPL at 23:00 UTC (inside 22:00-06:00 window) should be pushed to 06:00."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Set now to 19:00 UTC; 4h active tier -> wake at 23:00 UTC, inside EPL window (22-06)
        fake_now = datetime(2026, 4, 18, 19, 0, tzinfo=UTC)
        next_ko = fake_now + timedelta(hours=10)

        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
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
            mk.return_value = next_ko
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


class TestScheduleNext:
    """Tests for schedule_next() called standalone (bootstrap use case)."""

    @pytest.mark.asyncio
    async def test_returns_schedule_result_with_kickoff(self) -> None:
        """schedule_next() returns ScheduleResult with correct hours_until_ko."""
        next_ko = datetime.now(UTC) + timedelta(hours=30)

        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", new_callable=AsyncMock) as sched,
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = next_ko
            result = await schedule_next("soccer_epl")

        assert isinstance(result, ScheduleResult)
        assert result.hours_until_ko is not None
        assert 29.9 < result.hours_until_ko < 30.1
        assert result.compound_job_name == "agent-run-epl"
        assert result.overnight_start_utc == 22
        assert result.overnight_resume_utc == 6
        sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_hours_when_no_fixtures(self) -> None:
        with (
            patch("odds_lambda.jobs.agent_run.get_next_kickoff", new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", new_callable=AsyncMock),
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = None
            result = await schedule_next("soccer_epl")

        assert result.hours_until_ko is None

    @pytest.mark.asyncio
    async def test_unknown_sport_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="No overnight window configured for basketball_nba"):
            await schedule_next("basketball_nba")


class TestPreviewToolInput:
    """Tests for the tool-input formatter used in live main-log events."""

    def test_none_input_returns_none(self) -> None:
        assert _preview_tool_input(None) is None

    def test_empty_dict_returns_none(self) -> None:
        assert _preview_tool_input({}) is None

    def test_single_key_renders_as_key_value(self) -> None:
        assert _preview_tool_input({"file_path": "/tmp/x.md"}) == "file_path=/tmp/x.md"

    def test_multiple_keys_all_rendered_space_separated(self) -> None:
        preview = _preview_tool_input({"command": "ls", "description": "list files"})
        assert preview == "command=ls description=list files"

    def test_long_values_truncated_with_ellipsis(self) -> None:
        long_value = "x" * 200
        preview = _preview_tool_input({"payload": long_value}, max_value_chars=20)
        assert preview is not None
        assert preview.startswith("payload=" + "x" * 20)
        assert preview.endswith("...")

    def test_non_string_values_stringified(self) -> None:
        assert _preview_tool_input({"count": 42, "flag": True}) == "count=42 flag=True"

    def test_newlines_collapsed_to_spaces(self) -> None:
        # Multi-line command or file_text shouldn't break the single-line log format.
        preview = _preview_tool_input({"command": "line1\nline2\r\nline3"})
        assert preview == "command=line1 line2  line3"


class TestLogStreamMessage:
    """Tests for the stream-json message classifier that drives tee logging."""

    def test_tool_use_emits_agent_tool_use(self) -> None:
        msg = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ]
            },
        }
        with patch("odds_lambda.jobs.agent_run.logger") as mock_logger:
            _log_stream_message(msg)

        mock_logger.info.assert_called_once_with(
            "agent_tool_use",
            tool="Bash",
            input_preview="command=ls",
        )

    def test_result_emits_agent_run_summary_with_fields(self) -> None:
        msg = {
            "type": "result",
            "result": "done",
            "num_turns": 7,
            "duration_ms": 12345,
            "total_cost_usd": 0.5,
        }
        with patch("odds_lambda.jobs.agent_run.logger") as mock_logger:
            _log_stream_message(msg)

        mock_logger.info.assert_called_once_with(
            "agent_run_summary",
            text="done",
            num_turns=7,
            duration_ms=12345,
            cost_usd=0.5,
        )

    def test_assistant_text_block_is_ignored(self) -> None:
        msg = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
        with patch("odds_lambda.jobs.agent_run.logger") as mock_logger:
            _log_stream_message(msg)

        mock_logger.info.assert_not_called()

    def test_assistant_with_non_dict_message_is_silent(self) -> None:
        msg = {"type": "assistant", "message": None}
        with patch("odds_lambda.jobs.agent_run.logger") as mock_logger:
            _log_stream_message(msg)

        mock_logger.info.assert_not_called()

    def test_unknown_type_is_silent(self) -> None:
        with patch("odds_lambda.jobs.agent_run.logger") as mock_logger:
            _log_stream_message({"type": "system", "subtype": "init"})

        mock_logger.info.assert_not_called()
