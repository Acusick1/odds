"""Tests for the agent-run job module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import reset_settings_cache
from odds_lambda.jobs.agent_run import (
    OVERNIGHT_WINDOWS,
    ScheduleResult,
    _build_agent_subprocess_env,
    _log_stream_message,
    _preview_tool_input,
    main,
    schedule_next,
)
from odds_lambda.scheduling.jobs import JobContext

# schedule_next delegates proximity computation to the unified decision engine;
# patch the kickoff query where the engine imports it.
_KICKOFF_PATCH = "odds_lambda.scheduling.decision.get_next_kickoff"


class TestMainNoSport:
    """Test that main() rejects a missing sport."""

    @pytest.mark.asyncio
    async def test_no_sport_raises(self) -> None:
        with patch(_KICKOFF_PATCH) as mock_kickoff:
            with pytest.raises(ValueError, match="requires a sport"):
                await main(JobContext(sport=None))
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
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=mock_schedule),
            patch("odds_lambda.jobs.agent_run.run_claude_agent", side_effect=mock_run),
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
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run.run_claude_agent",
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
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run.run_claude_agent",
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
        assert schedule_calls[0]["reason"].startswith("pre-schedule")

    @pytest.mark.asyncio
    async def test_agent_override_later_than_default_applied(self) -> None:
        """Agent-requested wakeup later than default should still be applied."""
        schedule_calls: list[dict] = []

        async def track_schedule(**kwargs: object) -> None:
            schedule_calls.append(dict(kwargs))

        # Default for 30h away is TIER_RESEARCH_HOURS (12h), so override at +20h is later
        override_time = datetime.now(UTC) + timedelta(hours=20)

        with (
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run.run_claude_agent",
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
        ctx = JobContext(sport="tennis_atp")  # type: ignore[arg-type]
        with (
            patch("odds_core.config.get_settings"),
            pytest.raises(ValueError, match="No overnight window configured for tennis_atp"),
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
        # PREGAME tier (10h to KO) -> 4h interval -> wake at 08:00 UTC,
        # inside MLB window (05-14)
        next_ko = fake_now + timedelta(hours=10)

        with (
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run.run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
            patch("odds_lambda.scheduling.decision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)  # noqa: DTZ001
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
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", side_effect=track_schedule),
            patch(
                "odds_lambda.jobs.agent_run.run_claude_agent",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "odds_lambda.jobs.agent_run._check_agent_requested_wakeup",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("odds_core.config.get_settings"),
            patch("odds_lambda.scheduling.decision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)  # noqa: DTZ001
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
        assert "basketball_nba" in OVERNIGHT_WINDOWS


class TestScheduleNext:
    """Tests for schedule_next() called standalone (bootstrap use case)."""

    @pytest.mark.asyncio
    async def test_returns_schedule_result_with_kickoff(self) -> None:
        """schedule_next() returns a ScheduleResult and schedules once."""
        next_ko = datetime.now(UTC) + timedelta(hours=30)

        with (
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", new_callable=AsyncMock) as sched,
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = next_ko
            result = await schedule_next("soccer_epl")

        assert isinstance(result, ScheduleResult)
        assert result.compound_job_name == "agent-run-epl"
        assert result.next_time is not None
        sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedules_off_season_checkin_when_no_fixtures(self) -> None:
        """With no fixtures, schedule_next still schedules an off-season check-in."""
        with (
            patch(_KICKOFF_PATCH, new_callable=AsyncMock) as mk,
            patch("odds_lambda.jobs.agent_run.self_schedule", new_callable=AsyncMock) as sched,
            patch("odds_core.config.get_settings"),
        ):
            mk.return_value = None
            result = await schedule_next("soccer_epl")

        assert result.next_time is not None
        sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_sport_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="No overnight window configured for tennis_atp"):
            await schedule_next("tennis_atp")  # type: ignore[arg-type]


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


class TestBuildAgentSubprocessEnv:
    """Tests for the AGENT_DATABASE_URL override behaviour."""

    PARENT_ENV = {"DATABASE_URL": "postgresql://parent:pw@host/db", "PATH": "/usr/bin"}

    @staticmethod
    def _settings(agent_url: str | None) -> MagicMock:
        settings = MagicMock()
        settings.database.agent_url = agent_url
        return settings

    @pytest.mark.parametrize("agent_url", [None, ""])
    def test_unset_inherits_parent_dsn_and_warns(self, agent_url: str | None) -> None:
        with (
            patch.dict("os.environ", self.PARENT_ENV, clear=True),
            patch(
                "odds_lambda.jobs.agent_run.get_settings", return_value=self._settings(agent_url)
            ),
            patch("odds_lambda.jobs.agent_run.logger") as mock_logger,
        ):
            env = _build_agent_subprocess_env()

        assert env == self.PARENT_ENV
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][0] == "agent_database_url_not_set"

    def test_set_overrides_parent_dsn(self) -> None:
        agent_url = "postgresql://odds_agent:pw@host/db"
        with (
            patch.dict("os.environ", self.PARENT_ENV, clear=True),
            patch(
                "odds_lambda.jobs.agent_run.get_settings", return_value=self._settings(agent_url)
            ),
            patch("odds_lambda.jobs.agent_run.logger") as mock_logger,
        ):
            env = _build_agent_subprocess_env()

        assert env["DATABASE_URL"] == agent_url
        assert env["PATH"] == "/usr/bin"
        mock_logger.warning.assert_not_called()

    def test_reads_agent_url_from_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value set only in `.env` reaches the subprocess, not only an exported one."""
        (tmp_path / ".env").write_text("AGENT_DATABASE_URL=postgresql://odds_agent:pw@host/db\n")
        monkeypatch.chdir(tmp_path)
        reset_settings_cache()
        try:
            with (
                patch.dict("os.environ", self.PARENT_ENV, clear=True),
                patch("odds_lambda.jobs.agent_run.logger"),
            ):
                env = _build_agent_subprocess_env()
        finally:
            reset_settings_cache()

        assert env["DATABASE_URL"] == "postgresql://odds_agent:pw@host/db"

    def test_returned_env_is_a_copy(self) -> None:
        """Mutating the returned env dict must not affect ``os.environ``."""
        import os

        with (
            patch.dict("os.environ", {"EXISTING_KEY": "original"}, clear=True),
            patch("odds_lambda.jobs.agent_run.get_settings", return_value=self._settings("x")),
            patch("odds_lambda.jobs.agent_run.logger"),
        ):
            env = _build_agent_subprocess_env()
            env["EXISTING_KEY"] = "mutated"

            assert os.environ["EXISTING_KEY"] == "original"
            assert "DATABASE_URL" not in os.environ


class TestRunClaudeAgentCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_kills_the_subprocess(self, tmp_path) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from odds_lambda.jobs.agent_run import run_claude_agent

        class HangingStdout:
            def __aiter__(self) -> HangingStdout:
                return self

            async def __anext__(self) -> bytes:
                await asyncio.sleep(3600)
                raise StopAsyncIteration

        proc = MagicMock()
        proc.stdout = HangingStdout()
        proc.returncode = None
        settings = MagicMock(project_root=tmp_path)

        with (
            patch("odds_lambda.jobs.agent_run.get_settings", return_value=settings),
            patch(
                "odds_lambda.jobs.agent_run.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=proc),
            ),
        ):
            task = asyncio.create_task(run_claude_agent("soccer_epl"))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        proc.kill.assert_called_once()
