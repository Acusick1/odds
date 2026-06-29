"""Tests for shared scheduling helpers."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.helpers import apply_overnight_skip, get_next_kickoff, self_schedule


class TestApplyOvernightSkip:
    """Tests for the overnight skip helper (wrap-midnight, same-day, no-op)."""

    def test_daytime_unchanged(self) -> None:
        dt = datetime(2026, 4, 15, 14, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == dt

    def test_late_night_pushed_to_next_morning(self) -> None:
        dt = datetime(2026, 4, 15, 23, 30, tzinfo=UTC)
        expected = datetime(2026, 4, 16, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == expected

    def test_early_morning_pushed_to_same_day_resume(self) -> None:
        dt = datetime(2026, 4, 15, 3, 0, tzinfo=UTC)
        expected = datetime(2026, 4, 15, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == expected

    def test_exactly_at_overnight_start_pushed(self) -> None:
        dt = datetime(2026, 4, 15, 22, 0, tzinfo=UTC)
        expected = datetime(2026, 4, 16, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == expected

    def test_exactly_at_resume_not_pushed(self) -> None:
        dt = datetime(2026, 4, 15, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == dt

    def test_wrap_midnight_window_defaults(self) -> None:
        dt = datetime(2026, 4, 15, 0, 30, tzinfo=UTC)
        expected = datetime(2026, 4, 15, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == expected

    def test_same_day_window_inside(self) -> None:
        dt = datetime(2026, 6, 16, 7, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_same_day_window_outside_before(self) -> None:
        dt = datetime(2026, 6, 16, 4, 30, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_same_day_window_outside_after(self) -> None:
        dt = datetime(2026, 6, 16, 23, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_same_day_window_boundary_start(self) -> None:
        dt = datetime(2026, 6, 16, 5, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_same_day_window_boundary_resume(self) -> None:
        dt = datetime(2026, 6, 16, 14, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt


class TestGetNextKickoff:
    """Tests for the get_next_kickoff function signature and contract."""

    def test_is_async(self) -> None:
        assert inspect.iscoroutinefunction(get_next_kickoff)

    def test_accepts_sport_key_param(self) -> None:
        sig = inspect.signature(get_next_kickoff)
        assert "sport_key" in sig.parameters


class TestSelfSchedule:
    """Tests for the self_schedule function."""

    def test_is_async(self) -> None:
        assert inspect.iscoroutinefunction(self_schedule)

    def test_signature_has_required_params(self) -> None:
        sig = inspect.signature(self_schedule)
        params = sig.parameters
        assert "job_name" in params
        assert "next_time" in params
        assert "dry_run" in params
        assert "sport" in params
        assert "interval_hours" in params
        assert "reason" in params

    @pytest.mark.asyncio
    async def test_calls_scheduler_backend(self) -> None:
        mock_backend = MagicMock()
        mock_backend.schedule_next_execution = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with patch(
            "odds_lambda.scheduling.helpers.get_scheduler_backend",
            return_value=mock_backend,
        ):
            await self_schedule(
                job_name="test-job",
                next_time=datetime.now(UTC) + timedelta(hours=1),
                dry_run=True,
                sport="soccer_epl",
                reason="test reason",
            )

        mock_backend.schedule_next_execution.assert_called_once()
        call_kwargs = mock_backend.schedule_next_execution.call_args.kwargs
        assert call_kwargs["job_name"] == "test-job"
        assert call_kwargs["payload"] == {"sport": "soccer_epl"}

    @pytest.mark.asyncio
    async def test_no_sport_sends_none_payload(self) -> None:
        mock_backend = MagicMock()
        mock_backend.schedule_next_execution = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with patch(
            "odds_lambda.scheduling.helpers.get_scheduler_backend",
            return_value=mock_backend,
        ):
            await self_schedule(
                job_name="test-job",
                next_time=datetime.now(UTC) + timedelta(hours=1),
                dry_run=False,
            )

        call_kwargs = mock_backend.schedule_next_execution.call_args.kwargs
        assert call_kwargs["payload"] is None


class TestSuppressScheduling:
    """The ``suppress_scheduling`` scope forces every backend into dry-run.

    This is the single chokepoint that guarantees no rows are written to the
    schedule store during a smoke run, regardless of which job self-schedules.
    """

    def test_default_allows_scheduling(self) -> None:
        from odds_lambda.scheduling.backends import scheduling_enabled

        assert scheduling_enabled() is True

    def test_scope_disables_and_restores(self) -> None:
        from odds_lambda.scheduling.backends import scheduling_enabled, suppress_scheduling

        with suppress_scheduling():
            assert scheduling_enabled() is False
        assert scheduling_enabled() is True

    def test_backend_forced_dry_run_under_scope(self) -> None:
        from odds_lambda.scheduling.backends import get_scheduler_backend, suppress_scheduling

        # Even with an explicit dry_run=False, the scope forces dry-run.
        with suppress_scheduling():
            backend = get_scheduler_backend(backend_type="local", dry_run=False)
            assert backend.dry_run is True

    def test_backend_live_outside_scope(self) -> None:
        from odds_lambda.scheduling.backends import get_scheduler_backend

        backend = get_scheduler_backend(backend_type="local", dry_run=False)
        assert backend.dry_run is False


class TestGateBypass:
    """Under ``SMOKE_POLICY`` (respect_gate=False) the body runs even when the
    cadence decision says ``should_execute=False``.

    This is the property that makes ``scheduler smoke`` exercise the real
    fetch+ingest body of gated jobs that are not "due" at deploy time.
    """

    @staticmethod
    def _not_due_decision() -> ScheduleDecision:
        return ScheduleDecision(
            should_execute=False,
            reason="no upcoming game",
            next_execution=datetime.now(UTC) + timedelta(hours=6),
            tier=None,
        )

    @pytest.mark.asyncio
    async def test_smoke_policy_runs_body_when_not_due(self) -> None:
        from odds_lambda.jobs import fetch_scores
        from odds_lambda.scheduling.jobs import SMOKE_POLICY, JobContext

        body = AsyncMock()
        with (
            patch.object(fetch_scores, "_scores_decision", return_value=self._not_due_decision()),
            patch.object(fetch_scores, "self_schedule", new=AsyncMock()),
            patch.object(fetch_scores, "_fetch_and_update_scores", new=body),
        ):
            await fetch_scores.main(JobContext(sport="soccer_epl", policy=SMOKE_POLICY))

        body.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_live_policy_skips_body_when_not_due(self) -> None:
        from odds_lambda.jobs import fetch_scores
        from odds_lambda.scheduling.jobs import JobContext

        body = AsyncMock()
        with (
            patch.object(fetch_scores, "_scores_decision", return_value=self._not_due_decision()),
            patch.object(fetch_scores, "self_schedule", new=AsyncMock()),
            patch.object(fetch_scores, "_fetch_and_update_scores", new=body),
        ):
            # Default JobContext policy is live (respect_gate=True).
            await fetch_scores.main(JobContext(sport="soccer_epl"))

        body.assert_not_awaited()


class TestSmokeTags:
    """Tag distinguishing the cost-excluded (expensive) jobs from the rest.

    There is no longer an outward-posting tag: outward side effects are
    suppressed universally at their delivery sinks during a smoke run, so no
    per-job registry can be "forgotten".
    """

    def test_agent_run_is_smoke_expensive(self) -> None:
        from odds_lambda.scheduling.jobs import is_smoke_expensive

        assert is_smoke_expensive("agent-run")
        # Compound (sport-suffixed) names resolve to the same base.
        assert is_smoke_expensive("agent-run-epl")

    def test_data_jobs_are_not_expensive(self) -> None:
        from odds_lambda.scheduling.jobs import is_smoke_expensive

        for job in ("fetch-oddsportal", "fetch-espn-fixtures", "fetch-betfair-exchange"):
            assert not is_smoke_expensive(job)
