"""Tests for shared scheduling helpers."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.helpers import (
    apply_overnight_skip,
    book_next_run,
    get_next_kickoff,
    self_schedule,
)


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


class TestBookNextRun:
    @staticmethod
    def _decision(next_execution: datetime | None) -> ScheduleDecision:
        return ScheduleDecision(
            should_execute=True, reason="kickoff soon", next_execution=next_execution, tier=None
        )

    @pytest.mark.asyncio
    async def test_books_the_decision_with_its_label(self) -> None:
        next_time = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
        with patch(
            "odds_lambda.scheduling.helpers.self_schedule", new_callable=AsyncMock
        ) as schedule:
            booked = await book_next_run(
                self._decision(next_time),
                job_name="fetch-odds-epl",
                sport="soccer_epl",
                dry_run=False,
                label="pre-schedule",
                final=False,
            )

        assert booked is True
        schedule.assert_awaited_once_with(
            job_name="fetch-odds-epl",
            next_time=next_time,
            dry_run=False,
            sport="soccer_epl",
            reason="pre-schedule: kickoff soon",
        )

    @pytest.mark.asyncio
    async def test_no_next_execution_books_nothing(self) -> None:
        with patch(
            "odds_lambda.scheduling.helpers.self_schedule", new_callable=AsyncMock
        ) as schedule:
            booked = await book_next_run(
                self._decision(None),
                job_name="fetch-odds-epl",
                sport="soccer_epl",
                dry_run=False,
                label="pre-schedule",
                final=False,
            )

        assert booked is False
        schedule.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_alerts_and_reports_no_booking(self) -> None:
        with (
            patch(
                "odds_lambda.scheduling.helpers.self_schedule",
                side_effect=RuntimeError("no backend"),
            ),
            patch(
                "odds_lambda.scheduling.helpers.send_scheduling_failure", new_callable=AsyncMock
            ) as failure,
        ):
            booked = await book_next_run(
                self._decision(datetime(2026, 11, 1, 12, 0, tzinfo=UTC)),
                job_name="fetch-odds-epl",
                sport="soccer_epl",
                dry_run=False,
                label="post-fetch",
                final=True,
            )

        assert booked is False
        failure.assert_awaited_once()
        assert failure.call_args.args[0] == "fetch-odds-epl"
        assert failure.call_args.kwargs["final"] is True


class TestSmokeTags:
    """Tags distinguishing smoke-unsafe and outward-posting jobs."""

    def test_run_candidates_has_its_own_smoke_gate(self) -> None:
        from odds_lambda.scheduling.jobs import is_smoke_unsafe, places_candidate_trades

        assert places_candidate_trades("run-candidates-epl")
        assert not is_smoke_unsafe("run-candidates-epl")

    def test_agent_run_is_smoke_unsafe(self) -> None:
        from odds_lambda.scheduling.jobs import is_smoke_unsafe

        assert is_smoke_unsafe("agent-run")
        # Compound (sport-suffixed) names resolve to the same base.
        assert is_smoke_unsafe("agent-run-epl")

    def test_odds_api_jobs_spend_quota(self) -> None:
        from odds_lambda.scheduling.jobs import spends_api_quota

        assert spends_api_quota("fetch-odds")
        assert spends_api_quota("fetch-scores-epl")
        assert not spends_api_quota("fetch-oddsportal")

    def test_data_jobs_are_smoke_safe(self) -> None:
        from odds_lambda.scheduling.jobs import is_smoke_unsafe

        for job in ("fetch-oddsportal", "fetch-espn-fixtures", "fetch-betfair-exchange"):
            assert not is_smoke_unsafe(job)
