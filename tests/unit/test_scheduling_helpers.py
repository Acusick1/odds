"""Tests for shared scheduling helpers."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
