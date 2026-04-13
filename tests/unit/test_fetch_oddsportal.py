"""Tests for fetch_oddsportal job utilities."""

from __future__ import annotations

from datetime import UTC, datetime

from odds_lambda.jobs.fetch_oddsportal import _apply_overnight_skip


class TestApplyOvernightSkip:
    def test_epl_defaults_during_day(self) -> None:
        dt = datetime(2026, 4, 13, 14, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == dt

    def test_epl_defaults_late_night(self) -> None:
        dt = datetime(2026, 4, 13, 23, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 14, 6, 0, tzinfo=UTC)

    def test_epl_defaults_early_morning(self) -> None:
        dt = datetime(2026, 4, 13, 3, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 13, 6, 0, tzinfo=UTC)

    def test_epl_defaults_boundary_start(self) -> None:
        dt = datetime(2026, 4, 13, 22, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 14, 6, 0, tzinfo=UTC)

    def test_epl_defaults_boundary_resume(self) -> None:
        dt = datetime(2026, 4, 13, 6, 0, tzinfo=UTC)
        assert _apply_overnight_skip(dt) == dt

    def test_mlb_hours_during_games(self) -> None:
        dt = datetime(2026, 6, 15, 23, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_mlb_hours_overnight(self) -> None:
        dt = datetime(2026, 6, 16, 7, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_mlb_hours_boundary_start(self) -> None:
        dt = datetime(2026, 6, 16, 5, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_mlb_hours_boundary_resume(self) -> None:
        dt = datetime(2026, 6, 16, 14, 0, tzinfo=UTC)
        result = _apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_mlb_hours_before_overnight(self) -> None:
        dt = datetime(2026, 6, 16, 4, 30, tzinfo=UTC)
        result = _apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt
