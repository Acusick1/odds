"""Tests for fetch_oddsportal self-scheduling logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.jobs.fetch_oddsportal import (
    GAME_LOOKAHEAD_HOURS,
    MAX_FAST_RETRIES,
    NORMAL_INTERVAL_HOURS,
    OVERNIGHT_RESUME_HOUR_UTC,
    RETRY_DELAY_MINUTES_MAX,
    RETRY_DELAY_MINUTES_MIN,
    _apply_overnight_skip,
    _calculate_next_execution,
    main,
)


class TestCalculateNextExecution:
    """Tests for _calculate_next_execution scheduling logic."""

    def test_success_schedules_normal_interval(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        next_time, retry_count = _calculate_next_execution(success=True, retry_count=0, now=now)
        assert next_time == now + timedelta(hours=NORMAL_INTERVAL_HOURS)
        assert retry_count == 0

    def test_success_resets_retry_count(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        _, retry_count = _calculate_next_execution(success=True, retry_count=2, now=now)
        assert retry_count == 0

    def test_first_failure_retries_quickly(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        next_time, retry_count = _calculate_next_execution(success=False, retry_count=0, now=now)
        delay_minutes = (next_time - now).total_seconds() / 60
        assert RETRY_DELAY_MINUTES_MIN <= delay_minutes <= RETRY_DELAY_MINUTES_MAX
        assert retry_count == 1

    def test_second_failure_retries_quickly(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        next_time, retry_count = _calculate_next_execution(success=False, retry_count=1, now=now)
        delay_minutes = (next_time - now).total_seconds() / 60
        assert RETRY_DELAY_MINUTES_MIN <= delay_minutes <= RETRY_DELAY_MINUTES_MAX
        assert retry_count == 2

    def test_exhausted_retries_returns_normal_interval(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        next_time, retry_count = _calculate_next_execution(
            success=False, retry_count=MAX_FAST_RETRIES, now=now
        )
        assert next_time == now + timedelta(hours=NORMAL_INTERVAL_HOURS)
        assert retry_count == 0

    def test_exhausted_retries_above_max(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        next_time, retry_count = _calculate_next_execution(
            success=False, retry_count=MAX_FAST_RETRIES + 5, now=now
        )
        assert next_time == now + timedelta(hours=NORMAL_INTERVAL_HOURS)
        assert retry_count == 0


class TestApplyOvernightSkip:
    """Tests for _apply_overnight_skip logic."""

    def test_overnight_no_games_skips_to_morning(self) -> None:
        next_time = datetime(2026, 4, 7, 23, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time, next_game_time=None)
        assert result.hour == OVERNIGHT_RESUME_HOUR_UTC
        assert result.minute == 0
        assert result > next_time

    def test_overnight_with_imminent_game_no_skip(self) -> None:
        next_time = datetime(2026, 4, 7, 23, 0, tzinfo=UTC)
        game_time = next_time + timedelta(hours=2)
        result = _apply_overnight_skip(next_time, next_game_time=game_time)
        assert result == next_time

    def test_afternoon_no_skip(self) -> None:
        next_time = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time, next_game_time=None)
        assert result == next_time

    def test_early_morning_no_games_skips_to_morning(self) -> None:
        next_time = datetime(2026, 4, 7, 3, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time, next_game_time=None)
        assert result.hour == OVERNIGHT_RESUME_HOUR_UTC
        assert result.day == next_time.day

    def test_overnight_distant_game_skips(self) -> None:
        next_time = datetime(2026, 4, 7, 23, 0, tzinfo=UTC)
        game_time = next_time + timedelta(hours=GAME_LOOKAHEAD_HOURS + 1)
        result = _apply_overnight_skip(next_time, next_game_time=game_time)
        assert result.hour == OVERNIGHT_RESUME_HOUR_UTC
        assert result > next_time


class TestRetryCountIntegration:
    """Verify retry_count flows correctly through main()."""

    @pytest.mark.asyncio
    async def test_failure_increments_retry_count(self) -> None:
        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league",
                new_callable=AsyncMock,
            ) as mock_ingest,
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_game_time",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(league="test", matches_scraped=0)

            await main(retry_count=0)

            call_kwargs = mock_backend.schedule_next_execution.call_args
            assert call_kwargs.kwargs["payload"] == {"retry_count": 1}

    @pytest.mark.asyncio
    async def test_success_resets_retry_count(self) -> None:
        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league",
                new_callable=AsyncMock,
            ) as mock_ingest,
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_game_time",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            await main(retry_count=2)

            call_kwargs = mock_backend.schedule_next_execution.call_args
            # retry_count=0 means no payload (None)
            assert call_kwargs.kwargs.get("payload") is None
