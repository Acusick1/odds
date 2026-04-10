"""Tests for fetch_oddsportal self-scheduling logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.jobs.fetch_oddsportal import (
    MAX_FAST_RETRIES,
    NORMAL_INTERVAL_HOURS,
    OVERNIGHT_RESUME_HOUR_UTC,
    RETRY_DELAY_MINUTES_MAX,
    RETRY_DELAY_MINUTES_MIN,
    _apply_overnight_skip,
    _calculate_next_execution,
    main,
)
from odds_lambda.scheduling.jobs import JobContext


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

    def test_overnight_skips_to_morning(self) -> None:
        next_time = datetime(2026, 4, 7, 23, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time)
        assert result.hour == OVERNIGHT_RESUME_HOUR_UTC
        assert result.minute == 0
        assert result > next_time

    def test_afternoon_no_skip(self) -> None:
        next_time = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time)
        assert result == next_time

    def test_early_morning_skips_to_morning(self) -> None:
        next_time = datetime(2026, 4, 7, 3, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time)
        assert result.hour == OVERNIGHT_RESUME_HOUR_UTC
        assert result.day == next_time.day

    def test_exactly_at_resume_hour_no_skip(self) -> None:
        next_time = datetime(2026, 4, 7, OVERNIGHT_RESUME_HOUR_UTC, 0, tzinfo=UTC)
        result = _apply_overnight_skip(next_time)
        assert result == next_time


class TestDefensivePreScheduling:
    """Verify defensive scheduling: pre-schedule at retry cadence, reschedule on success."""

    @staticmethod
    @asynccontextmanager
    async def _noop_alert_context(name: str) -> AsyncIterator[None]:
        yield

    @pytest.mark.asyncio
    async def test_preschedule_at_retry_cadence_before_scrape(self) -> None:
        """Pre-schedule must use retry cadence and fire before ingest_league."""
        call_order: list[str] = []

        async def fake_ingest(spec):
            call_order.append("ingest")
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            return IngestionStats(league="test", matches_scraped=5, snapshots_stored=3)

        async def fake_schedule(**kwargs):
            call_order.append("schedule")

        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"
        mock_backend.schedule_next_execution = AsyncMock(side_effect=fake_schedule)

        with (
            patch("odds_lambda.jobs.fetch_oddsportal.ingest_league", side_effect=fake_ingest),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext(retry_count=0))

        assert call_order[0] == "schedule", (
            "schedule_next_execution must be called before ingest_league"
        )
        assert "ingest" in call_order

    @pytest.mark.asyncio
    async def test_success_reschedules_at_normal_cadence(self) -> None:
        """On success, a second schedule call pushes to normal cadence."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league", new_callable=AsyncMock
            ) as mock_ingest,
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            await main(JobContext(retry_count=0))

        # Pre-schedule (retry cadence) + success reschedule (normal cadence)
        assert mock_backend.schedule_next_execution.call_count == 2
        # Last call should have no retry payload (success resets)
        last_call = mock_backend.schedule_next_execution.call_args
        assert last_call.kwargs.get("payload") is None

    @pytest.mark.asyncio
    async def test_failure_keeps_defensive_schedule(self) -> None:
        """On failure, no second schedule call — the defensive one stands."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league", new_callable=AsyncMock
            ) as mock_ingest,
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch("odds_core.alerts.send_job_warning", new_callable=AsyncMock),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(league="test", matches_scraped=0)

            await main(JobContext(retry_count=0))

        # Only the defensive pre-schedule fires
        assert mock_backend.schedule_next_execution.call_count == 1

    @pytest.mark.asyncio
    async def test_chain_survives_ingest_exception(self) -> None:
        """If ingest_league raises, the defensive schedule is already set."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name.return_value = "test"

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league",
                new_callable=AsyncMock,
                side_effect=Exception("simulated timeout"),
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch("odds_core.alerts.send_job_warning", new_callable=AsyncMock),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext(retry_count=0))

        # Only the defensive pre-schedule fires (no success reschedule)
        assert mock_backend.schedule_next_execution.call_count == 1


class TestRetryCountIntegration:
    """Verify retry_count flows correctly through main()."""

    @staticmethod
    @asynccontextmanager
    async def _noop_alert_context(name: str) -> AsyncIterator[None]:
        yield

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
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch("odds_core.alerts.send_job_warning", new_callable=AsyncMock),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(league="test", matches_scraped=0)

            await main(JobContext(retry_count=0))

            # Defensive pre-schedule has retry_count=1
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
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            await main(JobContext(retry_count=2))

            # Last call is the success reschedule with no retry payload
            assert mock_backend.schedule_next_execution.call_count == 2
            last_call = mock_backend.schedule_next_execution.call_args
            assert last_call.kwargs.get("payload") is None
