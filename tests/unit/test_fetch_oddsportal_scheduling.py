"""Tests for fetch_oddsportal proximity-based scheduling logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.jobs.fetch_oddsportal import (
    CLOSING_INTERVAL_HOURS,
    DB_FALLBACK_INTERVAL_HOURS,
    FAR_INTERVAL_HOURS,
    PREGAME_INTERVAL_HOURS,
    _interval_for_kickoff,
    main,
)
from odds_lambda.scheduling.jobs import JobContext


class TestIntervalForKickoff:
    """Tests for _interval_for_kickoff proximity logic."""

    def test_no_upcoming_games_returns_far_interval(self) -> None:
        assert _interval_for_kickoff(None) == FAR_INTERVAL_HOURS

    def test_game_within_closing_window(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now + timedelta(hours=1)
        assert _interval_for_kickoff(kickoff, now=now) == CLOSING_INTERVAL_HOURS

    def test_game_at_closing_boundary(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        # Exactly at 3h boundary — should be closing (< 3h is false, so pregame)
        kickoff = now + timedelta(hours=FetchTier.CLOSING.max_hours)
        assert _interval_for_kickoff(kickoff, now=now) == PREGAME_INTERVAL_HOURS

    def test_game_just_under_closing_threshold(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now + timedelta(hours=2.99)
        assert _interval_for_kickoff(kickoff, now=now) == CLOSING_INTERVAL_HOURS

    def test_game_in_pregame_window(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now + timedelta(hours=6)
        assert _interval_for_kickoff(kickoff, now=now) == PREGAME_INTERVAL_HOURS

    def test_game_at_pregame_boundary(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now + timedelta(hours=FetchTier.PREGAME.max_hours)
        assert _interval_for_kickoff(kickoff, now=now) == FAR_INTERVAL_HOURS

    def test_game_far_away(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now + timedelta(hours=24)
        assert _interval_for_kickoff(kickoff, now=now) == FAR_INTERVAL_HOURS

    def test_game_already_started(self) -> None:
        now = datetime(2026, 4, 7, 14, 0, tzinfo=UTC)
        kickoff = now - timedelta(hours=1)
        # Negative hours_until — should still return closing interval
        assert _interval_for_kickoff(kickoff, now=now) == CLOSING_INTERVAL_HOURS


class TestProximityScheduling:
    """Verify proximity-based scheduling in main()."""

    @staticmethod
    @asynccontextmanager
    async def _noop_alert_context(name: str) -> AsyncIterator[None]:
        yield

    @pytest.mark.asyncio
    async def test_preschedule_before_scrape(self) -> None:
        """Pre-schedule must fire before ingest_league."""
        call_order: list[str] = []

        async def fake_ingest(spec):
            call_order.append("ingest")
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            return IngestionStats(league="test", matches_scraped=5, snapshots_stored=3)

        async def fake_schedule(**kwargs):
            call_order.append("schedule")

        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")
        mock_backend.schedule_next_execution = AsyncMock(side_effect=fake_schedule)

        with (
            patch("odds_lambda.jobs.fetch_oddsportal.ingest_league", side_effect=fake_ingest),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext())

        assert call_order[0] == "schedule", (
            "schedule_next_execution must be called before ingest_league"
        )
        assert "ingest" in call_order

    @pytest.mark.asyncio
    async def test_success_reschedules_with_updated_interval(self) -> None:
        """On success, a second schedule call fires with re-queried interval."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            await main(JobContext())

        # Pre-schedule + success reschedule
        assert mock_backend.schedule_next_execution.call_count == 2
        # No retry_count in any payload
        for call in mock_backend.schedule_next_execution.call_args_list:
            payload = call.kwargs.get("payload")
            if payload:
                assert "retry_count" not in payload

    @pytest.mark.asyncio
    async def test_failure_keeps_prescheduled(self) -> None:
        """On failure, no second schedule call — the pre-scheduled one stands."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(league="test", matches_scraped=0)

            await main(JobContext())

        # Only the pre-schedule fires
        assert mock_backend.schedule_next_execution.call_count == 1

    @pytest.mark.asyncio
    async def test_chain_survives_ingest_exception(self) -> None:
        """If ingest_league raises, the pre-schedule is already set."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext())

        # Only the pre-schedule fires
        assert mock_backend.schedule_next_execution.call_count == 1

    @pytest.mark.asyncio
    async def test_db_failure_falls_back_to_1h(self) -> None:
        """If DB query fails, pre-schedule uses DB_FALLBACK_INTERVAL_HOURS."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                side_effect=Exception("DB connection refused"),
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal._apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext())

        # Pre-schedule should use fallback interval (~1h from now)
        first_call = mock_backend.schedule_next_execution.call_args_list[0]
        scheduled_time = first_call.kwargs["next_time"]
        delta_hours = (scheduled_time - now).total_seconds() / 3600
        assert 0.9 <= delta_hours <= DB_FALLBACK_INTERVAL_HOURS + 0.1

    @pytest.mark.asyncio
    async def test_closing_game_schedules_30min(self) -> None:
        """Game within 3h should produce a 30-min interval."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

        kickoff = datetime.now(UTC) + timedelta(hours=1)

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=kickoff,
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal._apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext())

        # Pre-schedule at ~30 min
        first_call = mock_backend.schedule_next_execution.call_args_list[0]
        scheduled_time = first_call.kwargs["next_time"]
        delta_minutes = (scheduled_time - now).total_seconds() / 60
        assert 25 <= delta_minutes <= 35

    @pytest.mark.asyncio
    async def test_far_game_schedules_2h(self) -> None:
        """Game 24h+ away should produce a 2h interval."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

        kickoff = datetime.now(UTC) + timedelta(hours=24)

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
            patch(
                "odds_lambda.jobs.fetch_oddsportal._get_next_kickoff",
                new_callable=AsyncMock,
                return_value=kickoff,
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal._apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext())

        # Pre-schedule at ~2h
        first_call = mock_backend.schedule_next_execution.call_args_list[0]
        scheduled_time = first_call.kwargs["next_time"]
        delta_hours = (scheduled_time - now).total_seconds() / 3600
        assert 1.9 <= delta_hours <= 2.1
