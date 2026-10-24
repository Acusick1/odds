"""Tests for fetch_oddsportal proximity-based scheduling logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.jobs.fetch_oddsportal import CADENCE, main
from odds_lambda.scheduling.decision import DORMANCY_HORIZON_DAYS
from odds_lambda.scheduling.jobs import JobContext


class TestCadenceBands:
    """The OddsPortal cadence preserves the previous browser-scrape bands."""

    def test_closing_band(self) -> None:
        assert CADENCE.interval_for(FetchTier.CLOSING) == 0.5

    def test_pregame_band(self) -> None:
        assert CADENCE.interval_for(FetchTier.PREGAME) == 1.0

    def test_far_bands_all_2h(self) -> None:
        assert CADENCE.interval_for(FetchTier.SHARP) == 2.0
        assert CADENCE.interval_for(FetchTier.EARLY) == 2.0
        assert CADENCE.interval_for(FetchTier.OPENING) == 2.0

    def test_no_game_band(self) -> None:
        assert CADENCE.no_game == 6.0

    def test_db_fallback(self) -> None:
        assert CADENCE.db_fallback == 1.0


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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(hours=20),
            ),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext(sport="soccer_epl"))

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(hours=20),
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            await main(JobContext(sport="soccer_epl"))

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.require_rows_for_fixtures",
                new_callable=AsyncMock,
            ),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(hours=20),
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(league="test", matches_scraped=0)

            await main(JobContext(sport="soccer_epl"))

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(hours=20),
            ),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            with pytest.raises(RuntimeError, match="simulated timeout"):
                await main(JobContext(sport="soccer_epl"))

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                side_effect=Exception("DB connection refused"),
            ),
            patch(
                "odds_lambda.scheduling.decision.apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext(sport="soccer_epl"))

        # Pre-schedule should use fallback interval (~1h from now)
        first_call = mock_backend.schedule_next_execution.call_args_list[0]
        scheduled_time = first_call.kwargs["next_time"]
        delta_hours = (scheduled_time - now).total_seconds() / 3600
        assert 0.9 <= delta_hours <= CADENCE.db_fallback + 0.1

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=kickoff,
            ),
            patch(
                "odds_lambda.scheduling.decision.apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext(sport="soccer_epl"))

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
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=self._noop_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=kickoff,
            ),
            patch(
                "odds_lambda.scheduling.decision.apply_overnight_skip",
                side_effect=lambda t, **kw: t,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_settings.return_value.scheduler.dry_run = False
            mock_ingest.return_value = IngestionStats(
                league="test", matches_scraped=5, snapshots_stored=3
            )

            now = datetime.now(UTC)
            await main(JobContext(sport="soccer_epl"))

        # Pre-schedule at ~2h
        first_call = mock_backend.schedule_next_execution.call_args_list[0]
        scheduled_time = first_call.kwargs["next_time"]
        delta_hours = (scheduled_time - now).total_seconds() / 3600
        assert 1.9 <= delta_hours <= 2.1


class TestDormancy:
    """With no fixture in sight the job only books its next check-in."""

    @staticmethod
    async def _run(next_kickoff: datetime | None) -> tuple[AsyncMock, AsyncMock, list[str]]:
        entered: list[str] = []

        @asynccontextmanager
        async def alert_context(name: str) -> AsyncIterator[None]:
            entered.append(name)
            yield

        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")
        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league", new_callable=AsyncMock
            ) as mock_ingest,
            patch(
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=next_kickoff,
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.require_rows_for_fixtures",
                new_callable=AsyncMock,
            ),
        ):
            from odds_lambda.jobs.fetch_oddsportal import IngestionStats

            mock_ingest.return_value = IngestionStats(league="mlb")
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext(sport="baseball_mlb"))
        return mock_ingest, mock_backend, entered

    @pytest.mark.asyncio
    async def test_no_fixture_skips_the_scrape_and_checks_in_later(self) -> None:
        now = datetime.now(UTC)

        ingest, backend, _ = await self._run(None)

        ingest.assert_not_called()
        backend.schedule_next_execution.assert_awaited_once()
        booked = backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert (booked - now).total_seconds() / 3600 == pytest.approx(CADENCE.no_game, abs=0.1)

    @pytest.mark.asyncio
    async def test_a_dormant_run_still_writes_a_heartbeat(self) -> None:
        _, _, entered = await self._run(None)

        assert entered == ["fetch-oddsportal-mlb"]

    @pytest.mark.asyncio
    async def test_a_fixture_beyond_the_horizon_wakes_the_chain_when_it_enters(self) -> None:
        kickoff = datetime.now(UTC) + timedelta(days=DORMANCY_HORIZON_DAYS, hours=2)

        ingest, backend, _ = await self._run(kickoff)

        ingest.assert_not_called()
        booked = backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert booked == kickoff - timedelta(days=DORMANCY_HORIZON_DAYS)

    @pytest.mark.asyncio
    async def test_a_fixture_inside_the_horizon_scrapes(self) -> None:
        kickoff = datetime.now(UTC) + timedelta(days=DORMANCY_HORIZON_DAYS - 1)

        ingest, _, _ = await self._run(kickoff)

        ingest.assert_awaited_once()


class TestDroppedGamesAlert:
    @pytest.mark.asyncio
    async def test_mlb_run_alerts_dropped_games_and_still_completes(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import IngestionStats
        from odds_lambda.live_events import DroppedGame, SkipReason
        from odds_lambda.mlb_game_resolver import GameRow
        from odds_lambda.scheduling.jobs import make_compound_job_name

        completed: list[str] = []

        @asynccontextmanager
        async def recording_alert_context(name: str) -> AsyncIterator[None]:
            yield
            completed.append(name)

        dropped = DroppedGame(
            GameRow("Chicago Cubs", "St. Louis Cardinals", datetime.now(UTC) + timedelta(hours=5)),
            SkipReason.UNRESOLVED,
            "ambiguous_doubleheader",
        )
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")
        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal.ingest_league",
                new_callable=AsyncMock,
                return_value=IngestionStats(
                    league="mlb", matches_scraped=2, snapshots_stored=1, dropped_games=[dropped]
                ),
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal.alert_dropped_games", new_callable=AsyncMock
            ) as alert,
            patch(
                "odds_lambda.jobs.fetch_oddsportal.require_rows_for_fixtures",
                new_callable=AsyncMock,
            ),
            patch(
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_core.config.get_settings") as mock_settings,
            patch("odds_core.alerts.job_alert_context", side_effect=recording_alert_context),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(hours=5),
            ),
        ):
            mock_settings.return_value.scheduler.dry_run = False
            await main(JobContext(sport="baseball_mlb"))

        job_name = make_compound_job_name("fetch-oddsportal", "baseball_mlb")
        alert.assert_awaited_once_with(job_name, [dropped])
        assert completed == [job_name]
