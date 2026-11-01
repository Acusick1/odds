"""Unit tests for health monitoring system."""

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, APIConfig, SchedulerConfig, Settings
from odds_core.vocabulary import Market
from odds_lambda.health_monitor import (
    HealthMetrics,
    HealthMonitor,
    HealthStatus,
    QuotaReading,
    StaleSource,
)
from odds_lambda.scheduling.backends.base import JobStatus, ScheduledJob
from odds_lambda.storage.readers import KeyQuota


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        alerts=AlertConfig(
            alert_enabled=True,
            stale_reminder_minutes=360,
            quota_warning_threshold=0.2,
            quota_critical_threshold=0.1,
            stale_data_hours=2,
        ),
        api=APIConfig(key="test-key"),
    )


@pytest.fixture
async def mock_session():
    """Create mock database session."""
    session = AsyncMock()
    # In SQLAlchemy async sessions, add() is synchronous but commit() is async
    session.add = MagicMock()  # Synchronous method
    session.commit = AsyncMock()  # Async method
    return session


class TestHealthMonitor:
    """Test HealthMonitor class."""

    @pytest.mark.asyncio
    async def test_send_alert_respects_disabled_flag(self, mock_session, mock_settings):
        """Should not send alert when alerts are disabled."""
        mock_settings.alerts.alert_enabled = False
        monitor = HealthMonitor(mock_session, mock_settings)

        result = await monitor._send_alert(
            alert_type="test_alert", severity="warning", message="Test"
        )

        assert result is False
        # Should not check rate limiting or send
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alert_suppressed_in_scheduler_dry_run(self, mock_session, mock_settings):
        """A dry-run process neither posts nor records the alert (no rate-limit row)."""
        mock_settings.scheduler.dry_run = True
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as send:
            result = await monitor._send_alert(
                alert_type="test_alert", severity="critical", message="Test"
            )

        assert result is False
        send.assert_not_called()
        mock_session.execute.assert_not_called()
        mock_session.add.assert_not_called()

    async def test_check_api_quota_healthy(self, mock_session, mock_settings):
        """Should return healthy when quota is above warning threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_latest_quota_by_key.return_value = {
                0: KeyQuota(("soccer_epl",), 125, 375),  # 25% of 500
            }
            mock_reader_class.return_value = mock_reader

            readings = await monitor.check_api_quota()

            assert readings == [QuotaReading(0, ("soccer_epl",), 125, 500, 0.25)]

    @pytest.mark.asyncio
    async def test_check_api_quota_low_warning(self, mock_session, mock_settings):
        """Should return unhealthy when quota is below warning threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_latest_quota_by_key.return_value = {
                1: KeyQuota(("basketball_nba",), 450, 50),
                0: KeyQuota(("soccer_epl",), 75, 425),  # 15% of 500 (below 20% warning)
            }
            mock_reader_class.return_value = mock_reader

            readings = await monitor.check_api_quota()

            assert readings == [
                QuotaReading(0, ("soccer_epl",), 75, 500, 0.15),
                QuotaReading(1, ("basketball_nba",), 450, 500, 0.9),
            ]

    @pytest.mark.asyncio
    async def test_check_api_quota_uses_the_keys_plan_not_a_setting(
        self, mock_session: AsyncMock, mock_settings: Settings
    ) -> None:
        """900 left of a 20000-credit plan is 4.5%, well under the 10% critical threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_latest_quota_by_key.return_value = {
                0: KeyQuota(("soccer_epl",), 900, 19100),
            }
            mock_reader_class.return_value = mock_reader

            readings = await monitor.check_api_quota()

        assert readings == [QuotaReading(0, ("soccer_epl",), 900, 20000, 0.045)]
        assert readings[0].fraction < mock_settings.alerts.quota_critical_threshold

    @pytest.mark.asyncio
    async def test_collect_metrics_returns_full_metrics(self, mock_session, mock_settings):
        """Should collect all health metrics."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_database_stats.return_value = {
                "api_quota_by_key": {0: KeyQuota(("soccer_epl",), 15000, 5000)},
                "events_by_status": {
                    "scheduled": 10,
                    "live": 2,
                    "final": 100,
                },
            }
            mock_reader_class.return_value = mock_reader

            # Mock shared snapshot query and check methods
            with patch.object(monitor, "_get_hours_since_latest_snapshot", return_value=0.5):
                with patch.object(monitor, "check_data_quality", return_value=(True, 2)):
                    metrics = await monitor.collect_metrics()

            assert isinstance(metrics, HealthMetrics)
            assert metrics.api_quota_by_key == {0: 15000}
            assert metrics.hours_since_last_fetch is not None
            assert metrics.data_quality_errors_24h == 2
            assert metrics.events_scheduled == 10
            assert metrics.events_live == 2
            assert metrics.events_final == 100

    @pytest.mark.asyncio
    async def test_check_system_health_all_healthy(self, mock_session, mock_settings):
        """Should return overall healthy when all checks pass."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock all checks to return healthy
        with patch.object(monitor, "collect_metrics") as mock_collect:
            mock_collect.return_value = HealthMetrics(
                hours_since_last_fetch=0.5,
                api_quota_by_key={0: 15000},
                data_quality_errors_24h=2,
                events_scheduled=5,
                events_live=0,
                events_final=50,
            )

            with (
                patch.object(monitor, "check_stale_sources", return_value=[]),
                patch.object(monitor, "check_missing_betfair_closes", return_value=[]),
            ):
                with patch.object(
                    monitor,
                    "check_api_quota",
                    return_value=[QuotaReading(0, ("soccer_epl",), 15000, 20000, 0.75)],
                ):
                    with patch.object(monitor, "check_data_quality", return_value=(True, 2)):
                        with (
                            patch.object(monitor, "check_job_heartbeats", return_value=[]),
                            patch.object(monitor, "check_event_identity", return_value=[]),
                            patch.object(monitor, "check_unsettled_events", return_value=[]),
                        ):
                            status = await monitor.check_system_health()

            assert isinstance(status, HealthStatus)
            assert status.overall_healthy is True
            assert len(status.issues_detected) == 0
            assert len(status.alerts_sent) == 0

    @pytest.mark.asyncio
    async def test_check_system_health_sends_alerts(self, mock_session, mock_settings):
        """Should send alerts when issues detected."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock metrics
        with patch.object(monitor, "collect_metrics") as mock_collect:
            mock_collect.return_value = HealthMetrics(
                hours_since_last_fetch=3.0,
                api_quota_by_key={0: 1000},
                data_quality_errors_24h=15,
                events_scheduled=0,
                events_live=0,
                events_final=50,
            )

            # Mock checks to return unhealthy
            with (
                patch.object(monitor, "check_missing_betfair_closes", return_value=[]),
                patch.object(
                    monitor,
                    "check_stale_sources",
                    return_value=[
                        StaleSource(
                            source="oddsportal",
                            sport_key="soccer_epl",
                            market=Market.ONE_X_TWO,
                            last_snapshot=datetime.now(UTC) - timedelta(hours=9),
                            active_hours=9.0,
                            threshold_hours=3.0,
                        )
                    ],
                ),
            ):
                with patch.object(
                    monitor,
                    "check_api_quota",
                    return_value=[QuotaReading(0, ("soccer_epl",), 1000, 20000, 0.05)],
                ):
                    with patch.object(monitor, "check_data_quality", return_value=(False, 15)):
                        with (
                            patch.object(monitor, "check_job_heartbeats", return_value=[]),
                            patch.object(monitor, "check_event_identity", return_value=[]),
                            patch.object(monitor, "check_unsettled_events", return_value=[]),
                        ):
                            # Mock alert sending
                            with patch.object(monitor, "_send_alert", return_value=True):
                                status = await monitor.check_system_health()

            assert status.overall_healthy is False
            assert len(status.issues_detected) > 0
            # Should have attempted to send alerts
            assert "oddsportal has written no soccer_epl 1x2 snapshot" in status.issues_detected[0]

    @pytest.mark.asyncio
    async def test_check_system_health_handles_errors_gracefully(self, mock_session, mock_settings):
        """Should handle errors gracefully and return unhealthy status."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock collect_metrics to raise exception
        with patch.object(
            monitor, "collect_metrics", side_effect=Exception("Database connection error")
        ):
            # Mock _send_alert to not raise
            with patch.object(monitor, "_send_alert", return_value=False):
                status = await monitor.check_system_health()

        assert status.overall_healthy is False
        assert len(status.issues_detected) == 1
        assert "Database connection error" in status.issues_detected[0]

    @pytest.mark.asyncio
    async def test_check_job_heartbeats_healthy(self, mock_session, mock_settings):
        """Should return empty list when all heartbeats are recent."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock query: all jobs have recent heartbeats
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = datetime.now(UTC) - timedelta(minutes=30)
        mock_session.execute.return_value = mock_result

        with patch.object(monitor, "_send_alert", return_value=False):
            issues = await monitor.check_job_heartbeats()

        assert issues == []

    @pytest.mark.asyncio
    async def test_check_job_heartbeats_no_history_skipped(self, mock_session, mock_settings):
        """Should skip alerting when no heartbeat history exists (first deploy)."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock query: no heartbeat found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch.object(monitor, "_send_alert", return_value=True) as mock_alert:
            issues = await monitor.check_job_heartbeats()

        assert issues == []
        mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_job_heartbeats_stale(self, mock_session, mock_settings):
        """Should flag jobs with old heartbeats."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock query: heartbeat from 60 hours ago (exceeds every expectation)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = datetime.now(UTC) - timedelta(hours=60)
        mock_session.execute.return_value = mock_result

        with patch.object(monitor, "_send_alert", return_value=True):
            issues = await monitor.check_job_heartbeats()

        assert len(issues) == len(mock_settings.alerts.heartbeat_expectations)


class FakeSchedules:
    def __init__(self, next_runs: dict[str, datetime | None], outstanding: set[str]) -> None:
        self.next_runs = next_runs
        self.outstanding = outstanding

    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        return [
            ScheduledJob(job_name=name, next_run_time=run, status=JobStatus.SCHEDULED)
            for name, run in self.next_runs.items()
        ]

    async def get_outstanding_job_names(self) -> set[str]:
        return self.outstanding


class TestCheckScheduleChains:
    NOW = datetime(2026, 10, 17, 13, 5, tzinfo=UTC)

    @pytest.fixture
    def chain_settings(self, mock_settings):
        mock_settings.scheduler = SchedulerConfig(
            bootstrap_jobs=["fetch-betfair-exchange", "fetch-oddsportal", "fetch-espn-fixtures"]
        )
        mock_settings.data_collection = SimpleNamespace(sports=["soccer_epl"])
        mock_settings.betfair = SimpleNamespace(enabled=True)
        mock_settings.polymarket = SimpleNamespace(enabled=False)
        return mock_settings

    async def _dead(
        self,
        session,
        settings,
        next_runs: dict[str, datetime | None],
        outstanding: set[str] = frozenset(),
    ) -> list[str]:
        monitor = HealthMonitor(session, settings, FakeSchedules(next_runs, set(outstanding)))
        return [chain.job_name for chain in await monitor.check_schedule_chains(now=self.NOW)]

    @pytest.mark.asyncio
    async def test_booked_runs_are_alive(self, mock_session, chain_settings):
        soon = self.NOW + timedelta(minutes=10)
        dead = await self._dead(
            mock_session,
            chain_settings,
            {"fetch-betfair-exchange-epl": soon, "fetch-oddsportal-epl": soon},
        )
        assert dead == []

    @pytest.mark.asyncio
    async def test_missing_schedule_is_dead(self, mock_session, chain_settings):
        dead = await self._dead(
            mock_session,
            chain_settings,
            {"fetch-oddsportal-epl": self.NOW + timedelta(minutes=10)},
        )
        assert dead == ["fetch-betfair-exchange-epl"]

    @pytest.mark.asyncio
    async def test_overdue_run_is_dead(self, mock_session, chain_settings):
        soon = self.NOW + timedelta(minutes=10)
        dead = await self._dead(
            mock_session,
            chain_settings,
            {
                "fetch-betfair-exchange-epl": self.NOW - timedelta(minutes=16),
                "fetch-oddsportal-epl": soon,
            },
        )
        assert dead == ["fetch-betfair-exchange-epl"]

    @pytest.mark.asyncio
    async def test_running_job_without_schedule_is_alive(self, mock_session, chain_settings):
        dead = await self._dead(
            mock_session,
            chain_settings,
            {"fetch-oddsportal-epl": self.NOW + timedelta(minutes=10)},
            outstanding={"fetch-betfair-exchange-epl"},
        )
        assert dead == []

    @pytest.mark.asyncio
    async def test_cron_jobs_and_disabled_jobs_are_not_checked(self, mock_session, chain_settings):
        chain_settings.betfair = SimpleNamespace(enabled=False)
        dead = await self._dead(
            mock_session,
            chain_settings,
            {"fetch-oddsportal-epl": self.NOW + timedelta(minutes=10)},
        )
        assert dead == []

    @pytest.mark.asyncio
    async def test_without_schedule_reader_check_is_skipped(self, mock_session, chain_settings):
        monitor = HealthMonitor(mock_session, chain_settings)
        assert await monitor.check_schedule_chains(now=self.NOW) == []

    @pytest.mark.asyncio
    async def test_dead_chain_sends_critical_alert(self, mock_session, chain_settings):
        monitor = HealthMonitor(
            mock_session,
            chain_settings,
            FakeSchedules({"fetch-oddsportal-epl": datetime.now(UTC) + timedelta(hours=1)}, set()),
        )
        healthy = [
            patch.object(monitor, name, return_value=value)
            for name, value in (
                ("check_stale_sources", []),
                ("check_missing_betfair_closes", []),
                ("check_api_quota", []),
                ("check_data_quality", (True, 0)),
                ("check_job_heartbeats", []),
                ("check_event_identity", []),
                ("check_unsettled_events", []),
                ("purge_old_heartbeats", 0),
                (
                    "collect_metrics",
                    HealthMetrics(
                        hours_since_last_fetch=0.1,
                        api_quota_by_key={},
                        data_quality_errors_24h=0,
                        events_scheduled=1,
                        events_live=0,
                        events_final=0,
                    ),
                ),
            )
        ]
        for p in healthy:
            p.start()
        try:
            with patch.object(monitor, "_send_alert", return_value=True) as send:
                status = await monitor.check_system_health()
        finally:
            for p in healthy:
                p.stop()

        assert status.alerts_sent == ["dead_schedule_chain:fetch-betfair-exchange-epl"]
        assert send.call_args.kwargs["severity"] == "critical"


class TestQuotaAlerts:
    @pytest.mark.asyncio
    async def test_a_low_key_alerts_once_while_the_other_key_stays_quiet(
        self, mock_session, mock_settings
    ):
        monitor = HealthMonitor(mock_session, mock_settings)
        readings = [
            QuotaReading(0, ("soccer_epl", "baseball_mlb"), 30, 500, 0.06),
            QuotaReading(1, ("basketball_nba",), 450, 500, 0.9),
        ]
        quiet = [
            patch.object(monitor, name, return_value=value)
            for name, value in (
                ("check_stale_sources", []),
                ("check_missing_betfair_closes", []),
                ("check_api_quota", readings),
                ("check_data_quality", (True, 0)),
                ("check_job_heartbeats", []),
                ("check_event_identity", []),
                ("check_unsettled_events", []),
                ("purge_old_heartbeats", 0),
                (
                    "collect_metrics",
                    HealthMetrics(
                        hours_since_last_fetch=0.1,
                        api_quota_by_key={0: 30, 1: 450},
                        data_quality_errors_24h=0,
                        events_scheduled=1,
                        events_live=0,
                        events_final=0,
                    ),
                ),
            )
        ]
        with ExitStack() as stack:
            for p in quiet:
                stack.enter_context(p)
            send = stack.enter_context(patch.object(monitor, "_send_alert", return_value=True))
            await monitor.check_system_health()

        quota_alerts = [
            call.kwargs["alert_type"]
            for call in send.call_args_list
            if call.kwargs["alert_type"].startswith("quota_")
        ]
        assert quota_alerts == ["quota_critical:key0"]
