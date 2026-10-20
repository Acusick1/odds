"""Unit tests for health monitoring system."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, APIConfig, SchedulerConfig, Settings
from odds_core.models import AlertHistory
from odds_lambda.health_monitor import HealthMetrics, HealthMonitor, HealthStatus, StaleSource
from odds_lambda.scheduling.backends.base import JobStatus, ScheduledJob


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        alerts=AlertConfig(
            alert_enabled=True,
            alert_rate_limit_minutes=30,
            quota_warning_threshold=0.2,
            quota_critical_threshold=0.1,
            stale_data_hours=2,
        ),
        api=APIConfig(key="test-key", quota=500),
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
    async def test_should_send_alert_when_no_recent_alerts(self, mock_session, mock_settings):
        """Should allow alert when no recent alerts of same type."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock query to return 0 recent alerts
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0
        mock_session.execute.return_value = mock_result

        should_send = await monitor._should_send_alert("test_alert")

        assert should_send is True

    @pytest.mark.asyncio
    async def test_should_not_send_alert_when_recent_alert_exists(
        self, mock_session, mock_settings
    ):
        """Should block alert when recent alert of same type exists."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock query to return 1 recent alert
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 1
        mock_session.execute.return_value = mock_result

        should_send = await monitor._should_send_alert("test_alert")

        assert should_send is False

    @pytest.mark.asyncio
    async def test_record_alert_stores_in_database(self, mock_session, mock_settings):
        """Should store alert record in database."""
        monitor = HealthMonitor(mock_session, mock_settings)

        await monitor._record_alert(
            alert_type="test_alert",
            severity="warning",
            message="Test message",
            context={"key": "value"},
        )

        # Verify session.add was called with AlertHistory
        mock_session.add.assert_called_once()
        added_record = mock_session.add.call_args[0][0]
        assert isinstance(added_record, AlertHistory)
        assert added_record.alert_type == "test_alert"
        assert added_record.severity == "warning"
        assert added_record.message == "Test message"

        # Verify commit was called
        mock_session.commit.assert_called_once()

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
            mock_reader.get_database_stats.return_value = {
                "api_quota_remaining": 125,  # 25% of 500
            }
            mock_reader_class.return_value = mock_reader

            is_healthy, quota_remaining, quota_fraction = await monitor.check_api_quota()

            assert is_healthy is True
            assert quota_remaining == 125
            assert quota_fraction == 0.25

    @pytest.mark.asyncio
    async def test_check_api_quota_low_warning(self, mock_session, mock_settings):
        """Should return unhealthy when quota is below warning threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_database_stats.return_value = {
                "api_quota_remaining": 75,  # 15% of 500 (below 20% warning)
            }
            mock_reader_class.return_value = mock_reader

            is_healthy, quota_remaining, quota_fraction = await monitor.check_api_quota()

            assert is_healthy is False
            assert quota_remaining == 75
            assert quota_fraction == 0.15

    @pytest.mark.asyncio
    async def test_collect_metrics_returns_full_metrics(self, mock_session, mock_settings):
        """Should collect all health metrics."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_database_stats.return_value = {
                "fetch_success_rate_24h": 95.5,
                "api_quota_remaining": 15000,
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
            assert metrics.fetch_success_rate_24h == 95.5
            assert metrics.api_quota_remaining == 15000
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
                fetch_success_rate_24h=95.0,
                hours_since_last_fetch=0.5,
                api_quota_remaining=15000,
                data_quality_errors_24h=2,
                events_scheduled=5,
                events_live=0,
                events_final=50,
            )

            with (
                patch.object(monitor, "check_stale_sources", return_value=[]),
                patch.object(monitor, "check_mlb_calendar", return_value=None),
                patch.object(monitor, "check_missing_betfair_closes", return_value=[]),
            ):
                with patch.object(monitor, "check_api_quota", return_value=(True, 15000, 0.75)):
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
                fetch_success_rate_24h=50.0,
                hours_since_last_fetch=3.0,
                api_quota_remaining=1000,
                data_quality_errors_24h=15,
                events_scheduled=0,
                events_live=0,
                events_final=50,
            )

            # Mock checks to return unhealthy
            with (
                patch.object(monitor, "check_mlb_calendar", return_value=None),
                patch.object(monitor, "check_missing_betfair_closes", return_value=[]),
                patch.object(
                    monitor,
                    "check_stale_sources",
                    return_value=[
                        StaleSource(
                            source="oddsportal",
                            sport_key="soccer_epl",
                            last_snapshot=datetime.now(UTC) - timedelta(hours=9),
                            active_hours=9.0,
                            threshold_hours=3.0,
                        )
                    ],
                ),
            ):
                with patch.object(monitor, "check_api_quota", return_value=(False, 1000, 0.05)):
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
            assert "oddsportal has written no soccer_epl snapshot" in status.issues_detected[0]

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

        # Mock query: heartbeat from 30 hours ago (exceeds every expectation)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = datetime.now(UTC) - timedelta(hours=30)
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
                ("check_mlb_calendar", None),
                ("check_missing_betfair_closes", []),
                ("check_api_quota", (True, None, None)),
                ("check_data_quality", (True, 0)),
                ("check_job_heartbeats", []),
                ("check_event_identity", []),
                ("check_unsettled_events", []),
                ("purge_old_heartbeats", 0),
                (
                    "collect_metrics",
                    HealthMetrics(
                        fetch_success_rate_24h=100.0,
                        hours_since_last_fetch=0.1,
                        api_quota_remaining=None,
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
