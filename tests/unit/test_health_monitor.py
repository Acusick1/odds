"""Unit tests for health monitoring system."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, APIConfig, Settings
from odds_core.models import AlertHistory, FetchLog
from sqlalchemy import select

from odds_lambda.health_monitor import HealthMetrics, HealthMonitor, HealthStatus


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.alerts = AlertConfig(
        alert_enabled=True,
        alert_rate_limit_minutes=30,
        quota_warning_threshold=0.2,
        quota_critical_threshold=0.1,
        consecutive_failures_threshold=3,
        stale_data_hours=2,
    )
    settings.api = APIConfig(
        key="test-key",
        quota=20000,
    )
    return settings


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
    async def test_check_stale_data_healthy(self, mock_session, mock_settings):
        """Should return healthy when recent fetch exists."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock recent fetch log
        recent_fetch = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=5,
            bookmakers_count=8,
            success=True,
        )

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_fetch_logs.return_value = [recent_fetch]
            mock_reader_class.return_value = mock_reader

            is_healthy, issue = await monitor.check_stale_data()

            assert is_healthy is True
            assert issue is None

    @pytest.mark.asyncio
    async def test_check_stale_data_unhealthy(self, mock_session, mock_settings):
        """Should return unhealthy when data is stale."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock stale fetch log (3 hours ago)
        stale_fetch = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(hours=3),
            sport_key="basketball_nba",
            events_count=5,
            bookmakers_count=8,
            success=True,
        )

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_fetch_logs.return_value = [stale_fetch]
            mock_reader_class.return_value = mock_reader

            is_healthy, issue = await monitor.check_stale_data()

            assert is_healthy is False
            assert "No data fetched in" in issue
            assert "3." in issue  # Should mention 3 hours

    @pytest.mark.asyncio
    async def test_check_consecutive_failures_healthy(self, mock_session, mock_settings):
        """Should return healthy when no consecutive failures."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock fetch logs with success
        fetch_logs = [
            FetchLog(
                fetch_time=datetime.now(UTC) - timedelta(minutes=i * 30),
                sport_key="basketball_nba",
                events_count=5,
                bookmakers_count=8,
                success=True,
            )
            for i in range(5)
        ]

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_fetch_logs.return_value = fetch_logs
            mock_reader_class.return_value = mock_reader

            is_healthy, failure_count = await monitor.check_consecutive_failures()

            assert is_healthy is True
            assert failure_count == 0

    @pytest.mark.asyncio
    async def test_check_consecutive_failures_unhealthy(self, mock_session, mock_settings):
        """Should return unhealthy when consecutive failures exceed threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        # Mock fetch logs with 3 consecutive failures (threshold = 3)
        fetch_logs = [
            FetchLog(
                fetch_time=datetime.now(UTC) - timedelta(minutes=i * 30),
                sport_key="basketball_nba",
                events_count=0,
                bookmakers_count=0,
                success=False,
                error_message="API error",
            )
            for i in range(3)
        ]

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_fetch_logs.return_value = fetch_logs
            mock_reader_class.return_value = mock_reader

            is_healthy, failure_count = await monitor.check_consecutive_failures()

            assert is_healthy is False
            assert failure_count == 3

    @pytest.mark.asyncio
    async def test_check_api_quota_healthy(self, mock_session, mock_settings):
        """Should return healthy when quota is above warning threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_database_stats.return_value = {
                "api_quota_remaining": 5000,  # 25% of 20000
            }
            mock_reader_class.return_value = mock_reader

            is_healthy, quota_remaining, quota_fraction = await monitor.check_api_quota()

            assert is_healthy is True
            assert quota_remaining == 5000
            assert quota_fraction == 0.25

    @pytest.mark.asyncio
    async def test_check_api_quota_low_warning(self, mock_session, mock_settings):
        """Should return unhealthy when quota is below warning threshold."""
        monitor = HealthMonitor(mock_session, mock_settings)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = AsyncMock()
            mock_reader.get_database_stats.return_value = {
                "api_quota_remaining": 3000,  # 15% of 20000 (below 20% warning)
            }
            mock_reader_class.return_value = mock_reader

            is_healthy, quota_remaining, quota_fraction = await monitor.check_api_quota()

            assert is_healthy is False
            assert quota_remaining == 3000
            assert quota_fraction == 0.15

    @pytest.mark.asyncio
    async def test_collect_metrics_returns_full_metrics(self, mock_session, mock_settings):
        """Should collect all health metrics."""
        monitor = HealthMonitor(mock_session, mock_settings)

        recent_fetch = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=5,
            bookmakers_count=8,
            success=True,
        )

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
            mock_reader.get_fetch_logs.return_value = [recent_fetch]
            mock_reader_class.return_value = mock_reader

            # Mock check methods
            with patch.object(monitor, "check_consecutive_failures", return_value=(True, 0)):
                with patch.object(monitor, "check_data_quality", return_value=(True, 2)):
                    metrics = await monitor.collect_metrics()

            assert isinstance(metrics, HealthMetrics)
            assert metrics.fetch_success_rate_24h == 95.5
            assert metrics.api_quota_remaining == 15000
            assert metrics.hours_since_last_fetch is not None
            assert metrics.consecutive_failures == 0
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
                consecutive_failures=0,
                data_quality_errors_24h=2,
                events_scheduled=5,
                events_live=0,
                events_final=50,
            )

            with patch.object(monitor, "check_stale_data", return_value=(True, None)):
                with patch.object(monitor, "check_consecutive_failures", return_value=(True, 0)):
                    with patch.object(
                        monitor, "check_api_quota", return_value=(True, 15000, 0.75)
                    ):
                        with patch.object(monitor, "check_data_quality", return_value=(True, 2)):
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
                consecutive_failures=4,
                data_quality_errors_24h=15,
                events_scheduled=0,
                events_live=0,
                events_final=50,
            )

            # Mock checks to return unhealthy
            with patch.object(
                monitor, "check_stale_data", return_value=(False, "Stale data issue")
            ):
                with patch.object(monitor, "check_consecutive_failures", return_value=(False, 4)):
                    with patch.object(
                        monitor, "check_api_quota", return_value=(False, 1000, 0.05)
                    ):
                        with patch.object(monitor, "check_data_quality", return_value=(False, 15)):
                            # Mock alert sending
                            with patch.object(monitor, "_send_alert", return_value=True):
                                status = await monitor.check_system_health()

            assert status.overall_healthy is False
            assert len(status.issues_detected) > 0
            # Should have attempted to send alerts
            assert "Stale data issue" in status.issues_detected[0]

    @pytest.mark.asyncio
    async def test_check_system_health_handles_errors_gracefully(
        self, mock_session, mock_settings
    ):
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
