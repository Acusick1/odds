"""Integration tests for health check job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import AlertConfig, Settings, get_settings
from odds_core.models import AlertHistory, DataQualityLog, FetchLog
from odds_lambda.health_monitor import HealthStatus
from odds_lambda.jobs import check_health
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select


class TestHealthCheckIntegration:
    """Integration tests for health check job."""

    @pytest.mark.asyncio
    async def test_health_check_job_with_healthy_system(self, test_session):
        """Test health check job when system is healthy."""
        writer = OddsWriter(test_session)

        # Create recent successful fetch logs
        for i in range(5):
            fetch_log = FetchLog(
                fetch_time=datetime.now(UTC) - timedelta(minutes=i * 30),
                sport_key="basketball_nba",
                events_count=10,
                bookmakers_count=8,
                success=True,
                api_quota_remaining=15000,
            )
            test_session.add(fetch_log)

        await test_session.commit()

        # Mock alert system to prevent actual alerts
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            # Mock get_settings to use test database
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False  # Disable alerts for test
                mock_get_settings.return_value = mock_settings

                # Call check_system_health directly instead of mocking session
                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is True
        assert len(health_status.issues_detected) == 0
        assert health_status.metrics.fetch_success_rate_24h == 100.0
        assert health_status.metrics.api_quota_remaining == 15000
        assert health_status.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_detects_stale_data(self, test_session):
        """Test health check detects stale data (no recent fetches)."""
        writer = OddsWriter(test_session)

        # Create old fetch log (5 hours ago, threshold is 2 hours)
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(hours=5),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=15000,
        )
        test_session.add(fetch_log)
        await test_session.commit()

        # Mock alert system
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("No data fetched in" in issue for issue in health_status.issues_detected)

    @pytest.mark.asyncio
    async def test_health_check_detects_consecutive_failures(self, test_session):
        """Test health check detects consecutive failures."""
        writer = OddsWriter(test_session)

        # Create 4 consecutive failures (threshold is 3)
        for i in range(4):
            fetch_log = FetchLog(
                fetch_time=datetime.now(UTC) - timedelta(minutes=i * 30),
                sport_key="basketball_nba",
                events_count=0,
                bookmakers_count=0,
                success=False,
                error_message="API error",
                api_quota_remaining=15000,
            )
            test_session.add(fetch_log)

        await test_session.commit()

        # Mock alert system
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("consecutive" in issue.lower() for issue in health_status.issues_detected)
        assert health_status.metrics.consecutive_failures == 4

    @pytest.mark.asyncio
    async def test_health_check_detects_low_quota(self, test_session):
        """Test health check detects low API quota."""
        writer = OddsWriter(test_session)

        # Create fetch log with low quota (1000 out of 20000 = 5%, below 10% critical)
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=1000,
        )
        test_session.add(fetch_log)
        await test_session.commit()

        # Mock alert system
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("quota" in issue.lower() for issue in health_status.issues_detected)
        assert health_status.metrics.api_quota_remaining == 1000

    @pytest.mark.asyncio
    async def test_health_check_detects_data_quality_issues(self, test_session):
        """Test health check detects high data quality error rate."""
        from odds_core.models import Event, EventStatus

        writer = OddsWriter(test_session)

        # Create test event for foreign key constraint
        test_event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Test Team A",
            away_team="Test Team B",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(test_event)
        await test_session.commit()  # Commit event before adding quality logs

        # Create successful fetch so we don't get stale data warning
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=15000,
        )
        test_session.add(fetch_log)

        # Create 12 data quality errors in last 24h (threshold is 10)
        for i in range(12):
            quality_log = DataQualityLog(
                event_id="test_event",
                severity="error",
                issue_type="suspicious_odds",
                description=f"Test error {i}",
                created_at=datetime.now(UTC) - timedelta(hours=i),
            )
            test_session.add(quality_log)

        await test_session.commit()

        # Mock alert system
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("data quality" in issue.lower() for issue in health_status.issues_detected)
        assert health_status.metrics.data_quality_errors_24h == 12

    @pytest.mark.asyncio
    async def test_alert_deduplication_prevents_duplicate_alerts(self, test_session):
        """Test that alert deduplication prevents duplicate alerts within rate limit window."""
        writer = OddsWriter(test_session)

        # Create old fetch log to trigger stale data alert
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(hours=5),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=15000,
        )
        test_session.add(fetch_log)
        await test_session.commit()

        # Mock alert system to track calls
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = True  # Enable alerts
                mock_settings.alerts.alert_rate_limit_minutes = 30
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                # First health check - should send alert
                monitor = HealthMonitor(test_session, mock_settings)
                health_status1 = await monitor.check_system_health()
                first_alert_count = mock_alert.call_count

                # Second health check immediately after - should NOT send duplicate alert
                monitor2 = HealthMonitor(test_session, mock_settings)
                health_status2 = await monitor2.check_system_health()
                second_alert_count = mock_alert.call_count

        # Verify first check sent alert
        assert first_alert_count > 0
        assert health_status1.overall_healthy is False
        assert len(health_status1.alerts_sent) > 0

        # Verify second check was rate-limited
        assert second_alert_count == first_alert_count  # No new alerts
        assert health_status2.overall_healthy is False
        assert len(health_status2.alerts_sent) == 0  # Rate-limited

        # Verify AlertHistory records exist
        reader = OddsReader(test_session)
        result = await test_session.execute(select(AlertHistory))
        alert_records = list(result.scalars().all())
        assert len(alert_records) > 0
        assert any(record.alert_type == "stale_data" for record in alert_records)

    @pytest.mark.asyncio
    async def test_health_check_with_configurable_thresholds(self, test_session):
        """Test that health check respects configurable thresholds."""
        from odds_core.models import Event, EventStatus

        writer = OddsWriter(test_session)

        # Create test event for foreign key constraint
        test_event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Test Team A",
            away_team="Test Team B",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(test_event)
        await test_session.commit()  # Commit event before adding quality logs

        # Create successful fetch
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=15000,
        )
        test_session.add(fetch_log)

        # Create 5 data quality errors (default threshold is 10)
        for i in range(5):
            quality_log = DataQualityLog(
                event_id="test_event",
                severity="error",
                issue_type="suspicious_odds",
                description=f"Test error {i}",
                created_at=datetime.now(UTC) - timedelta(hours=i),
            )
            test_session.add(quality_log)

        await test_session.commit()

        # Test with default threshold (10) - should be healthy
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_settings.alerts.data_quality_error_threshold = 10  # Default
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is True
        assert health_status.metrics.data_quality_errors_24h == 5

        # Test with lower threshold (3) - should be unhealthy
        with patch("odds_cli.alerts.base.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_settings.alerts.data_quality_error_threshold = 3  # Lower threshold
                mock_get_settings.return_value = mock_settings

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert any("data quality" in issue.lower() for issue in health_status.issues_detected)
