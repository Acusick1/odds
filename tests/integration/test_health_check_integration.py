"""Integration tests for health check job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import get_settings
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import (
    AlertHistory,
    DataQualityLog,
    Event,
    EventStatus,
    FetchLog,
    OddsSnapshot,
)
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy import select


@pytest.fixture(autouse=True)
def deliverable_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("odds_core.alerts.alert_manager.enabled", True)


async def _create_recent_heartbeats(session) -> None:
    """Insert recent heartbeat records for all monitored jobs."""
    from odds_core.config import AlertConfig

    now = datetime.now(UTC)
    session.add(
        MlbProbablePitchers(
            game_pk=1,
            commence_time=now + timedelta(hours=5),
            fetched_at=now - timedelta(minutes=10),
            home_team="A",
            away_team="B",
        )
    )
    for job_name in AlertConfig().heartbeat_expectations:
        session.add(
            AlertHistory(
                alert_type=f"heartbeat:{job_name}",
                severity="info",
                message=f"Job {job_name} completed successfully",
                sent_at=now - timedelta(minutes=10),
            )
        )


async def _create_recent_snapshot(session, *, minutes_ago: int = 30) -> None:
    """Insert an Event + OddsSnapshot so the health check finds recent data."""
    event = Event(
        id=f"health_test_{minutes_ago}",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime.now(UTC) + timedelta(hours=24),
        home_team="Team A",
        away_team="Team B",
        status=EventStatus.SCHEDULED,
    )
    session.add(event)
    await session.flush()
    snapshot = OddsSnapshot(
        event_id=event.id,
        snapshot_time=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        raw_data={},
        bookmaker_count=1,
    )
    session.add(snapshot)


async def _create_stale_oddsportal_source(session, *, hours_ago: float = 40) -> None:
    """An EPL fixture inside the horizon whose OddsPortal feed last wrote hours ago."""
    now = datetime.now(UTC)
    event = Event(
        id="health_stale_epl",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=now + timedelta(hours=6),
        home_team="Arsenal",
        away_team="Chelsea",
        status=EventStatus.SCHEDULED,
    )
    session.add(event)
    await session.flush()
    session.add(
        OddsSnapshot(
            event_id=event.id,
            snapshot_time=now - timedelta(hours=hours_ago),
            raw_data={"source": "oddsportal_live"},
            bookmaker_count=1,
            source=Source.ODDSPORTAL,
            time_basis=TimeBasis.OBSERVED,
        )
    )
    await session.commit()


class TestHealthCheckIntegration:
    """Integration tests for health check job."""

    @pytest.mark.asyncio
    async def test_health_check_job_with_healthy_system(self, test_session):
        """Test health check job when system is healthy."""
        # Create recent snapshot and heartbeats so all checks pass
        await _create_recent_snapshot(test_session)
        await _create_recent_heartbeats(test_session)

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
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
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
        assert health_status.metrics.api_quota_remaining == 15000

    @pytest.mark.asyncio
    async def test_health_check_detects_stale_data(self, test_session):
        """Test health check detects stale data (no recent fetches)."""
        await _create_stale_oddsportal_source(test_session)

        # Mock alert system
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("oddsportal has written no" in issue for issue in health_status.issues_detected)

    @pytest.mark.asyncio
    async def test_health_check_detects_low_quota(self, test_session):
        """Test health check detects low API quota."""
        # Create recent snapshot so stale data check passes
        await _create_recent_snapshot(test_session, minutes_ago=10)

        # Create fetch log with low quota (25 out of 500 = 5%, below 10% critical)
        fetch_log = FetchLog(
            fetch_time=datetime.now(UTC) - timedelta(minutes=30),
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=25,
        )
        test_session.add(fetch_log)
        await test_session.commit()

        # Mock alert system
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_settings.api.quota = 500  # Set explicit quota for test (25/500 = 5%)
                mock_get_settings.return_value = mock_settings

                from odds_lambda.health_monitor import HealthMonitor

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert len(health_status.issues_detected) > 0
        assert any("quota" in issue.lower() for issue in health_status.issues_detected)
        assert health_status.metrics.api_quota_remaining == 25

    @pytest.mark.asyncio
    async def test_health_check_detects_data_quality_issues(self, test_session):
        """Test health check detects high data quality error rate."""
        # Create recent snapshot so stale data check passes
        await _create_recent_snapshot(test_session, minutes_ago=15)

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
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
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
        await _create_stale_oddsportal_source(test_session)

        # Mock alert system to track calls
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = True  # Enable alerts
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
        result = await test_session.execute(select(AlertHistory))
        alert_records = list(result.scalars().all())
        assert len(alert_records) > 0
        assert any(
            record.alert_type == "stale_data:oddsportal:soccer_epl" for record in alert_records
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", ["data_quality_errors", "quota_critical"])
    async def test_persistent_warning_repeats_only_after_reminder_interval(
        self, test_session, condition
    ):
        """A condition that stays true alerts once, then again each reminder interval."""
        await _create_recent_snapshot(test_session, minutes_ago=20)
        await _create_recent_heartbeats(test_session)
        test_session.add(
            Event(
                id="test_event",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC),
                home_team="Test Team A",
                away_team="Test Team B",
                status=EventStatus.SCHEDULED,
            )
        )
        await test_session.commit()
        if condition == "data_quality_errors":
            for i in range(12):
                test_session.add(
                    DataQualityLog(
                        event_id="test_event",
                        severity="error",
                        issue_type="suspicious_odds",
                        description=f"Test error {i}",
                        created_at=datetime.now(UTC) - timedelta(hours=i),
                    )
                )
        else:
            test_session.add(
                FetchLog(
                    fetch_time=datetime.now(UTC) - timedelta(minutes=30),
                    sport_key="basketball_nba",
                    events_count=10,
                    bookmakers_count=8,
                    success=True,
                    api_quota_remaining=25,
                )
            )
        await test_session.commit()

        from odds_lambda.health_monitor import HealthMonitor

        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            settings = get_settings()
            settings.alerts.alert_enabled = True
            settings.api.quota = 500
            reminder = timedelta(minutes=settings.alerts.stale_reminder_minutes)

            async def run_check() -> list[str]:
                status = await HealthMonitor(test_session, settings).check_system_health()
                return status.alerts_sent

            assert condition in await run_check()
            for _ in range(3):
                assert condition not in await run_check()
            assert mock_alert.call_count == 1

            row = (
                await test_session.execute(
                    select(AlertHistory).where(AlertHistory.alert_type == condition)
                )
            ).scalar_one()
            row.sent_at = datetime.now(UTC) - reminder - timedelta(minutes=1)
            await test_session.commit()

            row.sent_at = datetime.now(UTC) - timedelta(minutes=60)
            await test_session.commit()
            assert condition not in await run_check()

            row.sent_at = datetime.now(UTC) - reminder - timedelta(minutes=1)
            await test_session.commit()
            assert condition in await run_check()
            assert mock_alert.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check_with_configurable_thresholds(self, test_session):
        """Test that health check respects configurable thresholds."""
        # Create recent snapshot and heartbeats so non-quality checks pass
        await _create_recent_snapshot(test_session, minutes_ago=20)
        await _create_recent_heartbeats(test_session)

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
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
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
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
            with patch("odds_lambda.health_monitor.get_settings") as mock_get_settings:
                mock_settings = get_settings()
                mock_settings.alerts.alert_enabled = False
                mock_settings.alerts.data_quality_error_threshold = 3  # Lower threshold
                mock_get_settings.return_value = mock_settings

                monitor = HealthMonitor(test_session, mock_settings)
                health_status = await monitor.check_system_health()

        assert health_status.overall_healthy is False
        assert any("data quality" in issue.lower() for issue in health_status.issues_detected)
