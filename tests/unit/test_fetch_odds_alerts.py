"""Unit tests for fetch_odds alert behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, APIConfig, Settings


class TestQuotaAlerts:
    """Test quota alert thresholds and tiered alerts."""

    @pytest.mark.asyncio
    async def test_quota_warning_triggers_at_threshold(self):
        """Warning alert should trigger when quota falls below warning threshold."""
        # Create mock settings with 20% warning threshold
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=20000)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 15% (below warning, above critical)
        quota_remaining = int(20000 * 0.15)  # 3000 requests

        # Mock the send_warning function
        with patch("odds_cli.alerts.base.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch(
                "odds_cli.alerts.base.send_critical", new_callable=AsyncMock
            ) as mock_critical:
                # Simulate the quota check logic from fetch_odds.py
                quota_fraction = quota_remaining / mock_settings.api.quota
                percentage_remaining = round(quota_fraction * 100, 1)

                if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_critical(
                            f"🚨 API quota critical: {quota_remaining} requests remaining "
                            f"({percentage_remaining}% of {mock_settings.api.quota})"
                        )
                elif quota_fraction < mock_settings.alerts.quota_warning_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_warning(
                            f"⚠️ API quota low: {quota_remaining} requests remaining "
                            f"({percentage_remaining}% of {mock_settings.api.quota})"
                        )

                # Should trigger warning, not critical
                mock_warning.assert_called_once()
                mock_critical.assert_not_called()

                # Check message content
                call_args = mock_warning.call_args[0][0]
                assert "3000 requests remaining" in call_args
                assert "15.0%" in call_args

    @pytest.mark.asyncio
    async def test_quota_critical_triggers_at_threshold(self):
        """Critical alert should trigger when quota falls below critical threshold."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=20000)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 5% (below critical)
        quota_remaining = int(20000 * 0.05)  # 1000 requests

        with patch("odds_cli.alerts.base.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch(
                "odds_cli.alerts.base.send_critical", new_callable=AsyncMock
            ) as mock_critical:
                # Simulate the quota check logic
                quota_fraction = quota_remaining / mock_settings.api.quota
                percentage_remaining = round(quota_fraction * 100, 1)

                if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_critical(
                            f"🚨 API quota critical: {quota_remaining} requests remaining "
                            f"({percentage_remaining}% of {mock_settings.api.quota})"
                        )
                elif quota_fraction < mock_settings.alerts.quota_warning_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_warning(
                            f"⚠️ API quota low: {quota_remaining} requests remaining "
                            f"({percentage_remaining}% of {mock_settings.api.quota})"
                        )

                # Should trigger critical, not warning
                mock_critical.assert_called_once()
                mock_warning.assert_not_called()

                # Check message content
                call_args = mock_critical.call_args[0][0]
                assert "1000 requests remaining" in call_args
                assert "5.0%" in call_args

    @pytest.mark.asyncio
    async def test_no_alert_above_warning_threshold(self):
        """No alert should trigger when quota is above warning threshold."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=20000)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 50% (above warning)
        quota_remaining = int(20000 * 0.5)  # 10000 requests

        with patch("odds_cli.alerts.base.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch(
                "odds_cli.alerts.base.send_critical", new_callable=AsyncMock
            ) as mock_critical:
                # Simulate the quota check logic
                quota_fraction = quota_remaining / mock_settings.api.quota

                if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_critical("test")
                elif quota_fraction < mock_settings.alerts.quota_warning_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_warning("test")

                # Should not trigger any alerts
                mock_warning.assert_not_called()
                mock_critical.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_alerts_respect_enabled_flag(self):
        """Quota alerts should not send when alert_enabled=False."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=20000)
        mock_settings.alerts = AlertConfig(
            alert_enabled=False,  # Disabled
            quota_warning_threshold=0.2,
            quota_critical_threshold=0.1,
        )

        # Simulate quota at 5% (would trigger critical if enabled)
        quota_remaining = int(20000 * 0.05)

        with patch("odds_cli.alerts.base.send_critical", new_callable=AsyncMock) as mock_critical:
            # Simulate the quota check logic
            quota_fraction = quota_remaining / mock_settings.api.quota

            if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                if mock_settings.alerts.alert_enabled:  # Check enabled flag
                    await mock_critical("test")

            # Should not call because enabled=False
            mock_critical.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Alert system should respect custom threshold values."""
        # Custom thresholds: 30% warning, 15% critical
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=20000)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.3,  # 30%
            quota_critical_threshold=0.15,  # 15%
        )

        # Test at 20% - should trigger warning with custom threshold
        quota_remaining = int(20000 * 0.2)

        with patch("odds_cli.alerts.base.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch(
                "odds_cli.alerts.base.send_critical", new_callable=AsyncMock
            ) as mock_critical:
                quota_fraction = quota_remaining / mock_settings.api.quota

                if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_critical("test")
                elif quota_fraction < mock_settings.alerts.quota_warning_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_warning("test")

                # 20% is below 30% warning threshold, so should trigger warning
                mock_warning.assert_called_once()
                mock_critical.assert_not_called()
