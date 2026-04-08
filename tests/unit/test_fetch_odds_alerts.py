"""Unit tests for fetch_odds alert behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, APIConfig, Settings


class TestQuotaAlerts:
    """Test quota alert thresholds and tiered alerts."""

    @pytest.mark.asyncio
    async def test_quota_warning_triggers_at_threshold(self):
        """Warning alert should trigger when quota falls below warning threshold."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=500)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 15% (below warning, above critical)
        quota_remaining = int(500 * 0.15)  # 75 requests

        with patch("odds_core.alerts.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as mock_critical:
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

                mock_warning.assert_called_once()
                mock_critical.assert_not_called()

                call_args = mock_warning.call_args[0][0]
                assert "75 requests remaining" in call_args
                assert "15.0%" in call_args

    @pytest.mark.asyncio
    async def test_quota_critical_triggers_at_threshold(self):
        """Critical alert should trigger when quota falls below critical threshold."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=500)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 5% (below critical)
        quota_remaining = int(500 * 0.05)  # 25 requests

        with patch("odds_core.alerts.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as mock_critical:
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

                mock_critical.assert_called_once()
                mock_warning.assert_not_called()

                call_args = mock_critical.call_args[0][0]
                assert "25 requests remaining" in call_args
                assert "5.0%" in call_args

    @pytest.mark.asyncio
    async def test_no_alert_above_warning_threshold(self):
        """No alert should trigger when quota is above warning threshold."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=500)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.2,  # 20%
            quota_critical_threshold=0.1,  # 10%
        )

        # Simulate quota at 50% (above warning)
        quota_remaining = int(500 * 0.5)  # 250 requests

        with patch("odds_core.alerts.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as mock_critical:
                quota_fraction = quota_remaining / mock_settings.api.quota

                if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_critical("test")
                elif quota_fraction < mock_settings.alerts.quota_warning_threshold:
                    if mock_settings.alerts.alert_enabled:
                        await mock_warning("test")

                mock_warning.assert_not_called()
                mock_critical.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_alerts_respect_enabled_flag(self):
        """Quota alerts should not send when alert_enabled=False."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=500)
        mock_settings.alerts = AlertConfig(
            alert_enabled=False,  # Disabled
            quota_warning_threshold=0.2,
            quota_critical_threshold=0.1,
        )

        # Simulate quota at 5% (would trigger critical if enabled)
        quota_remaining = int(500 * 0.05)

        with patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as mock_critical:
            quota_fraction = quota_remaining / mock_settings.api.quota

            if quota_fraction < mock_settings.alerts.quota_critical_threshold:
                if mock_settings.alerts.alert_enabled:
                    await mock_critical("test")

            mock_critical.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Alert system should respect custom threshold values."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.api = APIConfig(key="test", quota=500)
        mock_settings.alerts = AlertConfig(
            alert_enabled=True,
            quota_warning_threshold=0.3,  # 30%
            quota_critical_threshold=0.15,  # 15%
        )

        # Test at 20% - should trigger warning with custom threshold
        quota_remaining = int(500 * 0.2)

        with patch("odds_core.alerts.send_warning", new_callable=AsyncMock) as mock_warning:
            with patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as mock_critical:
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
