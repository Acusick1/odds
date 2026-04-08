"""Unit tests for alert infrastructure."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.alerts import (
    AlertManager,
    DiscordAlert,
    job_alert_context,
    send_critical,
    send_error,
    send_warning,
)
from odds_core.config import AlertConfig, Settings


class TestAlertManager:
    """Test AlertManager initialization and configuration."""

    def test_alert_manager_reads_nested_config_fields(self):
        """AlertManager should correctly read alerts.* nested config fields."""
        # Create mock settings with nested alert config
        settings = MagicMock(spec=Settings)
        settings.alerts = AlertConfig(
            alert_enabled=True, discord_webhook_url="https://discord.com/api/webhooks/test"
        )

        manager = AlertManager(config=settings)

        assert manager.enabled is True
        assert len(manager.channels) == 1
        assert isinstance(manager.channels[0], DiscordAlert)

    def test_alert_manager_disabled_when_flag_false(self):
        """AlertManager should be disabled when alert_enabled=False."""
        settings = MagicMock(spec=Settings)
        settings.alerts = AlertConfig(
            alert_enabled=False, discord_webhook_url="https://discord.com/api/webhooks/test"
        )

        manager = AlertManager(config=settings)

        assert manager.enabled is False
        # Channels still initialized but won't be used
        assert len(manager.channels) == 1

    def test_alert_manager_no_channels_without_webhook(self):
        """AlertManager should have no channels if no webhook configured."""
        settings = MagicMock(spec=Settings)
        settings.alerts = AlertConfig(alert_enabled=True, discord_webhook_url=None)

        manager = AlertManager(config=settings)

        assert manager.enabled is True
        assert len(manager.channels) == 0

    @pytest.mark.asyncio
    async def test_alert_respects_enabled_flag(self):
        """Alerts should not send when enabled=False."""
        settings = MagicMock(spec=Settings)
        settings.alerts = AlertConfig(
            alert_enabled=False, discord_webhook_url="https://discord.com/api/webhooks/test"
        )

        manager = AlertManager(config=settings)

        # Mock the channel send method
        with patch.object(manager.channels[0], "send", new_callable=AsyncMock) as mock_send:
            await manager.alert("Test message", "warning")

            # Should not call send because enabled=False
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_alert_sends_when_enabled(self):
        """Alerts should send when enabled=True."""
        settings = MagicMock(spec=Settings)
        settings.alerts = AlertConfig(
            alert_enabled=True, discord_webhook_url="https://discord.com/api/webhooks/test"
        )

        manager = AlertManager(config=settings)

        # Mock the channel send method
        with patch.object(manager.channels[0], "send", new_callable=AsyncMock) as mock_send:
            await manager.alert("Test message", "warning")

            # Should call send because enabled=True
            mock_send.assert_called_once_with("Test message", "warning")


class TestDiscordAlert:
    """Test Discord webhook alert implementation."""

    @pytest.mark.asyncio
    async def test_discord_alert_sends_payload(self):
        """Discord alert should send properly formatted payload."""
        webhook_url = "https://discord.com/api/webhooks/test"
        alert = DiscordAlert(webhook_url)

        # Mock aiohttp session
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204
            mock_response.__aenter__.return_value = mock_response

            mock_post = AsyncMock()
            mock_post.return_value = mock_response

            mock_session.return_value.__aenter__.return_value.post = mock_post

            await alert.send("Test alert message", "warning")

            # Verify webhook was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check URL
            assert call_args[0][0] == webhook_url

            # Check payload structure
            payload = call_args[1]["json"]
            assert "embeds" in payload
            assert len(payload["embeds"]) == 1
            assert payload["embeds"][0]["description"] == "Test alert message"
            assert payload["embeds"][0]["color"] == 16776960  # Yellow for warning

    @pytest.mark.asyncio
    async def test_discord_alert_handles_failure(self):
        """Discord alert should handle HTTP errors gracefully."""
        webhook_url = "https://discord.com/api/webhooks/test"
        alert = DiscordAlert(webhook_url)

        # Mock aiohttp session with error response
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad Request")
            mock_response.__aenter__.return_value = mock_response

            mock_post = AsyncMock()
            mock_post.return_value = mock_response

            mock_session.return_value.__aenter__.return_value.post = mock_post

            # Should not raise exception
            await alert.send("Test message", "error")


class TestConvenienceFunctions:
    """Test convenience functions for sending alerts."""

    @pytest.mark.asyncio
    async def test_send_warning_calls_alert_manager(self):
        """send_warning should call alert manager with warning severity."""
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            await send_warning("Test warning")
            mock_alert.assert_called_once_with("Test warning", "warning")

    @pytest.mark.asyncio
    async def test_send_error_calls_alert_manager(self):
        """send_error should call alert manager with error severity."""
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            await send_error("Test error")
            mock_alert.assert_called_once_with("Test error", "error")

    @pytest.mark.asyncio
    async def test_send_critical_calls_alert_manager(self):
        """send_critical should call alert manager with critical severity."""
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock) as mock_alert:
            await send_critical("Test critical")
            mock_alert.assert_called_once_with("Test critical", "critical")


class TestJobAlertContext:
    """Test job_alert_context context manager."""

    @pytest.mark.asyncio
    @patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock)
    async def test_records_heartbeat_on_success(self, mock_record: AsyncMock) -> None:
        """Should write heartbeat row on clean exit."""
        async with job_alert_context("test-job"):
            pass

        mock_record.assert_called_once()
        assert mock_record.call_args[1]["alert_type"] == "heartbeat:test-job"
        assert mock_record.call_args[1]["severity"] == "info"

    @pytest.mark.asyncio
    @patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock)
    @patch("odds_core.alerts.check_rate_limit", new_callable=AsyncMock, return_value=True)
    @patch("odds_core.alerts.alert_manager")
    async def test_sends_alert_on_failure(
        self,
        mock_manager: MagicMock,
        mock_rate_limit: AsyncMock,
        mock_record: AsyncMock,
    ) -> None:
        """Should send CRITICAL alert and re-raise on exception."""
        mock_manager.enabled = True
        mock_manager.alert = AsyncMock()

        with pytest.raises(ValueError, match="boom"):
            async with job_alert_context("test-job"):
                raise ValueError("boom")

        mock_manager.alert.assert_called_once()
        msg = mock_manager.alert.call_args[0][0]
        assert "test-job" in msg
        assert "ValueError" in msg
        assert "boom" in msg

        # Should record the failure alert, NOT the heartbeat
        mock_record.assert_called_once()
        assert mock_record.call_args[0][0] == "job_failure:test-job"

    @pytest.mark.asyncio
    @patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock)
    @patch("odds_core.alerts.check_rate_limit", new_callable=AsyncMock, return_value=False)
    @patch("odds_core.alerts.alert_manager")
    async def test_rate_limits_failure_alerts(
        self,
        mock_manager: MagicMock,
        mock_rate_limit: AsyncMock,
        mock_record: AsyncMock,
    ) -> None:
        """Should suppress alert when rate-limited, still re-raise."""
        mock_manager.enabled = True
        mock_manager.alert = AsyncMock()

        with pytest.raises(ValueError):
            async with job_alert_context("test-job"):
                raise ValueError("boom")

        mock_manager.alert.assert_not_called()
        mock_record.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock)
    @patch("odds_core.alerts.alert_manager")
    async def test_skips_alert_when_disabled(
        self,
        mock_manager: MagicMock,
        mock_record: AsyncMock,
    ) -> None:
        """Should not send alert when alerts are disabled."""
        mock_manager.enabled = False
        mock_manager.alert = AsyncMock()

        with pytest.raises(ValueError):
            async with job_alert_context("test-job"):
                raise ValueError("boom")

        mock_manager.alert.assert_not_called()
