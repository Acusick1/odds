"""Alert infrastructure for future use."""

from abc import ABC, abstractmethod

import aiohttp
import structlog

from core.config import settings

logger = structlog.get_logger()


class AlertBase(ABC):
    """Base class for all alert types."""

    @abstractmethod
    async def send(self, message: str, severity: str = "info"):
        """
        Send alert via specific channel.

        Args:
            message: Alert message content
            severity: Severity level (info, warning, error, critical)
        """
        pass


class DiscordAlert(AlertBase):
    """Discord webhook implementation."""

    def __init__(self, webhook_url: str):
        """
        Initialize Discord alert.

        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url

    async def send(self, message: str, severity: str = "info"):
        """
        Send alert to Discord via webhook.

        Args:
            message: Alert message content
            severity: Severity level
        """
        # Color codes for different severity levels
        colors = {
            "info": 3447003,  # Blue
            "warning": 16776960,  # Yellow
            "error": 15158332,  # Red
            "critical": 10038562,  # Dark red
        }

        payload = {
            "embeds": [
                {
                    "title": f"Betting Odds Pipeline - {severity.upper()}",
                    "description": message,
                    "color": colors.get(severity, colors["info"]),
                    "timestamp": None,  # Discord will use current time
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        logger.info("discord_alert_sent", severity=severity)
                    else:
                        logger.error(
                            "discord_alert_failed",
                            status=response.status,
                            response=await response.text(),
                        )
        except Exception as e:
            logger.error("discord_alert_error", error=str(e))


class AlertManager:
    """Route alerts to appropriate channels."""

    def __init__(self, config: settings | None = None):
        """
        Initialize alert manager.

        Args:
            config: Settings instance (defaults to global settings)
        """
        self.config = config or settings
        self.enabled = self.config.alert_enabled
        self.channels: list[AlertBase] = []

        # Initialize configured channels
        if self.config.discord_webhook_url:
            self.channels.append(DiscordAlert(self.config.discord_webhook_url))

        if self.enabled:
            logger.info(
                "alert_manager_initialized", channels=len(self.channels), enabled=self.enabled
            )

    async def alert(self, message: str, severity: str = "info"):
        """
        Send alert to all configured channels.

        Args:
            message: Alert message content
            severity: Severity level (info, warning, error, critical)

        Note:
            Only sends if alerts are enabled in configuration
        """
        if not self.enabled:
            logger.debug("alert_skipped_disabled", message=message)
            return

        if not self.channels:
            logger.warning("alert_no_channels", message=message)
            return

        for channel in self.channels:
            try:
                await channel.send(message, severity)
            except Exception as e:
                logger.error("alert_channel_failed", channel=type(channel).__name__, error=str(e))


# Global alert manager instance
alert_manager = AlertManager()


# Convenience functions
async def send_info(message: str):
    """Send info alert."""
    await alert_manager.alert(message, "info")


async def send_warning(message: str):
    """Send warning alert."""
    await alert_manager.alert(message, "warning")


async def send_error(message: str):
    """Send error alert."""
    await alert_manager.alert(message, "error")


async def send_critical(message: str):
    """Send critical alert."""
    await alert_manager.alert(message, "critical")
