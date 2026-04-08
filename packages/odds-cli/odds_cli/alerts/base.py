"""Alert infrastructure: Discord webhooks, job failure alerts, and heartbeat recording."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import aiohttp
import structlog
from odds_core.config import Settings, get_settings

logger = structlog.get_logger()


class AlertBase(ABC):
    """Base class for all alert types."""

    @abstractmethod
    async def send(self, message: str, severity: str = "info") -> None:
        """
        Send alert via specific channel.

        Args:
            message: Alert message content
            severity: Severity level (info, warning, error, critical)
        """
        pass

    @abstractmethod
    async def send_embed(self, embed: dict) -> None:
        """Send a raw embed payload via specific channel."""
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

    async def _post(self, payload: dict, log_event: str = "discord_alert") -> None:
        """Post a JSON payload to the Discord webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:
                        logger.info(f"{log_event}_sent")
                    else:
                        logger.error(
                            f"{log_event}_failed",
                            status=response.status,
                            response=await response.text(),
                        )
        except Exception as e:
            logger.error(f"{log_event}_error", error=str(e))

    async def send(self, message: str, severity: str = "info") -> None:
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

        await self._post(payload, "discord_alert")

    async def send_embed(self, embed: dict) -> None:
        await self._post({"embeds": [embed]}, "discord_embed")


class AlertManager:
    """Route alerts to appropriate channels."""

    def __init__(self, config: Settings | None = None):
        """
        Initialize alert manager.

        Args:
            config: Settings instance (defaults to global settings)
        """
        self.config = config or get_settings()
        self.enabled = self.config.alerts.alert_enabled
        self.channels: list[AlertBase] = []

        # Initialize configured channels
        if self.config.alerts.discord_webhook_url:
            self.channels.append(DiscordAlert(self.config.alerts.discord_webhook_url))

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

    async def send_embed(self, embed: dict) -> None:
        """Send a raw embed to all configured channels."""
        if not self.enabled:
            logger.debug("embed_skipped_disabled")
            return

        if not self.channels:
            logger.warning("embed_no_channels")
            return

        for channel in self.channels:
            try:
                await channel.send_embed(embed)
            except Exception as e:
                logger.error("embed_channel_failed", channel=type(channel).__name__, error=str(e))


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


# ---------------------------------------------------------------------------
# Rate-limited job alerts and heartbeat recording
# ---------------------------------------------------------------------------

_DEFAULT_RATE_LIMIT_MINUTES = 30


async def check_rate_limit(
    alert_type: str, rate_limit_minutes: int = _DEFAULT_RATE_LIMIT_MINUTES
) -> bool:
    """Return True if an alert of this type is allowed (not rate-limited).

    Queries AlertHistory for recent alerts within the rate-limit window.
    """
    from odds_core.database import async_session_maker
    from odds_core.models import AlertHistory
    from sqlalchemy import func, select

    cutoff = datetime.now(UTC) - timedelta(minutes=rate_limit_minutes)
    async with async_session_maker() as session:
        result = await session.execute(
            select(func.count(AlertHistory.id)).where(
                AlertHistory.alert_type == alert_type,
                AlertHistory.sent_at >= cutoff,
            )
        )
        return result.scalar_one() == 0


async def record_to_alert_history(
    alert_type: str,
    severity: str,
    message: str,
) -> None:
    """Insert a row into AlertHistory (used for rate limiting and heartbeats)."""
    from odds_core.database import async_session_maker
    from odds_core.models import AlertHistory

    async with async_session_maker() as session:
        session.add(
            AlertHistory(
                alert_type=alert_type,
                severity=severity,
                message=message,
                sent_at=datetime.now(UTC),
            )
        )
        await session.commit()


@asynccontextmanager
async def job_alert_context(job_name: str) -> AsyncIterator[None]:
    """Wrap a job body to standardise failure alerts and heartbeat recording.

    On exception: send a rate-limited CRITICAL alert, then re-raise.
    On clean exit: write a heartbeat row to AlertHistory.

    Usage::

        async with job_alert_context("fetch-oddsportal"):
            # job logic (excluding self-scheduling)
            ...
    """
    try:
        yield
    except Exception as e:
        alert_type = f"job_failure:{job_name}"
        if alert_manager.enabled and await check_rate_limit(alert_type):
            msg = f"🚨 Job {job_name} failed: {type(e).__name__}: {e}"
            await alert_manager.alert(msg, "critical")
            await record_to_alert_history(alert_type, "critical", msg)
        raise
    else:
        # Record heartbeat on success
        try:
            await record_to_alert_history(
                alert_type=f"heartbeat:{job_name}",
                severity="info",
                message=f"Job {job_name} completed successfully",
            )
        except Exception:
            logger.warning("heartbeat_record_failed", job=job_name, exc_info=True)


async def send_job_warning(alert_type: str, message: str) -> bool:
    """Send a rate-limited WARNING alert for a soft failure (e.g. empty scrape).

    Returns True if the alert was sent, False if disabled or rate-limited.
    """
    if not alert_manager.enabled:
        return False
    if not await check_rate_limit(alert_type):
        return False
    await alert_manager.alert(message, "warning")
    await record_to_alert_history(alert_type, "warning", message)
    return True
