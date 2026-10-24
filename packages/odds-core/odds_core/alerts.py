"""Alert infrastructure: Discord webhooks, job failure alerts, and heartbeat recording."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import aiohttp
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from odds_core.config import Settings, get_settings

logger = structlog.get_logger()

ALERT_SEND_TIMEOUT_SECONDS = 30


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
        self.enabled = self.config.alerts_deliverable
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
    from sqlalchemy import func, select

    from odds_core.database import async_session_maker
    from odds_core.models import AlertHistory

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
    """Insert a row into AlertHistory (used for rate limiting and heartbeats).

    A no-op under ``SCHEDULER_DRY_RUN`` so validation runs leave the ledger untouched.
    """
    if get_settings().scheduler.dry_run:
        return

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


async def latest_sent_at(alert_type: str, session: AsyncSession | None = None) -> datetime | None:
    """Newest ``AlertHistory.sent_at`` of ``alert_type``, on ``session`` or a fresh one."""
    from sqlalchemy import func, select

    from odds_core.database import async_session_maker
    from odds_core.models import AlertHistory

    query = select(func.max(AlertHistory.sent_at)).where(AlertHistory.alert_type == alert_type)
    if session is not None:
        return (await session.execute(query)).scalar_one_or_none()
    async with async_session_maker() as own_session:
        return (await own_session.execute(query)).scalar_one_or_none()


async def job_outage_alerted_at(
    job_name: str, alert_type: str | None = None, session: AsyncSession | None = None
) -> datetime | None:
    """When the open outage of ``job_name`` last raised ``alert_type``, else None.

    An outage runs from a job's first failure to its next ``heartbeat:<job_name>``;
    an alert older than the job's latest heartbeat belongs to a closed outage.
    ``alert_type`` defaults to ``job_failure:<job_name>``.
    """
    alerted = await latest_sent_at(alert_type or f"job_failure:{job_name}", session)
    if alerted is None:
        return None
    heartbeat = await latest_sent_at(f"heartbeat:{job_name}", session)
    if heartbeat is not None and heartbeat > alerted:
        return None
    return alerted


async def outage_alert_due(job_name: str, alert_type: str, now: datetime) -> bool:
    """Whether ``alert_type`` should be raised for ``job_name`` at ``now``.

    It is due on the first occurrence in an outage and again every
    ``AlertConfig.stale_reminder_minutes`` while the outage lasts.
    """
    alerted = await job_outage_alerted_at(job_name, alert_type)
    if alerted is None:
        return True
    return now - alerted >= timedelta(minutes=get_settings().alerts.stale_reminder_minutes)


async def send_job_failure(job_name: str, error: Exception, *, now: datetime | None = None) -> None:
    """Send the CRITICAL ``job_failure:<job_name>`` alert once per outage and record it."""
    alert_type = f"job_failure:{job_name}"
    if not alert_manager.enabled:
        return
    if await outage_alert_due(job_name, alert_type, now or datetime.now(UTC)):
        msg = f"🚨 Job {job_name} failed: {type(error).__name__}: {error}"
        await alert_manager.alert(msg, "critical")
        await record_to_alert_history(alert_type, "critical", msg)


async def send_outage_warning(
    job_name: str, alert_type: str, message: str, *, now: datetime | None = None
) -> bool:
    """Send a WARNING once per outage of ``job_name`` and record it; True if sent."""
    if not alert_manager.enabled:
        return False
    if not await outage_alert_due(job_name, alert_type, now or datetime.now(UTC)):
        return False
    await alert_manager.alert(message, "warning")
    await record_to_alert_history(alert_type, "warning", message)
    return True


@asynccontextmanager
async def job_alert_context(job_name: str) -> AsyncIterator[None]:
    """Wrap a job body to standardise failure alerts and heartbeat recording.

    On exception: send a CRITICAL alert once per outage, then re-raise.
    On clean exit: write a heartbeat row to AlertHistory.

    Usage::

        async with job_alert_context("fetch-oddsportal"):
            # job logic (excluding self-scheduling)
            ...
    """
    try:
        yield
    except Exception as e:
        try:
            async with asyncio.timeout(ALERT_SEND_TIMEOUT_SECONDS):
                await send_job_failure(job_name, e)
        except Exception:
            logger.error("job_failure_alert_failed", job=job_name, exc_info=True)
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
