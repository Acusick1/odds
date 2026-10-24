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
# Recorded alerts and heartbeat recording
# ---------------------------------------------------------------------------

JOB_WARNING_INTERVAL = timedelta(minutes=30)


async def record_to_alert_history(
    alert_type: str,
    severity: str,
    message: str,
    *,
    context: dict | None = None,
    session: AsyncSession | None = None,
) -> None:
    """Insert a row into AlertHistory (the dedup ledger and heartbeat log).

    Uses ``session`` when given, else a fresh one. A no-op under ``SCHEDULER_DRY_RUN``
    so validation runs leave the ledger untouched.
    """
    if get_settings().scheduler.dry_run:
        return

    from odds_core.database import async_session_maker
    from odds_core.models import AlertHistory

    row = AlertHistory(
        alert_type=alert_type,
        severity=severity,
        message=message,
        context=context,
        sent_at=datetime.now(UTC),
    )
    if session is not None:
        session.add(row)
        await session.commit()
        return
    async with async_session_maker() as own_session:
        own_session.add(row)
        await own_session.commit()


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


async def open_alerted_at(
    alert_type: str, reset_by: str | None = None, session: AsyncSession | None = None
) -> datetime | None:
    """When ``alert_type`` last fired, unless a newer ``reset_by`` row closed it; else None."""
    alerted = await latest_sent_at(alert_type, session)
    if alerted is None or reset_by is None:
        return alerted
    reset = await latest_sent_at(reset_by, session)
    if reset is not None and reset > alerted:
        return None
    return alerted


async def job_outage_alerted_at(
    job_name: str, alert_type: str | None = None, session: AsyncSession | None = None
) -> datetime | None:
    """When the open outage of ``job_name`` last raised ``alert_type``, else None.

    An outage runs from a job's first failure to its next ``heartbeat:<job_name>``;
    an alert older than the job's latest heartbeat belongs to a closed outage.
    ``alert_type`` defaults to ``job_failure:<job_name>``.
    """
    return await open_alerted_at(
        alert_type or f"job_failure:{job_name}", f"heartbeat:{job_name}", session
    )


def stale_reminder_interval() -> timedelta:
    return timedelta(minutes=get_settings().alerts.stale_reminder_minutes)


async def send_recorded_alert(
    alert_type: str,
    severity: str,
    message: str,
    *,
    every: timedelta,
    reset_by: str | None = None,
    context: dict | None = None,
    session: AsyncSession | None = None,
    now: datetime | None = None,
) -> bool:
    """Send ``message`` and record it, unless ``alert_type`` was recorded within ``every``.

    The one dedup rule for every alert that repeats while its condition holds: the first
    occurrence sends, later ones only once ``every`` has passed since the newest row.
    A ``reset_by`` row newer than the newest ``alert_type`` row starts a new condition
    (a job's heartbeat ends its outage). The send is bounded by
    ``ALERT_SEND_TIMEOUT_SECONDS``; failures are logged, never raised.

    Returns True if sent, False if undeliverable, not yet due, or the send failed.
    """
    if not alert_manager.enabled:
        return False
    try:
        alerted = await open_alerted_at(alert_type, reset_by, session)
        if alerted is not None and (now or datetime.now(UTC)) - alerted < every:
            return False
        async with asyncio.timeout(ALERT_SEND_TIMEOUT_SECONDS):
            await alert_manager.alert(message, severity)
        await record_to_alert_history(
            alert_type, severity, message, context=context, session=session
        )
    except Exception:
        logger.error("recorded_alert_failed", alert_type=alert_type, exc_info=True)
        if session is not None:
            await session.rollback()
        return False
    return True


async def send_job_failure(job_name: str, error: Exception, *, now: datetime | None = None) -> None:
    """Send the CRITICAL ``job_failure:<job_name>`` alert once per outage."""
    await send_outage_warning(
        job_name,
        f"job_failure:{job_name}",
        f"🚨 Job {job_name} failed: {type(error).__name__}: {error}",
        severity="critical",
        now=now,
    )


async def send_outage_warning(
    job_name: str,
    alert_type: str,
    message: str,
    *,
    severity: str = "warning",
    now: datetime | None = None,
) -> bool:
    """Send ``alert_type`` once per outage of ``job_name``, reminding every stale interval."""
    return await send_recorded_alert(
        alert_type,
        severity,
        message,
        every=stale_reminder_interval(),
        reset_by=f"heartbeat:{job_name}",
        now=now,
    )


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
    """Send a WARNING for a soft failure (e.g. empty scrape), at most every 30 minutes."""
    return await send_recorded_alert(alert_type, "warning", message, every=JOB_WARNING_INTERVAL)
