"""Shared scheduling helpers used by multiple jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog

from odds_lambda.fetch_tier import FetchTier
from odds_lambda.scheduling.backends import get_scheduler_backend

logger = structlog.get_logger()


@dataclass(frozen=True)
class ProximitySchedule:
    """Per-job proximity-based scheduling intervals (hours).

    The boundary thresholds come from ``FetchTier`` so all jobs agree on what
    "closing" / "pregame" / "far" mean; only the polling cadence varies per
    job (e.g. cheap API polls more aggressively than browser-driven scrapes).
    """

    closing: float
    pregame: float
    far: float
    db_fallback: float = 1.0

    def interval_for(
        self,
        next_kickoff: datetime | None,
        *,
        now: datetime | None = None,
    ) -> float:
        """Compute the next-fire interval (hours) from proximity to kickoff."""
        if next_kickoff is None:
            return self.far
        if now is None:
            now = datetime.now(UTC)
        hours_until = (next_kickoff - now).total_seconds() / 3600
        if hours_until < FetchTier.CLOSING.max_hours:
            return self.closing
        if hours_until < FetchTier.PREGAME.max_hours:
            return self.pregame
        return self.far


async def get_next_kickoff(sport_key: str) -> datetime | None:
    """Earliest scheduled event commence_time for a sport."""
    from datetime import UTC

    from odds_core.database import async_session_maker
    from odds_core.models import Event, EventStatus
    from sqlalchemy import select
    from sqlmodel import col

    async with async_session_maker() as session:
        result = await session.execute(
            select(Event.commence_time)
            .where(
                col(Event.sport_key) == sport_key,
                col(Event.commence_time) > datetime.now(UTC),
                col(Event.status) == EventStatus.SCHEDULED,
            )
            .order_by(col(Event.commence_time))
            .limit(1)
        )
        return result.scalar_one_or_none()


def apply_overnight_skip(
    next_time: datetime,
    *,
    overnight_start_utc: int = 22,
    overnight_resume_utc: int = 6,
) -> datetime:
    """Push next_time to resume hour if it falls in the overnight window."""
    if overnight_start_utc > overnight_resume_utc:
        # Window wraps midnight (e.g. 22:00-06:00)
        is_overnight = (
            next_time.hour >= overnight_start_utc or next_time.hour < overnight_resume_utc
        )
    else:
        # Window within same day (e.g. 05:00-14:00)
        is_overnight = overnight_start_utc <= next_time.hour < overnight_resume_utc

    if is_overnight:
        resume = next_time.replace(hour=overnight_resume_utc, minute=0, second=0, microsecond=0)
        if resume <= next_time:
            resume += timedelta(days=1)
        return resume

    return next_time


async def self_schedule(
    *,
    job_name: str,
    next_time: datetime,
    dry_run: bool,
    sport: str | None = None,
    interval_hours: float | None = None,
    reason: str = "",
) -> None:
    """Schedule the next execution via the scheduler backend."""
    backend = get_scheduler_backend(dry_run=dry_run)

    payload: dict[str, object] = {}
    if sport:
        payload["sport"] = sport

    await backend.schedule_next_execution(
        job_name=job_name,
        next_time=next_time,
        payload=payload or None,
    )

    logger.info(
        "job_next_scheduled",
        job_name=job_name,
        next_time=next_time.isoformat(),
        interval_hours=interval_hours,
        reason=reason,
        backend=backend.get_backend_name(),
    )
