"""Shared scheduling helpers used by multiple jobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog

from odds_lambda.scheduling.backends import get_scheduler_backend

logger = structlog.get_logger()


async def get_next_kickoff(sport_key: str, *, now: datetime | None = None) -> datetime | None:
    """Earliest scheduled event commence_time for a sport after ``now``."""

    from odds_core.database import async_session_maker
    from odds_core.models import Event, EventStatus
    from sqlalchemy import select
    from sqlmodel import col

    now = now or datetime.now(UTC)

    async with async_session_maker() as session:
        result = await session.execute(
            select(Event.commence_time)
            .where(
                col(Event.sport_key) == sport_key,
                col(Event.commence_time) > now,
                col(Event.status) == EventStatus.SCHEDULED,
            )
            .order_by(col(Event.commence_time))
            .limit(1)
        )
        return result.scalar_one_or_none()


async def within_lead(
    sport_key: str,
    lead_days: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Whether the sport's next fixture is within ``lead_days`` of ``now``.

    Cheap DB-only season gate for cron-driven forward jobs that have no
    self-scheduling decision of their own (``daily_digest``,
    ``fetch_mlb_probables``). Returns ``False`` in the offseason (no fixture, or
    next fixture beyond the lead) so the caller can early-return before doing
    expensive work; the cron re-fires and re-gates on the next tick.
    """
    now = now or datetime.now(UTC)
    next_kickoff = await get_next_kickoff(sport_key, now=now)
    if next_kickoff is None:
        return False
    return next_kickoff <= now + timedelta(days=lead_days)


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
