"""Shared scheduling helpers used by multiple jobs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
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


SLATE_GAP = timedelta(hours=6)


def slates(kickoffs: Iterable[datetime], *, gap: timedelta = SLATE_GAP) -> list[list[datetime]]:
    """``kickoffs`` in time order, split where two are ``gap`` or more apart.

    A slate is a run of kickoffs each less than ``gap`` after the one before; an NBA night's
    games are one slate.
    """
    result: list[list[datetime]] = []
    for kickoff in sorted(kickoffs):
        if result and kickoff - result[-1][-1] < gap:
            result[-1].append(kickoff)
        else:
            result.append([kickoff])
    return result


def next_slate_start(
    kickoffs: Iterable[datetime], *, now: datetime, gap: timedelta = SLATE_GAP
) -> datetime | None:
    """First kickoff of the first slate of ``kickoffs`` that starts after ``now``.

    Games already under way belong to the slate they started, so a later game of the same
    night never starts a slate.
    """
    return next((slate[0] for slate in slates(kickoffs, gap=gap) if slate[0] > now), None)


def main_tip(slate: Sequence[datetime]) -> datetime:
    """The kickoff the most games of ``slate`` share, the latest of any tie."""
    counts = Counter(slate)
    return max(counts, key=lambda kickoff: (counts[kickoff], kickoff))


def next_slate_main_tip(
    kickoffs: Iterable[datetime],
    *,
    now: datetime,
    gap: timedelta = SLATE_GAP,
    min_after_start: timedelta = timedelta(minutes=90),
) -> datetime | None:
    """``main_tip`` of the first slate whose main tip is after ``now`` and at least
    ``min_after_start`` after the slate's first kickoff.

    A main tip closer to the first kickoff than that adds no run the slate's first-tip leads
    do not already make, so such a slate has none.
    """
    for slate in slates(kickoffs, gap=gap):
        tip = main_tip(slate)
        if tip > now and tip - slate[0] >= min_after_start:
            return tip
    return None


async def _stored_kickoffs(
    sport_key: str, *, now: datetime, gap: timedelta, horizon: timedelta
) -> list[datetime]:
    """Kickoffs of the sport's stored events from ``gap`` before ``now`` to ``horizon`` after it,
    excluding cancelled and postponed ones."""

    from odds_core.database import async_session_maker
    from odds_core.models import Event, EventStatus
    from sqlalchemy import select
    from sqlmodel import col

    async with async_session_maker() as session:
        result = await session.execute(
            select(Event.commence_time).where(
                col(Event.sport_key) == sport_key,
                col(Event.commence_time) > now - gap,
                col(Event.commence_time) <= now + horizon,
                col(Event.status).notin_([EventStatus.CANCELLED, EventStatus.POSTPONED]),
            )
        )
        return list(result.scalars())


async def get_next_slate_start(
    sport_key: str,
    *,
    now: datetime | None = None,
    gap: timedelta = SLATE_GAP,
    horizon: timedelta = timedelta(days=14),
) -> datetime | None:
    """``next_slate_start`` over the sport's stored events."""
    now = now or datetime.now(UTC)
    kickoffs = await _stored_kickoffs(sport_key, now=now, gap=gap, horizon=horizon)
    return next_slate_start(kickoffs, now=now, gap=gap)


async def get_next_slate_main_tip(
    sport_key: str,
    *,
    now: datetime | None = None,
    gap: timedelta = SLATE_GAP,
    horizon: timedelta = timedelta(days=14),
) -> datetime | None:
    """``next_slate_main_tip`` over the sport's stored events."""
    now = now or datetime.now(UTC)
    kickoffs = await _stored_kickoffs(sport_key, now=now, gap=gap, horizon=horizon)
    return next_slate_main_tip(kickoffs, now=now, gap=gap)


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
