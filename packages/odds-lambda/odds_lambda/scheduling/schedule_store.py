"""Read the local scheduler's schedules straight from APScheduler's tables.

Starting a scheduler to ask it (``LocalSchedulerBackend.get_scheduled_jobs``) writes the data
store's metadata, so a reader that must not write reads the tables instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class StoredSchedule:
    """One schedule in the data store; ``running`` counts its jobs a worker has acquired."""

    job_name: str
    next_fire_time: datetime | None
    last_fire_time: datetime | None
    paused: bool
    running: int


async def read_schedules(session: AsyncSession) -> list[StoredSchedule]:
    """Every schedule, by job name; none when the scheduler has never booted on the database
    (APScheduler creates its tables on first start)."""
    if await session.scalar(text("SELECT to_regclass('schedules')")) is None:
        return []
    rows = await session.execute(
        text(
            "SELECT s.id, s.next_fire_time, s.last_fire_time, s.paused, "
            "count(j.id) FILTER (WHERE j.acquired_by IS NOT NULL) AS running "
            "FROM schedules s LEFT JOIN jobs j ON j.schedule_id = s.id "
            "GROUP BY s.id ORDER BY s.id"
        )
    )
    return [StoredSchedule(*row) for row in rows.all()]
