from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odds_core.models import Event, EventStatus
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.event_merge import SEASON_START_MONTH, season_start_year


def season_start(now: datetime) -> datetime:
    return datetime(season_start_year(now), SEASON_START_MONTH, 1, tzinfo=UTC)


async def find_unsettled_events(
    session: AsyncSession, sport_key: str, older_than: timedelta, now: datetime
) -> list[Event]:
    """SCHEDULED or LIVE events of the current season that kicked off over ``older_than`` ago.

    CANCELLED and POSTPONED events are not pending.
    """
    result = await session.execute(
        select(Event)
        .where(
            Event.sport_key == sport_key,
            Event.status.in_((EventStatus.SCHEDULED, EventStatus.LIVE)),  # type: ignore[attr-defined]
            Event.commence_time < now - older_than,
            Event.commence_time >= season_start(now),
        )
        .order_by(Event.commence_time, Event.id)
    )
    return list(result.scalars().all())
