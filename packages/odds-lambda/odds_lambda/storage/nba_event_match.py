"""Match a team's NBA game on an Eastern calendar date to its stored event."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import structlog
from odds_core.models import Event
from odds_core.time import EASTERN
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

NBA_SPORT_KEY = "basketball_nba"

# NBA games tip off between about noon and 11 PM ET, so the window runs from 10:00 ET on the
# date to 06:00 ET the next day, past the UTC rollover of late tip-offs.
_WINDOW_START_ET_HOUR = 10
_WINDOW_LENGTH = timedelta(hours=20)


async def match_nba_event(
    session: AsyncSession, team_name: str, game_date: date, *, source: str
) -> int | None:
    """The id of the one NBA event ``team_name`` plays on the Eastern date ``game_date``.

    ``None`` when no event or more than one matches; an ambiguous match logs a warning
    naming ``source``.
    """
    day_start_et = datetime(
        game_date.year, game_date.month, game_date.day, _WINDOW_START_ET_HOUR, tzinfo=EASTERN
    )
    window_start = day_start_et.astimezone(UTC)
    window_end = (day_start_et + _WINDOW_LENGTH).astimezone(UTC)

    query = select(Event.id).where(
        and_(
            Event.sport_key == NBA_SPORT_KEY,
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            or_(Event.home_team == team_name, Event.away_team == team_name),
        )
    )
    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        logger.warning(
            "nba_event_ambiguous_match",
            source=source,
            team=team_name,
            game_date=str(game_date),
            candidates=len(candidates),
        )
    return None
