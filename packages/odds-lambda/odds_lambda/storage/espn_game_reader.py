"""Database read operations for ESPN game data."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from odds_core.epl_data_models import ESPN_STATUS_POSTPONED, EspnGame
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnGameReader:
    """Handles all read operations for ESPN game data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_all_games(self, until: datetime | None = None) -> list[EspnGame]:
        """Load all ESPN games ordered by date, up to ``until`` when it is given.

        Returns the full dataset for in-memory filtering by the feature group.
        """
        query = select(EspnGame).order_by(EspnGame.date)
        if until is not None:
            query = query.where(EspnGame.date <= until)
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("espn_games_loaded", count=len(rows))
        return rows

    async def get_games_by_season(self, season: str) -> list[EspnGame]:
        """Load ESPN games for a specific season."""
        query = select(EspnGame).where(EspnGame.season == season).order_by(EspnGame.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_postponed_home_games(self, competition: str) -> list[EspnGame]:
        """The home-side row of each postponed match in ``competition``, at its original kick-off."""
        query = (
            select(EspnGame)
            .where(
                EspnGame.competition == competition,
                EspnGame.status == ESPN_STATUS_POSTPONED,
                EspnGame.home_away == "home",
            )
            .order_by(EspnGame.date)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def has_unpostponed_game(
        self, competition: str, home_team: str, away_team: str, around: datetime, window: timedelta
    ) -> bool:
        """Whether ESPN lists the pairing within ``window`` of ``around`` with a status other than postponed."""
        query = (
            select(EspnGame.id)
            .where(
                EspnGame.competition == competition,
                EspnGame.team == home_team,
                EspnGame.opponent == away_team,
                EspnGame.home_away == "home",
                EspnGame.status != ESPN_STATUS_POSTPONED,
                EspnGame.date >= around - window,
                EspnGame.date <= around + window,
            )
            .limit(1)
        )
        return (await self.session.execute(query)).first() is not None

    async def get_games_for_team(self, team: str, season: str | None = None) -> list[EspnGame]:
        """Load ESPN games for a specific team, optionally filtered by season."""
        conditions = [EspnGame.team == team]
        if season is not None:
            conditions.append(EspnGame.season == season)

        query = select(EspnGame).where(*conditions).order_by(EspnGame.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())
