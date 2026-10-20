"""Database read operations for ESPN fixture data."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from odds_core.epl_data_models import ESPN_STATUS_POSTPONED, EspnFixture
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnFixtureReader:
    """Handles all read operations for ESPN fixture data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_all_fixtures(self, until: datetime | None = None) -> list[EspnFixture]:
        """Load all ESPN fixtures ordered by date, up to ``until`` when it is given.

        Returns the full dataset for in-memory filtering by the feature group.
        """
        query = select(EspnFixture).order_by(EspnFixture.date)
        if until is not None:
            query = query.where(EspnFixture.date <= until)
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("espn_fixtures_loaded", count=len(rows))
        return rows

    async def get_fixtures_by_season(self, season: str) -> list[EspnFixture]:
        """Load ESPN fixtures for a specific season."""
        query = select(EspnFixture).where(EspnFixture.season == season).order_by(EspnFixture.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_postponed_home_fixtures(self, competition: str) -> list[EspnFixture]:
        """The home-side row of each postponed match in ``competition``, at its original kick-off."""
        query = (
            select(EspnFixture)
            .where(
                EspnFixture.competition == competition,
                EspnFixture.status == ESPN_STATUS_POSTPONED,
                EspnFixture.home_away == "home",
            )
            .order_by(EspnFixture.date)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def has_unpostponed_fixture(
        self, competition: str, home_team: str, away_team: str, around: datetime, window: timedelta
    ) -> bool:
        """Whether ESPN lists the pairing within ``window`` of ``around`` with a status other than postponed."""
        query = (
            select(EspnFixture.id)
            .where(
                EspnFixture.competition == competition,
                EspnFixture.team == home_team,
                EspnFixture.opponent == away_team,
                EspnFixture.home_away == "home",
                EspnFixture.status != ESPN_STATUS_POSTPONED,
                EspnFixture.date >= around - window,
                EspnFixture.date <= around + window,
            )
            .limit(1)
        )
        return (await self.session.execute(query)).first() is not None

    async def get_fixtures_for_team(
        self, team: str, season: str | None = None
    ) -> list[EspnFixture]:
        """Load ESPN fixtures for a specific team, optionally filtered by season."""
        conditions = [EspnFixture.team == team]
        if season is not None:
            conditions.append(EspnFixture.season == season)

        query = select(EspnFixture).where(*conditions).order_by(EspnFixture.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())
