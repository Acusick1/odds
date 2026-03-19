"""Database read operations for ESPN fixture data."""

from __future__ import annotations

import structlog
from odds_core.epl_data_models import EspnFixture
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnFixtureReader:
    """Handles all read operations for ESPN fixture data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_all_fixtures(self) -> list[EspnFixture]:
        """Load all ESPN fixtures ordered by date.

        Returns the full dataset for in-memory filtering by the feature group.
        """
        query = select(EspnFixture).order_by(EspnFixture.date)
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("espn_fixtures_loaded", count=len(rows))
        return rows

    async def get_fixtures_by_season(self, season: str) -> list[EspnFixture]:
        """Load ESPN fixtures for a specific season."""
        query = select(EspnFixture).where(EspnFixture.season == season).order_by(EspnFixture.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())

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
