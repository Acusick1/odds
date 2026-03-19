"""Database read operations for ESPN lineup data."""

from __future__ import annotations

import structlog
from odds_core.epl_data_models import EspnLineup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnLineupReader:
    """Handles all read operations for ESPN lineup data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_all_lineups(self) -> list[EspnLineup]:
        """Load all ESPN lineups ordered by date.

        Returns the full dataset for in-memory filtering by the feature group.
        """
        query = select(EspnLineup).order_by(EspnLineup.date)
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("espn_lineups_loaded", count=len(rows))
        return rows

    async def get_starters(self) -> list[EspnLineup]:
        """Load only starting XI entries, ordered by date.

        This is the primary query for lineup-delta feature extraction.
        """
        query = select(EspnLineup).where(EspnLineup.starter.is_(True)).order_by(EspnLineup.date)
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("espn_lineup_starters_loaded", count=len(rows))
        return rows

    async def get_lineups_by_season(self, season: str) -> list[EspnLineup]:
        """Load ESPN lineups for a specific season."""
        query = select(EspnLineup).where(EspnLineup.season == season).order_by(EspnLineup.date)
        result = await self.session.execute(query)
        return list(result.scalars().all())
