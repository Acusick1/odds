"""Database read operations for FPL availability data."""

from __future__ import annotations

import structlog
from odds_core.epl_data_models import FplAvailability
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class FplAvailabilityReader:
    """Handles all read operations for FPL availability data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_all_availability(self) -> list[FplAvailability]:
        """Load all FPL availability records ordered by snapshot_time.

        Returns the full dataset for in-memory filtering by the feature group.
        """
        query = select(FplAvailability).order_by(
            FplAvailability.season, FplAvailability.gameweek, FplAvailability.snapshot_time
        )
        result = await self.session.execute(query)
        rows = list(result.scalars().all())
        logger.info("fpl_availability_loaded", count=len(rows))
        return rows

    async def get_availability_by_season(self, season: str) -> list[FplAvailability]:
        """Load FPL availability for a specific season."""
        query = (
            select(FplAvailability)
            .where(FplAvailability.season == season)
            .order_by(FplAvailability.gameweek, FplAvailability.snapshot_time)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_availability_for_gameweek(
        self, season: str, gameweek: int
    ) -> list[FplAvailability]:
        """Load FPL availability for a specific gameweek."""
        query = (
            select(FplAvailability)
            .where(
                FplAvailability.season == season,
                FplAvailability.gameweek == gameweek,
            )
            .order_by(FplAvailability.snapshot_time)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
