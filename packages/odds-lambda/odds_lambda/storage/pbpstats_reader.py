"""Database read operations for PBPStats player season statistics."""

from __future__ import annotations

from typing import TypedDict

import structlog
from odds_core.player_stats_models import NbaPlayerSeasonStats
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class PlayerStatsPipelineStats(TypedDict):
    """Aggregate statistics for the player stats pipeline."""

    total_rows: int
    season_counts: dict[str, int]
    unique_players: int


class PbpStatsReader:
    """Handles all read operations for player season stats."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_player_stats(self, player_name: str, season: str) -> NbaPlayerSeasonStats | None:
        """Look up a player's season stats by name and season."""
        query = select(NbaPlayerSeasonStats).where(
            NbaPlayerSeasonStats.player_name == player_name,
            NbaPlayerSeasonStats.season == season,
        )
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_team_players(
        self, team_abbreviation: str, season: str
    ) -> list[NbaPlayerSeasonStats]:
        """Get all players for a team in a season, ordered by minutes descending."""
        query = (
            select(NbaPlayerSeasonStats)
            .where(
                NbaPlayerSeasonStats.team_abbreviation == team_abbreviation,
                NbaPlayerSeasonStats.season == season,
            )
            .order_by(NbaPlayerSeasonStats.minutes.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_pipeline_stats(self) -> PlayerStatsPipelineStats:
        """Aggregate pipeline health statistics."""
        total_result = await self.session.execute(select(func.count(NbaPlayerSeasonStats.id)))
        total_rows = total_result.scalar_one()

        season_result = await self.session.execute(
            select(NbaPlayerSeasonStats.season, func.count(NbaPlayerSeasonStats.id))
            .group_by(NbaPlayerSeasonStats.season)
            .order_by(NbaPlayerSeasonStats.season)
        )
        season_counts = {str(row[0]): row[1] for row in season_result.all()}

        unique_result = await self.session.execute(
            select(func.count(distinct(NbaPlayerSeasonStats.player_id)))
        )
        unique_players = unique_result.scalar_one()

        return PlayerStatsPipelineStats(
            total_rows=total_rows,
            season_counts=season_counts,
            unique_players=unique_players,
        )
