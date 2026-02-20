"""Database write operations for PBPStats player season statistics."""

from __future__ import annotations

import structlog
from odds_core.player_stats_models import NbaPlayerSeasonStats
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.pbpstats_fetcher import PlayerSeasonRecord

logger = structlog.get_logger(__name__)


class PbpStatsWriter:
    """Handles all write operations for player season stats."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_player_stats(self, records: list[PlayerSeasonRecord]) -> int:
        """Insert or update player season stats.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        for idempotent upserts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        stat_dicts: list[dict] = [
            {
                "player_id": r.player_id,
                "player_name": r.player_name,
                "team_id": r.team_id,
                "team_abbreviation": r.team_abbreviation,
                "season": r.season,
                "minutes": r.minutes,
                "games_played": r.games_played,
                "on_off_rtg": r.on_off_rtg,
                "on_def_rtg": r.on_def_rtg,
                "usage": r.usage,
                "ts_pct": r.ts_pct,
                "efg_pct": r.efg_pct,
                "assists": r.assists,
                "turnovers": r.turnovers,
                "rebounds": r.rebounds,
                "steals": r.steals,
                "blocks": r.blocks,
                "points": r.points,
                "plus_minus": r.plus_minus,
            }
            for r in records
        ]

        batch_size = 1000
        for i in range(0, len(stat_dicts), batch_size):
            batch = stat_dicts[i : i + batch_size]
            stmt = insert(NbaPlayerSeasonStats).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in NbaPlayerSeasonStats.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_player_season_stats_player_season",
                set_=set_,
            )
            await self.session.execute(stmt)
        await self.session.flush()

        logger.info("player_stats_upserted", count=len(stat_dicts))
        return len(stat_dicts)
