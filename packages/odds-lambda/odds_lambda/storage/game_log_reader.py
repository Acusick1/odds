"""Database read operations for NBA team game log data."""

from __future__ import annotations

from datetime import date
from typing import TypedDict

import structlog
from odds_core.game_log_models import NbaTeamGameLog
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class GameLogPipelineStats(TypedDict):
    """Aggregate statistics for the game log pipeline."""

    total_rows: int
    season_counts: dict[str, int]
    events_matched: int
    events_unmatched: int
    earliest_game_date: date | None
    latest_game_date: date | None


class GameLogReader:
    """Handles all read operations for game log data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_game_logs_for_event(self, event_id: str) -> list[NbaTeamGameLog]:
        """Get game log rows linked to a specific event (both teams).

        Returns:
            Game log rows ordered by team_abbreviation.
        """
        query = (
            select(NbaTeamGameLog)
            .where(NbaTeamGameLog.event_id == event_id)
            .order_by(NbaTeamGameLog.team_abbreviation)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_pipeline_stats(self) -> GameLogPipelineStats:
        """Aggregate pipeline health statistics."""
        # Total rows
        total_result = await self.session.execute(select(func.count(NbaTeamGameLog.id)))
        total_rows = total_result.scalar_one()

        # Per-season counts
        season_result = await self.session.execute(
            select(NbaTeamGameLog.season, func.count(NbaTeamGameLog.id))
            .group_by(NbaTeamGameLog.season)
            .order_by(NbaTeamGameLog.season)
        )
        season_counts = {str(row[0]): row[1] for row in season_result.all()}

        # Events matched (distinct event_ids that are not null)
        matched_result = await self.session.execute(
            select(func.count(distinct(NbaTeamGameLog.event_id))).where(
                NbaTeamGameLog.event_id.is_not(None)
            )
        )
        events_matched = matched_result.scalar_one()

        # Unmatched: distinct (nba_game_id) where event_id is null
        # Each game has 2 rows; count distinct games not game-team pairs
        unmatched_result = await self.session.execute(
            select(func.count(distinct(NbaTeamGameLog.nba_game_id))).where(
                NbaTeamGameLog.event_id.is_(None)
            )
        )
        events_unmatched = unmatched_result.scalar_one()

        # Date coverage
        dates_result = await self.session.execute(
            select(
                func.min(NbaTeamGameLog.game_date),
                func.max(NbaTeamGameLog.game_date),
            )
        )
        date_row = dates_result.one()
        earliest_game_date = date_row[0]
        latest_game_date = date_row[1]

        return GameLogPipelineStats(
            total_rows=total_rows,
            season_counts=season_counts,
            events_matched=events_matched,
            events_unmatched=events_unmatched,
            earliest_game_date=earliest_game_date,
            latest_game_date=latest_game_date,
        )
