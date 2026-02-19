"""Database read operations for NBA injury report data."""

from __future__ import annotations

from datetime import date, datetime
from typing import TypedDict

import structlog
from odds_core.injury_models import InjuryReport
from sqlalchemy import and_, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class InjuryPipelineStats(TypedDict):
    """Aggregate statistics for the injury report pipeline."""

    total_reports: int
    unique_players: int
    events_matched: int
    events_unmatched: int
    earliest_game_date: date | None
    latest_game_date: date | None
    status_counts: dict[str, int]


class InjuryReader:
    """Handles all read operations for injury report data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_injuries_for_event(
        self,
        event_id: str,
        before_time: datetime | None = None,
    ) -> list[InjuryReport]:
        """Get injury reports linked to a specific event.

        Args:
            event_id: Sportsbook event ID.
            before_time: If set, only returns reports at or before this time
                to prevent look-ahead bias in backtesting.

        Returns:
            Injury reports ordered by report_time descending.
        """
        conditions = [InjuryReport.event_id == event_id]
        if before_time is not None:
            conditions.append(InjuryReport.report_time <= before_time)

        query = (
            select(InjuryReport).where(and_(*conditions)).order_by(InjuryReport.report_time.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_latest_report_for_event(
        self,
        event_id: str,
        as_of: datetime,
    ) -> list[InjuryReport]:
        """Get the most recent injury report snapshot for an event.

        Returns all player rows from the single most recent report_time
        that is at or before as_of. This gives the full injury picture
        at a point in time without look-ahead bias.

        Args:
            event_id: Sportsbook event ID.
            as_of: Cutoff time (UTC). Only reports at or before this time.

        Returns:
            All injury rows from the latest report snapshot, or empty list.
        """
        # Find the latest report_time for this event at or before as_of
        latest_time_query = select(func.max(InjuryReport.report_time)).where(
            and_(
                InjuryReport.event_id == event_id,
                InjuryReport.report_time <= as_of,
            )
        )
        result = await self.session.execute(latest_time_query)
        latest_time = result.scalar_one_or_none()

        if latest_time is None:
            return []

        # Fetch all rows at that report_time for this event
        query = (
            select(InjuryReport)
            .where(
                and_(
                    InjuryReport.event_id == event_id,
                    InjuryReport.report_time == latest_time,
                )
            )
            .order_by(InjuryReport.team, InjuryReport.player_name)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_pipeline_stats(self) -> InjuryPipelineStats:
        """Aggregate pipeline health statistics."""
        # Total reports
        total_result = await self.session.execute(select(func.count(InjuryReport.id)))
        total_reports = total_result.scalar_one()

        # Unique players
        players_result = await self.session.execute(
            select(func.count(distinct(InjuryReport.player_name)))
        )
        unique_players = players_result.scalar_one()

        # Events matched vs unmatched
        matched_result = await self.session.execute(
            select(func.count(distinct(InjuryReport.event_id))).where(
                InjuryReport.event_id.is_not(None)
            )
        )
        events_matched = matched_result.scalar_one()

        # Count distinct (team, game_date) pairs with no event_id
        unmatched_result = await self.session.execute(
            select(
                func.count(distinct(func.concat(InjuryReport.team, "-", InjuryReport.game_date)))
            ).where(InjuryReport.event_id.is_(None))
        )
        events_unmatched = unmatched_result.scalar_one()

        # Date coverage
        dates_result = await self.session.execute(
            select(
                func.min(InjuryReport.game_date),
                func.max(InjuryReport.game_date),
            )
        )
        date_row = dates_result.one()
        earliest_game_date = date_row[0]
        latest_game_date = date_row[1]

        # Status breakdown
        status_result = await self.session.execute(
            select(InjuryReport.status, func.count(InjuryReport.id)).group_by(InjuryReport.status)
        )
        status_counts = {str(row[0].value): row[1] for row in status_result.all()}

        return InjuryPipelineStats(
            total_reports=total_reports,
            unique_players=unique_players,
            events_matched=events_matched,
            events_unmatched=events_unmatched,
            earliest_game_date=earliest_game_date,
            latest_game_date=latest_game_date,
            status_counts=status_counts,
        )
