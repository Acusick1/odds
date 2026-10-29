"""Database read operations for odds data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from odds_core.models import (
    DataQualityLog,
    Event,
    EventStatus,
    FetchLog,
    OddsSnapshot,
    Price,
)
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class OddsReader:
    """Handles all read operations from the database."""

    def __init__(self, session: AsyncSession):
        """
        Initialize reader with database session.

        Args:
            session: Async database session
        """
        self.session = session

    async def get_event_by_id(self, event_id: int) -> Event | None:
        """
        Get event by ID.

        Args:
            event_id: Event identifier

        Returns:
            Event or None if not found
        """
        result = await self.session.execute(select(Event).where(Event.id == event_id))
        return result.scalar_one_or_none()

    async def get_events_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
        status: EventStatus | None = None,
        min_snapshots: int | None = None,
    ) -> list[Event]:
        """
        Get events within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            sport_key: Filter by sport (optional)
            status: Filter by status (optional)
            min_snapshots: Minimum number of distinct snapshot times required per event

        Returns:
            Event records ordered by commence time, then event ID
        """
        query = select(Event).where(
            and_(
                Event.commence_time >= start_date,
                Event.commence_time <= end_date,
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        if status:
            query = query.where(Event.status == status)

        if min_snapshots is not None:
            snapshot_count = (
                select(OddsSnapshot.event_id)
                .where(OddsSnapshot.event_id == Event.id)
                .correlate(Event)
                .group_by(OddsSnapshot.event_id)
                .having(func.count(func.distinct(OddsSnapshot.snapshot_time)) >= min_snapshots)
                .exists()
            )
            query = query.where(snapshot_count)

        # The event ID breaks kick-off ties so walk-forward folds do not depend on the query plan.
        query = query.order_by(Event.commence_time, Event.id)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_events_by_team(
        self,
        team_name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Event]:
        """
        Get events for a specific team.

        Args:
            team_name: Team name (partial match)
            start_date: Start of range (optional)
            end_date: End of range (optional)

        Returns:
            List of Event records
        """
        query = select(Event).where(
            (Event.home_team.ilike(f"%{team_name}%")) | (Event.away_team.ilike(f"%{team_name}%"))
        )

        if start_date:
            query = query.where(Event.commence_time >= start_date)

        if end_date:
            query = query.where(Event.commence_time <= end_date)

        query = query.order_by(Event.commence_time.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def snapshot_exists(
        self, event_id: int, snapshot_time: datetime, tolerance_minutes: int = 5
    ) -> bool:
        """
        Check if a snapshot exists for an event at a specific time.

        Args:
            event_id: Event identifier
            snapshot_time: Target snapshot time
            tolerance_minutes: Tolerance window in minutes

        Returns:
            True if snapshot exists within tolerance window
        """
        time_lower = snapshot_time - timedelta(minutes=tolerance_minutes)
        time_upper = snapshot_time + timedelta(minutes=tolerance_minutes)

        query = select(func.count(OddsSnapshot.id)).where(
            and_(
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.snapshot_time >= time_lower,
                OddsSnapshot.snapshot_time <= time_upper,
            )
        )

        result = await self.session.execute(query)
        count = result.scalar_one()
        return count > 0

    async def get_fetch_logs(
        self,
        sport_key: str | None = None,
        start_time: datetime | None = None,
        limit: int = 100,
        with_quota: bool = False,
    ) -> list[FetchLog]:
        """
        Get fetch operation logs.

        Args:
            sport_key: Filter by sport (optional)
            start_time: Only logs after this time (optional)
            limit: Maximum number of records
            with_quota: Only rows with a non-null api_quota_remaining

        Returns:
            List of FetchLog records
        """
        query = select(FetchLog)

        if sport_key:
            query = query.where(FetchLog.sport_key == sport_key)

        if start_time:
            query = query.where(FetchLog.fetch_time >= start_time)

        if with_quota:
            query = query.where(FetchLog.api_quota_remaining.is_not(None))

        query = query.order_by(FetchLog.fetch_time.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_data_quality_logs(
        self,
        event_id: int | None = None,
        severity: str | None = None,
        start_time: datetime | None = None,
        limit: int = 100,
    ) -> list[DataQualityLog]:
        """
        Get data quality issue logs.

        Args:
            event_id: Filter by event (optional)
            severity: Filter by severity (optional)
            start_time: Only logs after this time (optional)
            limit: Maximum number of records

        Returns:
            List of DataQualityLog records
        """
        query = select(DataQualityLog)

        if event_id:
            query = query.where(DataQualityLog.event_id == event_id)

        if severity:
            query = query.where(DataQualityLog.severity == severity)

        if start_time:
            query = query.where(DataQualityLog.created_at >= start_time)

        query = query.order_by(DataQualityLog.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_database_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with various statistics
        """
        # Count events by status
        events_query = select(Event.status, func.count(Event.id)).group_by(Event.status)
        events_result = await self.session.execute(events_query)
        events_by_status = {row[0].value: row[1] for row in events_result}

        price_rows = (await self.session.execute(select(func.count(Price.id)))).scalar_one()

        # Total snapshot times per event: a time holds one snapshot per market
        snapshots_count_query = select(func.count()).select_from(
            select(OddsSnapshot.event_id, OddsSnapshot.snapshot_time).distinct().subquery()
        )
        snapshots_result = await self.session.execute(snapshots_count_query)
        total_snapshots = snapshots_result.scalar_one()

        # Most recent API quota
        latest_quota_query = (
            select(FetchLog.api_quota_remaining)
            .where(FetchLog.api_quota_remaining.is_not(None))
            .order_by(FetchLog.fetch_time.desc())
            .limit(1)
        )
        quota_result = await self.session.execute(latest_quota_query)
        latest_quota = quota_result.scalar_one_or_none()

        return {
            "events_by_status": events_by_status,
            "total_events": sum(events_by_status.values()),
            "total_price_rows": price_rows,
            "total_snapshots": total_snapshots,
            "api_quota_remaining": latest_quota,
        }

    async def get_snapshot_count_for_event(self, event_id: int) -> int:
        """
        Get total count of odds snapshots for an event.

        Args:
            event_id: Event identifier

        Returns:
            Number of distinct snapshot times for this event (a time holds one snapshot per market)

        Example:
            reader = OddsReader(session)
            count = await reader.get_snapshot_count_for_event("abc123")
            print(f"Total snapshots: {count}")
        """
        query = select(func.count(func.distinct(OddsSnapshot.snapshot_time))).where(
            OddsSnapshot.event_id == event_id
        )

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_snapshots_for_event(self, event_id: int) -> list[OddsSnapshot]:
        """
        Get all odds snapshots for an event.

        Args:
            event_id: Event identifier

        Returns:
            List of all OddsSnapshot records for this event, ordered by time

        Example:
            reader = OddsReader(session)
            snapshots = await reader.get_snapshots_for_event("abc123")
        """
        query = (
            select(OddsSnapshot)
            .where(OddsSnapshot.event_id == event_id)
            .order_by(OddsSnapshot.snapshot_time, OddsSnapshot.id)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_games_by_date(
        self, target_date: datetime, status: EventStatus | None = EventStatus.FINAL
    ) -> list[Event]:
        """
        Get all games for a specific date using a 24-hour window (noon-to-noon UTC).

        This uses noon-to-noon UTC to capture a full NBA "game day" which spans
        two UTC calendar dates due to US timezones.

        Args:
            target_date: Start date for the 24-hour window. The window runs from
                        noon UTC on this date to noon UTC on the following day.
            status: Optional status filter (defaults to FINAL for validation)

        Returns:
            List of Event records with commence_time in the window

        Example:
            from datetime import date
            reader = OddsReader(session)
            # Gets games from noon Oct 24 to noon Oct 25 UTC
            games = await reader.get_games_by_date(
                target_date=date(2024, 10, 24),
                status=EventStatus.FINAL
            )
        """
        # Convert to datetime if date object passed
        if not isinstance(target_date, datetime):
            target_date = datetime.combine(target_date, datetime.min.time())

        # 24-hour window: noon UTC on target_date to noon UTC next day
        window_start = target_date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=UTC)
        window_end = window_start + timedelta(days=1)

        query = select(Event).where(
            and_(Event.commence_time >= window_start, Event.commence_time < window_end)
        )

        if status is not None:
            query = query.where(Event.status == status)

        query = query.order_by(Event.commence_time)

        result = await self.session.execute(query)
        return list(result.scalars().all())
