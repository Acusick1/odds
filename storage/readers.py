"""Database read operations for odds data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.fetch_tier import FetchTier
from core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot

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

    async def get_event_by_id(self, event_id: str) -> Event | None:
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
    ) -> list[Event]:
        """
        Get events within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            sport_key: Filter by sport (optional)
            status: Filter by status (optional)

        Returns:
            List of Event records
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

        query = query.order_by(Event.commence_time)

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

    async def get_odds_at_time(
        self,
        event_id: str,
        timestamp: datetime,
        tolerance_minutes: int = 5,
    ) -> list[Odds]:
        """
        Get odds snapshot closest to specified time.

        Critical for backtesting to prevent look-ahead bias.

        Args:
            event_id: Event identifier
            timestamp: Target timestamp
            tolerance_minutes: Maximum minutes before/after target

        Returns:
            List of Odds records closest to target time
        """
        time_lower = timestamp - timedelta(minutes=tolerance_minutes)
        time_upper = timestamp + timedelta(minutes=tolerance_minutes)

        # Find closest odds_timestamp directly from Odds table
        # First get all distinct timestamps in the window, then find the closest one
        distinct_timestamps_query = (
            select(Odds.odds_timestamp)
            .where(
                and_(
                    Odds.event_id == event_id,
                    Odds.odds_timestamp >= time_lower,
                    Odds.odds_timestamp <= time_upper,
                )
            )
            .distinct()
        )

        result = await self.session.execute(distinct_timestamps_query)
        all_timestamps = list(result.scalars().all())

        if not all_timestamps:
            logger.warning(
                "no_odds_at_time",
                event_id=event_id,
                timestamp=timestamp,
                tolerance_minutes=tolerance_minutes,
            )
            return []

        # Find the closest timestamp in Python
        closest_time = min(all_timestamps, key=lambda t: abs((t - timestamp).total_seconds()))

        # Get all odds at that timestamp
        odds_query = select(Odds).where(
            and_(
                Odds.event_id == event_id,
                Odds.odds_timestamp == closest_time,
            )
        )

        result = await self.session.execute(odds_query)
        return list(result.scalars().all())

    async def get_line_movement(
        self,
        event_id: str,
        bookmaker_key: str,
        market_key: str,
        outcome_name: str | None = None,
    ) -> list[Odds]:
        """
        Get time series of odds changes for analysis.

        Args:
            event_id: Event identifier
            bookmaker_key: Bookmaker to track
            market_key: Market to track (h2h, spreads, totals)
            outcome_name: Specific outcome (optional, filters if provided)

        Returns:
            List of Odds records ordered by time
        """
        query = (
            select(Odds)
            .where(
                and_(
                    Odds.event_id == event_id,
                    Odds.bookmaker_key == bookmaker_key,
                    Odds.market_key == market_key,
                )
            )
            .order_by(Odds.odds_timestamp)
        )

        if outcome_name:
            query = query.where(Odds.outcome_name == outcome_name)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_best_odds(
        self,
        event_id: str,
        market_key: str,
        outcome_name: str,
        timestamp: datetime | None = None,
    ) -> Odds | None:
        """
        Find highest odds across all bookmakers for specific outcome.

        Useful for line shopping.

        Args:
            event_id: Event identifier
            market_key: Market type
            outcome_name: Outcome to find best odds for
            timestamp: Specific time (defaults to most recent)

        Returns:
            Odds record with highest price, or None
        """
        query = select(Odds).where(
            and_(
                Odds.event_id == event_id,
                Odds.market_key == market_key,
                Odds.outcome_name == outcome_name,
            )
        )

        if timestamp:
            # Get closest snapshot time
            tolerance = timedelta(minutes=5)
            query = query.where(
                and_(
                    Odds.odds_timestamp >= timestamp - tolerance,
                    Odds.odds_timestamp <= timestamp + tolerance,
                )
            )
        else:
            # Get most recent odds
            latest_time_query = select(func.max(Odds.odds_timestamp)).where(
                Odds.event_id == event_id
            )
            result = await self.session.execute(latest_time_query)
            latest_time = result.scalar_one_or_none()

            if latest_time:
                query = query.where(Odds.odds_timestamp == latest_time)

        # Order by price (highest first for positive odds, least negative for negative)
        query = query.order_by(Odds.price.desc())

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_latest_snapshot(self, event_id: str) -> OddsSnapshot | None:
        """
        Get most recent odds snapshot for an event.

        Args:
            event_id: Event identifier

        Returns:
            Latest OddsSnapshot or None
        """
        query = (
            select(OddsSnapshot)
            .where(OddsSnapshot.event_id == event_id)
            .order_by(OddsSnapshot.snapshot_time.desc())
            .limit(1)
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def snapshot_exists(
        self, event_id: str, snapshot_time: datetime, tolerance_minutes: int = 5
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
    ) -> list[FetchLog]:
        """
        Get fetch operation logs.

        Args:
            sport_key: Filter by sport (optional)
            start_time: Only logs after this time (optional)
            limit: Maximum number of records

        Returns:
            List of FetchLog records
        """
        query = select(FetchLog)

        if sport_key:
            query = query.where(FetchLog.sport_key == sport_key)

        if start_time:
            query = query.where(FetchLog.fetch_time >= start_time)

        query = query.order_by(FetchLog.fetch_time.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_data_quality_logs(
        self,
        event_id: str | None = None,
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

        # Total odds records
        odds_count_query = select(func.count(Odds.id))
        odds_result = await self.session.execute(odds_count_query)
        total_odds = odds_result.scalar_one()

        # Total snapshots
        snapshots_count_query = select(func.count(OddsSnapshot.id))
        snapshots_result = await self.session.execute(snapshots_count_query)
        total_snapshots = snapshots_result.scalar_one()

        # Recent fetch success rate (last 24h)
        day_ago = datetime.now(UTC) - timedelta(hours=24)

        # Count all fetches
        total_fetches_query = select(func.count(FetchLog.id)).where(FetchLog.fetch_time >= day_ago)
        total_result = await self.session.execute(total_fetches_query)
        total_fetches = total_result.scalar_one()

        # Count successful fetches
        success_fetches_query = select(func.count(FetchLog.id)).where(
            and_(FetchLog.fetch_time >= day_ago, FetchLog.success.is_(True))
        )
        success_result = await self.session.execute(success_fetches_query)
        successful_fetches = success_result.scalar_one()

        success_rate = (successful_fetches / total_fetches * 100) if total_fetches > 0 else 0.0

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
            "total_odds_records": total_odds,
            "total_snapshots": total_snapshots,
            "fetch_success_rate_24h": round(success_rate, 2),
            "api_quota_remaining": latest_quota,
        }

    async def get_snapshots_by_tier(
        self, event_id: str, tier: FetchTier | None = None
    ) -> list[OddsSnapshot]:
        """
        Get odds snapshots for an event, optionally filtered by tier.

        Args:
            event_id: Event identifier
            tier: Optional FetchTier to filter by

        Returns:
            List of OddsSnapshot records

        Example:
            # Get all CLOSING tier snapshots for an event
            reader = OddsReader(session)
            closing_snapshots = await reader.get_snapshots_by_tier(
                event_id="abc123",
                tier=FetchTier.CLOSING
            )
        """
        query = select(OddsSnapshot).where(OddsSnapshot.event_id == event_id)

        if tier:
            query = query.where(OddsSnapshot.fetch_tier == tier.value)

        query = query.order_by(OddsSnapshot.snapshot_time)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_tier_coverage_for_event(self, event_id: str) -> dict[str, int]:
        """
        Get count of snapshots per tier for an event.

        Args:
            event_id: Event identifier

        Returns:
            Dictionary mapping tier name to snapshot count
            Example: {"opening": 2, "closing": 5, "pregame": 3}

        Example:
            reader = OddsReader(session)
            coverage = await reader.get_tier_coverage_for_event("abc123")
            print(f"CLOSING snapshots: {coverage.get('closing', 0)}")
        """
        query = (
            select(OddsSnapshot.fetch_tier, func.count(OddsSnapshot.id))
            .where(OddsSnapshot.event_id == event_id)
            .where(OddsSnapshot.fetch_tier.isnot(None))
            .group_by(OddsSnapshot.fetch_tier)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return dict(rows)

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

        if status:
            query = query.where(Event.status == status)

        query = query.order_by(Event.commence_time)

        result = await self.session.execute(query)
        return list(result.scalars().all())
