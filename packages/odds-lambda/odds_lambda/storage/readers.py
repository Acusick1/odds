"""Database read operations for odds data."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import structlog
from odds_core.config import APIConfig, get_settings
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


# A key's quota reading older than this no longer describes the key (the quota resets monthly).
QUOTA_MAX_AGE = timedelta(days=31)


class KeyQuota(NamedTuple):
    """The newest remaining and used quota of an Odds API key and the sports that spend it."""

    sports: tuple[str, ...]
    remaining: int
    used: int

    @property
    def limit(self) -> int:
        """The key's plan size: credits used since the last reset plus credits remaining."""
        return self.remaining + self.used

    @property
    def fraction(self) -> float:
        """Remaining credits as a fraction of the plan; 0.0 for a plan of zero."""
        return self.remaining / self.limit if self.limit else 0.0


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

    async def get_fetch_logs(
        self,
        sport_key: str | None = None,
        start_time: datetime | None = None,
        limit: int = 100,
        with_quota: bool = False,
        sport_keys: Sequence[str] | None = None,
    ) -> list[FetchLog]:
        """
        Get fetch operation logs.

        Args:
            sport_key: Filter by sport (optional)
            start_time: Only logs after this time (optional)
            limit: Maximum number of records
            with_quota: Only rows with a non-null api_quota_remaining
            sport_keys: Filter by any of these sports (optional)

        Returns:
            List of FetchLog records
        """
        query = select(FetchLog)

        if sport_key:
            query = query.where(FetchLog.sport_key == sport_key)

        if sport_keys is not None:
            query = query.where(FetchLog.sport_key.in_(list(sport_keys)))  # type: ignore[attr-defined]

        if start_time:
            query = query.where(FetchLog.fetch_time >= start_time)

        if with_quota:
            query = query.where(FetchLog.api_quota_remaining.is_not(None))

        query = query.order_by(FetchLog.fetch_time.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_latest_quota_by_key(self, api: APIConfig) -> dict[int, KeyQuota]:
        """The newest recorded remaining and used quota of each Odds API key, by key index.

        A key spends for the sports ``APIConfig.key_index_by_sport`` gives its index, so its
        balance is the newest reading of any of them. A reading older than ``QUOTA_MAX_AGE`` is a
        key no sport has spent for a month (an off-season, or a key left behind) and is ignored.
        A row needs both ``api_quota_remaining`` and ``api_quota_used``, which give the key's plan
        size; rows logged before ``api_quota_used`` existed are skipped.
        """
        result = await self.session.execute(
            select(
                FetchLog.sport_key,
                FetchLog.fetch_time,
                FetchLog.api_quota_remaining,
                FetchLog.api_quota_used,
            )
            .where(
                FetchLog.api_quota_remaining.is_not(None),
                FetchLog.api_quota_used.is_not(None),
                FetchLog.fetch_time >= datetime.now(UTC) - QUOTA_MAX_AGE,
            )
            .distinct(FetchLog.sport_key)
            .order_by(FetchLog.sport_key, FetchLog.fetch_time.desc())
        )
        readings: dict[int, tuple[datetime, int, int]] = {}
        sports: dict[int, list[str]] = {}
        for sport_key, fetch_time, remaining, used in result.all():
            index = api.key_index_for(sport_key)
            sports.setdefault(index, []).append(sport_key)
            if index not in readings or fetch_time > readings[index][0]:
                readings[index] = (fetch_time, remaining, used)
        return {
            index: KeyQuota(tuple(sports[index]), remaining, used)
            for index, (_, remaining, used) in sorted(readings.items())
        }

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

        quota_by_key = await self.get_latest_quota_by_key(get_settings().api)

        return {
            "events_by_status": events_by_status,
            "total_events": sum(events_by_status.values()),
            "total_price_rows": price_rows,
            "total_snapshots": total_snapshots,
            "api_quota_by_key": quota_by_key,
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
