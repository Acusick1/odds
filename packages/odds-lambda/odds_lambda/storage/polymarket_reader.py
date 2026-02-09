"""Database read operations for Polymarket prediction market data."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from odds_core.models import Event
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class PolymarketReader:
    """Handles all read operations for Polymarket data from the database."""

    def __init__(self, session: AsyncSession):
        """
        Initialize reader with database session.

        Args:
            session: Async database session
        """
        self.session = session

    async def get_active_events(self) -> list[PolymarketEvent]:
        """
        Get all active, non-closed Polymarket events.

        Returns:
            List of active PolymarketEvent records ordered by start_date
        """
        query = (
            select(PolymarketEvent)
            .where(and_(PolymarketEvent.active.is_(True), PolymarketEvent.closed.is_(False)))
            .order_by(PolymarketEvent.start_date)
        )

        result = await self.session.execute(query)
        events = list(result.scalars().all())

        logger.info("active_polymarket_events_fetched", count=len(events))

        return events

    async def get_markets_by_type(
        self, pm_event_id: int, market_type: PolymarketMarketType
    ) -> list[PolymarketMarket]:
        """
        Get markets for an event filtered by market type.

        Args:
            pm_event_id: PolymarketEvent database ID
            market_type: Market type to filter by

        Returns:
            List of matching PolymarketMarket records
        """
        query = select(PolymarketMarket).where(
            and_(
                PolymarketMarket.polymarket_event_id == pm_event_id,
                PolymarketMarket.market_type == market_type,
            )
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_moneyline_market(self, pm_event_id: int) -> PolymarketMarket | None:
        """
        Get the moneyline market for an event (convenience wrapper).

        Args:
            pm_event_id: PolymarketEvent database ID

        Returns:
            First moneyline market or None if not found
        """
        markets = await self.get_markets_by_type(pm_event_id, PolymarketMarketType.MONEYLINE)
        return markets[0] if markets else None

    async def get_price_at_time(
        self, market_id: int, target_time: datetime, tolerance_minutes: int = 5
    ) -> PolymarketPriceSnapshot | None:
        """
        Get price snapshot closest to target time without look-ahead bias.

        CRITICAL: Only returns snapshots at or before target_time to prevent look-ahead bias.

        Args:
            market_id: PolymarketMarket database ID
            target_time: Target timestamp
            tolerance_minutes: Maximum minutes before target to search

        Returns:
            Closest PolymarketPriceSnapshot at or before target_time, or None
        """
        time_lower = target_time - timedelta(minutes=tolerance_minutes)

        # Only look backward from target time (never future data)
        query = (
            select(PolymarketPriceSnapshot)
            .where(
                and_(
                    PolymarketPriceSnapshot.polymarket_market_id == market_id,
                    PolymarketPriceSnapshot.snapshot_time >= time_lower,
                    PolymarketPriceSnapshot.snapshot_time <= target_time,
                )
            )
            .order_by(PolymarketPriceSnapshot.snapshot_time.desc())
            .limit(1)
        )

        result = await self.session.execute(query)
        snapshot = result.scalar_one_or_none()

        if snapshot:
            time_diff = (target_time - snapshot.snapshot_time).total_seconds() / 60
            logger.info(
                "polymarket_price_at_time_found",
                market_id=market_id,
                target_time=target_time,
                snapshot_time=snapshot.snapshot_time,
                time_diff_minutes=round(time_diff, 2),
            )
        else:
            logger.warning(
                "polymarket_price_at_time_not_found",
                market_id=market_id,
                target_time=target_time,
                tolerance_minutes=tolerance_minutes,
            )

        return snapshot

    async def get_price_series(
        self, market_id: int, start: datetime, end: datetime
    ) -> list[PolymarketPriceSnapshot]:
        """
        Get price series for a market within a time range.

        Args:
            market_id: PolymarketMarket database ID
            start: Start of time range
            end: End of time range

        Returns:
            List of PolymarketPriceSnapshot records ordered by snapshot_time
        """
        query = (
            select(PolymarketPriceSnapshot)
            .where(
                and_(
                    PolymarketPriceSnapshot.polymarket_market_id == market_id,
                    PolymarketPriceSnapshot.snapshot_time >= start,
                    PolymarketPriceSnapshot.snapshot_time <= end,
                )
            )
            .order_by(PolymarketPriceSnapshot.snapshot_time)
        )

        result = await self.session.execute(query)
        snapshots = list(result.scalars().all())

        logger.info("polymarket_price_series_fetched", market_id=market_id, count=len(snapshots))

        return snapshots

    async def get_orderbook_at_time(
        self, market_id: int, target_time: datetime, tolerance_minutes: int = 5
    ) -> PolymarketOrderBookSnapshot | None:
        """
        Get order book snapshot closest to target time.

        Args:
            market_id: PolymarketMarket database ID
            target_time: Target timestamp
            tolerance_minutes: Maximum minutes before/after target to search

        Returns:
            Closest PolymarketOrderBookSnapshot or None
        """
        time_lower = target_time - timedelta(minutes=tolerance_minutes)
        time_upper = target_time + timedelta(minutes=tolerance_minutes)

        # Find all distinct timestamps in tolerance window
        distinct_timestamps_query = (
            select(PolymarketOrderBookSnapshot.snapshot_time)
            .where(
                and_(
                    PolymarketOrderBookSnapshot.polymarket_market_id == market_id,
                    PolymarketOrderBookSnapshot.snapshot_time >= time_lower,
                    PolymarketOrderBookSnapshot.snapshot_time <= time_upper,
                )
            )
            .distinct()
        )

        result = await self.session.execute(distinct_timestamps_query)
        all_timestamps = list(result.scalars().all())

        if not all_timestamps:
            logger.warning(
                "polymarket_orderbook_at_time_not_found",
                market_id=market_id,
                target_time=target_time,
                tolerance_minutes=tolerance_minutes,
            )
            return None

        # Find closest timestamp in Python
        closest_time = min(all_timestamps, key=lambda t: abs((t - target_time).total_seconds()))

        # Fetch snapshot at exact closest time
        snapshot_query = select(PolymarketOrderBookSnapshot).where(
            and_(
                PolymarketOrderBookSnapshot.polymarket_market_id == market_id,
                PolymarketOrderBookSnapshot.snapshot_time == closest_time,
            )
        )

        result = await self.session.execute(snapshot_query)
        snapshot = result.scalar_one_or_none()

        if snapshot:
            logger.info(
                "polymarket_orderbook_at_time_found",
                market_id=market_id,
                target_time=target_time,
                snapshot_time=snapshot.snapshot_time,
            )

        return snapshot

    async def get_linked_events(self) -> list[tuple[PolymarketEvent, Event]]:
        """
        Get Polymarket events linked to internal events.

        Returns:
            List of (PolymarketEvent, Event) tuples ordered by start_date
        """
        query = (
            select(PolymarketEvent, Event)
            .join(Event, PolymarketEvent.event_id == Event.id)
            .where(PolymarketEvent.event_id.isnot(None))
            .order_by(PolymarketEvent.start_date)
        )

        result = await self.session.execute(query)
        linked = list(result.all())

        logger.info("linked_polymarket_events_fetched", count=len(linked))

        return linked

    async def get_backfilled_market_ids(self, min_snapshots: int = 10) -> set[int]:
        """
        Get market IDs that already have sufficient historical snapshots.

        Useful for skip logic in recurring backfill jobs to avoid re-fetching.

        Args:
            min_snapshots: Minimum snapshot count to consider backfilled

        Returns:
            Set of PolymarketMarket database IDs with sufficient history
        """
        query = (
            select(PolymarketPriceSnapshot.polymarket_market_id)
            .group_by(PolymarketPriceSnapshot.polymarket_market_id)
            .having(func.count(PolymarketPriceSnapshot.id) >= min_snapshots)
        )

        result = await self.session.execute(query)
        market_ids = {row[0] for row in result.fetchall()}

        logger.info("backfilled_polymarket_markets_identified", count=len(market_ids))

        return market_ids
