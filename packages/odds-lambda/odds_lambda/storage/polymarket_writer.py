"""Database write operations for Polymarket prediction market data."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketFetchLog,
    PolymarketMarket,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_core.time import ensure_utc, parse_api_datetime
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.polymarket_fetcher import classify_market, process_order_book
from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier_from_timestamps

logger = structlog.get_logger()


class PolymarketWriter:
    """Handles all write operations for Polymarket data to the database."""

    def __init__(self, session: AsyncSession):
        """
        Initialize writer with database session.

        Args:
            session: Async database session
        """
        self.session = session

    async def upsert_event(self, event_data: dict) -> PolymarketEvent:
        """
        Insert or update a Polymarket event.

        Args:
            event_data: Event data from Gamma API

        Returns:
            Created or updated PolymarketEvent instance
        """
        pm_event_id = event_data["id"]

        # Parse timestamps from Gamma API format
        start_date = parse_api_datetime(event_data["startDate"])
        end_date = parse_api_datetime(event_data["endDate"])

        # Build event dict
        event_dict = {
            "pm_event_id": pm_event_id,
            "ticker": event_data["ticker"],
            "slug": event_data.get("slug", ""),
            "title": event_data["title"],
            "start_date": start_date,
            "end_date": end_date,
            "active": event_data.get("active", True),
            "closed": event_data.get("closed", False),
            "volume": event_data.get("volume"),
            "liquidity": event_data.get("liquidity"),
            "markets_count": len(event_data.get("markets", [])),
            "updated_at": datetime.now(UTC),
        }

        # Upsert using PostgreSQL INSERT ... ON CONFLICT DO UPDATE
        stmt = insert(PolymarketEvent).values(event_dict)

        # Update all fields except id, pm_event_id, created_at on conflict
        set_ = {
            col.name: stmt.excluded[col.name]
            for col in PolymarketEvent.__table__.columns
            if col.name not in ("id", "pm_event_id", "created_at")
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=["pm_event_id"],
            set_=set_,
        )

        await self.session.execute(stmt)
        await self.session.flush()

        # Fetch the upserted event with fresh data
        result = await self.session.execute(
            select(PolymarketEvent).where(PolymarketEvent.pm_event_id == pm_event_id)
        )
        event = result.scalar_one()
        await self.session.refresh(event)

        logger.info(
            "polymarket_event_upserted",
            pm_event_id=pm_event_id,
            ticker=event.ticker,
            title=event.title,
            markets_count=event.markets_count,
        )

        return event

    async def upsert_markets(
        self, pm_event_id: int, markets_data: list[dict], event_title: str
    ) -> list[PolymarketMarket]:
        """
        Insert or update markets for a Polymarket event.

        Args:
            pm_event_id: Parent PolymarketEvent database ID
            markets_data: List of market dicts from Gamma API
            event_title: Event title for market classification

        Returns:
            List of upserted PolymarketMarket instances
        """
        if not markets_data:
            return []

        market_dicts = []
        pm_market_ids = []

        for market_data in markets_data:
            pm_market_id = market_data["id"]
            pm_market_ids.append(pm_market_id)

            question = market_data["question"]
            market_type, point = classify_market(question, event_title)

            market_dict = {
                "polymarket_event_id": pm_event_id,
                "pm_market_id": pm_market_id,
                "condition_id": market_data["conditionId"],
                "question": question,
                "clob_token_ids": market_data["clobTokenIds"],
                "outcomes": market_data["outcomes"],
                "market_type": market_type,
                "group_item_title": market_data.get("groupItemTitle"),
                "point": point,
                "active": market_data.get("active", True),
                "closed": market_data.get("closed", False),
                "accepting_orders": market_data.get("acceptingOrders", True),
            }
            market_dicts.append(market_dict)

        # Bulk upsert
        stmt = insert(PolymarketMarket).values(market_dicts)

        # Update all fields except id, pm_market_id, created_at on conflict
        set_ = {
            col.name: stmt.excluded[col.name]
            for col in PolymarketMarket.__table__.columns
            if col.name not in ("id", "pm_market_id", "created_at")
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=["pm_market_id"],
            set_=set_,
        )

        await self.session.execute(stmt)
        await self.session.flush()

        # Fetch upserted markets
        result = await self.session.execute(
            select(PolymarketMarket).where(PolymarketMarket.pm_market_id.in_(pm_market_ids))
        )
        markets = list(result.scalars().all())

        # Count by market type for logging
        type_counts: dict[str, int] = {}
        for market in markets:
            type_counts[market.market_type.value] = type_counts.get(market.market_type.value, 0) + 1

        logger.info(
            "polymarket_markets_upserted",
            count=len(markets),
            **type_counts,
        )

        return markets

    async def store_price_snapshot(
        self,
        market: PolymarketMarket,
        prices: dict,
        commence_time: datetime,
        snapshot_time: datetime | None = None,
    ) -> PolymarketPriceSnapshot:
        """
        Store a price snapshot for a Polymarket market.

        Args:
            market: PolymarketMarket instance
            prices: Price data dict
            commence_time: Game commence time
            snapshot_time: Time of snapshot (defaults to now)

        Returns:
            Created PolymarketPriceSnapshot instance
        """
        snapshot_time = ensure_utc(snapshot_time or datetime.now(UTC))
        commence_time = ensure_utc(commence_time)

        # Calculate tier and hours_until
        fetch_tier = calculate_tier_from_timestamps(snapshot_time, commence_time)
        hours_until = calculate_hours_until_commence(snapshot_time, commence_time)

        # Calculate derived metrics if both bid and ask available
        best_bid = prices.get("best_bid")
        best_ask = prices.get("best_ask")
        spread = None
        midpoint = None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            midpoint = (best_bid + best_ask) / 2

        snapshot = PolymarketPriceSnapshot(
            polymarket_market_id=market.id,
            snapshot_time=snapshot_time,
            outcome_0_price=prices["outcome_0_price"],
            outcome_1_price=prices["outcome_1_price"],
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            midpoint=midpoint,
            volume=prices.get("volume"),
            liquidity=prices.get("liquidity"),
            fetch_tier=fetch_tier.value,
            hours_until_commence=hours_until,
        )

        self.session.add(snapshot)

        logger.info(
            "polymarket_price_snapshot_stored",
            market_id=market.id,
            snapshot_time=snapshot_time,
            fetch_tier=fetch_tier.value,
            hours_until=round(hours_until, 2),
        )

        return snapshot

    async def store_orderbook_snapshot(
        self,
        market: PolymarketMarket,
        raw_book: dict,
        commence_time: datetime,
        snapshot_time: datetime | None = None,
    ) -> PolymarketOrderBookSnapshot | None:
        """
        Store an order book snapshot for a Polymarket market.

        Args:
            market: PolymarketMarket instance
            raw_book: Raw order book from CLOB API
            commence_time: Game commence time
            snapshot_time: Time of snapshot (defaults to now)

        Returns:
            Created PolymarketOrderBookSnapshot instance or None if invalid book
        """
        snapshot_time = ensure_utc(snapshot_time or datetime.now(UTC))
        commence_time = ensure_utc(commence_time)

        # Process order book to extract metrics
        processed = process_order_book(raw_book)

        if processed is None:
            logger.warning(
                "polymarket_orderbook_invalid",
                market_id=market.id,
                snapshot_time=snapshot_time,
            )
            return None

        # Calculate tier and hours_until
        fetch_tier = calculate_tier_from_timestamps(snapshot_time, commence_time)
        hours_until = calculate_hours_until_commence(snapshot_time, commence_time)

        # Get token_id from market (use first clob_token_id for now)
        token_id = market.clob_token_ids[0] if market.clob_token_ids else ""

        snapshot = PolymarketOrderBookSnapshot(
            polymarket_market_id=market.id,
            snapshot_time=snapshot_time,
            token_id=token_id,
            raw_book=raw_book,
            best_bid=processed["best_bid"],
            best_ask=processed["best_ask"],
            spread=processed["spread"],
            midpoint=processed["midpoint"],
            bid_levels=processed["bid_levels"],
            ask_levels=processed["ask_levels"],
            bid_depth_total=processed["bid_depth_total"],
            ask_depth_total=processed["ask_depth_total"],
            imbalance=processed["imbalance"],
            weighted_mid=processed["weighted_mid"],
            fetch_tier=fetch_tier.value,
            hours_until_commence=hours_until,
        )

        self.session.add(snapshot)

        logger.info(
            "polymarket_orderbook_snapshot_stored",
            market_id=market.id,
            bid_levels=processed["bid_levels"],
            ask_levels=processed["ask_levels"],
            spread=round(processed["spread"], 4),
            fetch_tier=fetch_tier.value,
        )

        return snapshot

    async def bulk_store_price_history(
        self, market: PolymarketMarket, history: list[dict], commence_time: datetime
    ) -> int:
        """
        Bulk store historical price data for a market.

        Args:
            market: PolymarketMarket instance
            history: List of historical price points from CLOB API [{"t": unix_ts, "p": price_str}]
            commence_time: Game commence time

        Returns:
            Number of snapshots actually inserted (skips duplicates)
        """
        if not history:
            return 0

        commence_time = ensure_utc(commence_time)
        snapshot_dicts = []

        for point in history:
            # Parse Unix timestamp to datetime
            snapshot_time = ensure_utc(datetime.fromtimestamp(point["t"], tz=UTC))

            # Calculate tier for this historical point
            fetch_tier = calculate_tier_from_timestamps(snapshot_time, commence_time)
            hours_until = calculate_hours_until_commence(snapshot_time, commence_time)

            # Parse price (complementary probabilities)
            outcome_0_price = float(point["p"])
            outcome_1_price = 1.0 - outcome_0_price

            snapshot_dict = {
                "polymarket_market_id": market.id,
                "snapshot_time": snapshot_time,
                "outcome_0_price": outcome_0_price,
                "outcome_1_price": outcome_1_price,
                "best_bid": None,
                "best_ask": None,
                "spread": None,
                "midpoint": None,
                "volume": None,
                "liquidity": None,
                "fetch_tier": fetch_tier.value,
                "hours_until_commence": hours_until,
            }
            snapshot_dicts.append(snapshot_dict)

        # Pre-filter to check which snapshots already exist
        # Note: For very large datasets (thousands of points), WHERE IN (...) could be expensive.
        # We use this approach to provide accurate inserted/skipped counts for logging.
        # Alternative: Add unique constraint on (market_id, snapshot_time) and use ON CONFLICT DO NOTHING,
        # which would be more efficient but lose granular count reporting.
        if snapshot_dicts:
            existing_query = select(PolymarketPriceSnapshot.snapshot_time).where(
                and_(
                    PolymarketPriceSnapshot.polymarket_market_id == market.id,
                    PolymarketPriceSnapshot.snapshot_time.in_(
                        [d["snapshot_time"] for d in snapshot_dicts]
                    ),
                )
            )
            result = await self.session.execute(existing_query)
            existing_times = {row[0] for row in result.fetchall()}

            # Filter out existing snapshots
            new_snapshot_dicts = [
                d for d in snapshot_dicts if d["snapshot_time"] not in existing_times
            ]
        else:
            new_snapshot_dicts = []

        # Bulk insert only new snapshots
        inserted_count = len(new_snapshot_dicts)
        if new_snapshot_dicts:
            stmt = insert(PolymarketPriceSnapshot).values(new_snapshot_dicts)
            await self.session.execute(stmt)

        skipped_count = len(history) - inserted_count

        logger.info(
            "polymarket_price_history_bulk_inserted",
            total_points=len(history),
            inserted=inserted_count,
            skipped=skipped_count,
        )

        return inserted_count

    async def log_fetch(self, log: PolymarketFetchLog) -> PolymarketFetchLog:
        """
        Log a Polymarket fetch operation.

        Args:
            log: PolymarketFetchLog instance to persist

        Returns:
            Created PolymarketFetchLog record
        """
        self.session.add(log)

        logger.info(
            "polymarket_fetch_logged",
            job_type=log.job_type,
            events_count=log.events_count,
            markets_count=log.markets_count,
            snapshots_stored=log.snapshots_stored,
            success=log.success,
            error_message=log.error_message,
        )

        return log
