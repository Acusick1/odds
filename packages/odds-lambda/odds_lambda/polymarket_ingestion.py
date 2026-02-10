"""Service for ingesting Polymarket prediction market data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from odds_core.config import Settings
from odds_core.polymarket_models import PolymarketMarketType

from odds_lambda.fetch_tier import FetchTier
from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.storage.polymarket_reader import PolymarketReader
from odds_lambda.storage.polymarket_writer import PolymarketWriter
from odds_lambda.tier_utils import calculate_tier

logger = structlog.get_logger(__name__)

# Delay between CLOB calls to be rate-limit friendly (milliseconds)
DEFAULT_CLOB_DELAY_MS = 100


@dataclass(slots=True)
class PolymarketIngestionResult:
    """Outcome of a Polymarket ingestion run."""

    events_processed: int = 0
    markets_discovered: int = 0
    price_snapshots_stored: int = 0
    orderbook_snapshots_stored: int = 0
    errors: int = 0
    fetch_tier: FetchTier | None = None

    @property
    def total_snapshots(self) -> int:
        return self.price_snapshots_stored + self.orderbook_snapshots_stored


def _market_types_from_settings(settings: Settings) -> list[PolymarketMarketType]:
    """Build list of market types to collect based on settings."""
    types: list[PolymarketMarketType] = []
    if settings.polymarket.collect_moneyline:
        types.append(PolymarketMarketType.MONEYLINE)
    if settings.polymarket.collect_spreads:
        types.append(PolymarketMarketType.SPREAD)
    if settings.polymarket.collect_totals:
        types.append(PolymarketMarketType.TOTAL)
    if settings.polymarket.collect_player_props:
        types.append(PolymarketMarketType.PLAYER_PROP)
    return types


def should_fetch_orderbooks(tier_value: str, settings: Settings) -> bool:
    """Check if current tier should collect order books."""
    return tier_value in settings.polymarket.orderbook_tiers


class PolymarketIngestionService:
    """Encapsulates event discovery, price/orderbook snapshot collection."""

    def __init__(
        self,
        client: PolymarketClient,
        reader: PolymarketReader,
        writer: PolymarketWriter,
        settings: Settings,
        *,
        clob_delay_ms: int = DEFAULT_CLOB_DELAY_MS,
    ) -> None:
        self._client = client
        self._reader = reader
        self._writer = writer
        self._settings = settings
        self._clob_delay_s = clob_delay_ms / 1000

    async def discover_and_upsert_events(
        self,
        events_data: list[dict],
    ) -> PolymarketIngestionResult:
        """Upsert events and their markets from Gamma API response data.

        Commits once after processing all events.  Per-event errors are caught
        so one bad event doesn't block the rest.
        """
        result = PolymarketIngestionResult()

        for event_data in events_data:
            try:
                event = await self._writer.upsert_event(event_data)
                result.events_processed += 1

                markets_data = event_data.get("markets", [])
                if markets_data:
                    markets = await self._writer.upsert_markets(
                        pm_event_id=event.id,
                        markets_data=markets_data,
                        event_title=event.title,
                    )
                    result.markets_discovered += len(markets)

            except Exception as e:
                logger.error(
                    "event_processing_failed",
                    pm_event_id=event_data.get("id"),
                    title=event_data.get("title"),
                    error=str(e),
                    exc_info=True,
                )
                result.errors += 1

        await self._writer.session.commit()
        return result

    async def collect_snapshots(
        self,
        now: datetime | None = None,
    ) -> PolymarketIngestionResult:
        """Fetch prices (and optionally order books) for all active markets.

        Returns a result with snapshot counts.  Per-event errors are caught so
        one failing market doesn't block the rest.
        """
        now = now or datetime.now(UTC)
        result = PolymarketIngestionResult()

        active_events = await self._reader.get_active_events()
        if not active_events:
            logger.warning("no_active_events_after_upsert")
            return result

        closest_game = min(event.start_date for event in active_events)
        hours_until = (closest_game - now).total_seconds() / 3600
        fetch_tier = calculate_tier(hours_until)
        result.fetch_tier = fetch_tier
        fetch_books = should_fetch_orderbooks(fetch_tier.value, self._settings)

        logger.info(
            "fetch_polymarket_executing",
            closest_game=closest_game.isoformat(),
            hours_until=round(hours_until, 2),
            tier=fetch_tier.value,
            fetch_orderbooks=fetch_books,
        )

        market_types = _market_types_from_settings(self._settings)

        for event in active_events:
            try:
                for market_type in market_types:
                    markets = await self._reader.get_markets_by_type(event.id, market_type)

                    for market in markets:
                        token_ids = market.clob_token_ids

                        if not token_ids or len(token_ids) < 2:
                            logger.warning(
                                "market_missing_tokens",
                                market_id=market.id,
                                question=market.question,
                            )
                            continue

                        prices = await self._client.get_prices_batch(token_ids)
                        await asyncio.sleep(self._clob_delay_s)

                        if token_ids[0] in prices and token_ids[1] in prices:
                            price_0 = prices[token_ids[0]]
                            price_1 = prices[token_ids[1]]

                            if price_0 is not None and price_1 is not None:
                                await self._writer.store_price_snapshot(
                                    market=market,
                                    prices={
                                        "outcome_0_price": price_0,
                                        "outcome_1_price": price_1,
                                    },
                                    commence_time=event.start_date,
                                    snapshot_time=now,
                                )
                                result.price_snapshots_stored += 1
                            else:
                                logger.warning(
                                    "price_snapshot_skipped_none_price",
                                    market_id=market.id,
                                    question=market.question,
                                )
                        else:
                            logger.warning(
                                "price_snapshot_skipped_missing_tokens",
                                market_id=market.id,
                                question=market.question,
                                available_tokens=list(prices.keys()),
                                expected_tokens=token_ids,
                            )

                        if fetch_books and market_type == PolymarketMarketType.MONEYLINE:
                            token_id = token_ids[0]
                            raw_book = await self._client.get_order_book(token_id)
                            await asyncio.sleep(self._clob_delay_s)

                            snapshot = await self._writer.store_orderbook_snapshot(
                                market=market,
                                raw_book=raw_book,
                                commence_time=event.start_date,
                                snapshot_time=now,
                            )
                            if snapshot:
                                result.orderbook_snapshots_stored += 1

            except Exception as e:
                logger.error(
                    "market_fetch_failed",
                    pm_event_id=event.pm_event_id,
                    title=event.title,
                    error=str(e),
                    exc_info=True,
                )
                result.errors += 1

        await self._writer.session.commit()
        return result
