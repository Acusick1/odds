"""
Fetch Polymarket prices job - standalone executable or Lambda handler.

This job:
1. Checks if Polymarket collection is enabled
2. Discovers active NBA events from Gamma API
3. Upserts events and markets to DB
4. Fetches current prices for relevant markets
5. Fetches order books for moneyline markets (tier-gated)
6. Self-schedules next execution based on game proximity
"""

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import Settings, get_settings
from odds_core.database import async_session_maker
from odds_core.polymarket_models import (
    PolymarketFetchLog,
    PolymarketMarketType,
)

from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.storage.polymarket_reader import PolymarketReader
from odds_lambda.storage.polymarket_writer import PolymarketWriter
from odds_lambda.tier_utils import calculate_tier

logger = structlog.get_logger()


async def should_fetch_orderbooks(tier_value: str, settings: Settings) -> bool:
    """
    Check if current tier should collect order books.

    Args:
        tier_value: Current tier value string (e.g., "closing", "pregame")
        settings: Application settings

    Returns:
        True if order books should be collected for this tier
    """
    return tier_value in settings.polymarket.orderbook_tiers


async def get_closest_polymarket_game(reader: PolymarketReader) -> datetime | None:
    """
    Find the soonest upcoming Polymarket event start time.

    Args:
        reader: PolymarketReader instance

    Returns:
        Datetime of closest event start, or None if no active events
    """
    active_events = await reader.get_active_events()

    if not active_events:
        return None

    # Return earliest start_date
    return min(event.start_date for event in active_events)


async def main():
    """
    Main job execution flow.

    Flow:
    1. Check if Polymarket collection is enabled
    2. Discover active NBA events from Gamma API
    3. Upsert events and markets to DB
    4. Determine current fetch tier from closest game
    5. Fetch prices for relevant markets (based on config)
    6. If tier is in orderbook_tiers, fetch order books for moneyline markets
    7. Store snapshots with tier metadata
    8. Log fetch results
    9. Schedule next execution
    """
    app_settings = get_settings()

    logger.info("fetch_polymarket_job_started", backend=app_settings.scheduler.backend)

    # Check if Polymarket collection is enabled
    if not app_settings.polymarket.enabled:
        logger.info(
            "fetch_polymarket_skipped",
            reason="polymarket.enabled=False",
        )
        return

    # Statistics tracking
    stats = {
        "events_processed": 0,
        "markets_discovered": 0,
        "price_snapshots_stored": 0,
        "orderbook_snapshots_stored": 0,
        "errors": 0,
    }

    fetch_tier = None
    hours_until = None

    try:
        async with PolymarketClient() as client, async_session_maker() as session:
            reader = PolymarketReader(session)
            writer = PolymarketWriter(session)

            # Discover active NBA events from Gamma API
            logger.info("discovering_active_polymarket_events")
            events_data = await client.get_nba_events(active=True, closed=False)

            if not events_data:
                logger.info(
                    "fetch_polymarket_skipped",
                    reason="no_active_polymarket_events",
                )

                # Still schedule next check even if no events
                closest_game = await get_closest_polymarket_game(reader)
                if closest_game:
                    now = datetime.now(UTC)
                    hours_until = (closest_game - now).total_seconds() / 3600
                    fetch_tier = calculate_tier(hours_until)
                    next_execution = now + timedelta(
                        seconds=app_settings.polymarket.price_poll_interval
                    )
                else:
                    # No active events - check daily for new events
                    next_execution = datetime.now(UTC) + timedelta(hours=24)

                backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
                await backend.schedule_next_execution(
                    job_name="fetch-polymarket", next_time=next_execution
                )

                return

            logger.info("active_polymarket_events_discovered", count=len(events_data))

            # Process each event: upsert event and markets
            for event_data in events_data:
                try:
                    # Upsert event
                    event = await writer.upsert_event(event_data)
                    stats["events_processed"] += 1

                    # Upsert markets
                    markets_data = event_data.get("markets", [])
                    if markets_data:
                        markets = await writer.upsert_markets(
                            pm_event_id=event.id,
                            markets_data=markets_data,
                            event_title=event.title,
                        )
                        stats["markets_discovered"] += len(markets)

                except Exception as e:
                    logger.error(
                        "event_processing_failed",
                        pm_event_id=event_data.get("id"),
                        title=event_data.get("title"),
                        error=str(e),
                        exc_info=True,
                    )
                    stats["errors"] += 1
                    # Continue processing other events

            # Commit event and market upserts
            await session.commit()

            # Determine current fetch tier from closest game
            closest_game = await get_closest_polymarket_game(reader)

            if not closest_game:
                logger.warning("no_active_events_after_upsert")

                # Schedule next check even if no events to process
                next_execution = datetime.now(UTC) + timedelta(hours=24)
                backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
                await backend.schedule_next_execution(
                    job_name="fetch-polymarket", next_time=next_execution
                )
                return

            now = datetime.now(UTC)
            hours_until = (closest_game - now).total_seconds() / 3600
            fetch_tier = calculate_tier(hours_until)
            should_fetch_books = await should_fetch_orderbooks(fetch_tier.value, app_settings)

            logger.info(
                "fetch_polymarket_executing",
                closest_game=closest_game.isoformat(),
                hours_until=round(hours_until, 2),
                tier=fetch_tier.value,
                fetch_orderbooks=should_fetch_books,
            )

            # Fetch prices for all active events
            active_events = await reader.get_active_events()

            for event in active_events:
                try:
                    # Get markets for this event based on config
                    market_types_to_collect = []
                    if app_settings.polymarket.collect_moneyline:
                        market_types_to_collect.append(PolymarketMarketType.MONEYLINE)
                    if app_settings.polymarket.collect_spreads:
                        market_types_to_collect.append(PolymarketMarketType.SPREAD)
                    if app_settings.polymarket.collect_totals:
                        market_types_to_collect.append(PolymarketMarketType.TOTAL)
                    if app_settings.polymarket.collect_player_props:
                        market_types_to_collect.append(PolymarketMarketType.PLAYER_PROP)

                    for market_type in market_types_to_collect:
                        markets = await reader.get_markets_by_type(event.id, market_type)

                        for market in markets:
                            # Collect token IDs for price fetching
                            token_ids = market.clob_token_ids

                            if not token_ids or len(token_ids) < 2:
                                logger.warning(
                                    "market_missing_tokens",
                                    market_id=market.id,
                                    question=market.question,
                                )
                                continue

                            # Fetch prices for both outcomes
                            prices = await client.get_prices_batch(token_ids)

                            if len(prices) >= 2:
                                # Store price snapshot
                                price_data = {
                                    "outcome_0_price": prices.get(token_ids[0], 0.0) or 0.0,
                                    "outcome_1_price": prices.get(token_ids[1], 0.0) or 0.0,
                                }

                                await writer.store_price_snapshot(
                                    market=market,
                                    prices=price_data,
                                    commence_time=event.start_date,
                                    snapshot_time=now,
                                )
                                stats["price_snapshots_stored"] += 1

                            # Fetch order books for moneyline markets if tier-gated
                            if should_fetch_books and market_type == PolymarketMarketType.MONEYLINE:
                                # Fetch order book for first token (typically the "Yes" side)
                                token_id = token_ids[0]
                                raw_book = await client.get_order_book(token_id)

                                if raw_book:
                                    snapshot = await writer.store_orderbook_snapshot(
                                        market=market,
                                        raw_book=raw_book,
                                        commence_time=event.start_date,
                                        snapshot_time=now,
                                    )

                                    if snapshot:
                                        stats["orderbook_snapshots_stored"] += 1

                except Exception as e:
                    logger.error(
                        "market_fetch_failed",
                        pm_event_id=event.pm_event_id,
                        title=event.title,
                        error=str(e),
                        exc_info=True,
                    )
                    stats["errors"] += 1
                    # Continue processing other markets

            # Commit all snapshots
            await session.commit()

            logger.info(
                "fetch_polymarket_completed",
                events_processed=stats["events_processed"],
                markets_discovered=stats["markets_discovered"],
                price_snapshots_stored=stats["price_snapshots_stored"],
                orderbook_snapshots_stored=stats["orderbook_snapshots_stored"],
                errors=stats["errors"],
            )

        # Log fetch operation
        fetch_log = PolymarketFetchLog(
            job_type="fetch-polymarket",
            events_count=stats["events_processed"],
            markets_count=stats["markets_discovered"],
            snapshots_stored=stats["price_snapshots_stored"] + stats["orderbook_snapshots_stored"],
            success=True,
            error_message=None,
        )

        async with async_session_maker() as log_session:
            log_writer = PolymarketWriter(log_session)
            await log_writer.log_fetch(fetch_log)
            await log_session.commit()

    except Exception as e:
        logger.error("fetch_polymarket_failed", error=str(e), exc_info=True)

        # Log failed fetch
        fetch_log = PolymarketFetchLog(
            job_type="fetch-polymarket",
            events_count=stats["events_processed"],
            markets_count=stats["markets_discovered"],
            snapshots_stored=stats["price_snapshots_stored"] + stats["orderbook_snapshots_stored"],
            success=False,
            error_message=str(e),
        )

        async with async_session_maker() as log_session:
            log_writer = PolymarketWriter(log_session)
            await log_writer.log_fetch(fetch_log)
            await log_session.commit()

        # Send critical alert
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            await send_critical(f"🚨 Fetch Polymarket job failed: {type(e).__name__}: {str(e)}")

        # Don't schedule next run if we failed - let manual intervention happen
        raise

    # Self-schedule next execution
    if fetch_tier:
        try:
            # Use price poll interval for next execution
            next_execution = datetime.now(UTC) + timedelta(
                seconds=app_settings.polymarket.price_poll_interval
            )

            backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
            await backend.schedule_next_execution(
                job_name="fetch-polymarket", next_time=next_execution
            )
            logger.info(
                "fetch_polymarket_next_scheduled",
                next_time=next_execution.isoformat(),
                backend=backend.get_backend_name(),
                tier=fetch_tier.value,
            )
        except Exception as e:
            logger.error("fetch_polymarket_scheduling_failed", error=str(e), exc_info=True)

            # Send error alert
            if app_settings.alerts.alert_enabled:
                from odds_cli.alerts.base import send_error

                await send_error(
                    f"Fetch Polymarket scheduling failed: {type(e).__name__}: {str(e)}"
                )

            # Don't fail the job if scheduling fails - the fetch itself succeeded


if __name__ == "__main__":
    asyncio.run(main())
