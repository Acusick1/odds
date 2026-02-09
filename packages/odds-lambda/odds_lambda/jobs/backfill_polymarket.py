"""
Backfill Polymarket price history job.

This job captures historical price data from resolved Polymarket NBA markets
before it expires. The CLOB /prices-history endpoint has a ~30-day rolling
retention window, meaning data from games older than ~25 days is gradually purged.

Job flow:
1. Query Gamma API for closed NBA events
2. Paginate through all results
3. For each event:
   a. Upsert event and markets to DB
   b. Check if moneyline market already has price history (skip if so)
   c. Fetch /prices-history with interval=max, fidelity=5
   d. Bulk insert into polymarket_price_snapshots
   e. Optionally fetch spread/total histories (based on config)
4. Log results to PolymarketFetchLog
"""

import asyncio
from datetime import datetime

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.polymarket_models import (
    PolymarketFetchLog,
    PolymarketMarket,
    PolymarketMarketType,
)

from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.storage.polymarket_reader import PolymarketReader
from odds_lambda.storage.polymarket_writer import PolymarketWriter

logger = structlog.get_logger()

# Delay between CLOB calls to be rate-limit friendly (milliseconds)
CLOB_DELAY_MS = 100


async def main(
    include_spreads: bool = False,
    include_totals: bool = False,
    dry_run: bool = False,
):
    """
    Main backfill execution flow.

    Args:
        include_spreads: Whether to backfill spread market histories
        include_totals: Whether to backfill total market histories
        dry_run: Simulate execution without storing data

    Flow:
        1. Fetch all closed NBA events from Gamma API (with pagination)
        2. For each event, check if moneyline market needs backfill
        3. Fetch price history from CLOB API
        4. Bulk store in database
        5. Log results
    """
    app_settings = get_settings()

    if not app_settings.polymarket.enabled:
        logger.info("polymarket_backfill_skipped", reason="polymarket.enabled=False")
        return

    logger.info(
        "polymarket_backfill_started",
        include_spreads=include_spreads,
        include_totals=include_totals,
        dry_run=dry_run,
    )

    # Statistics tracking
    stats = {
        "events_processed": 0,
        "markets_backfilled": 0,
        "markets_skipped": 0,
        "total_price_points": 0,
        "errors": 0,
        "events_with_no_data": 0,
    }

    try:
        async with PolymarketClient() as client, async_session_maker() as session:
            reader = PolymarketReader(session)
            writer = PolymarketWriter(session)

            # Get market IDs that already have sufficient backfilled data
            backfilled_market_ids = await reader.get_backfilled_market_ids(min_snapshots=10)
            logger.info(
                "backfilled_markets_identified", already_backfilled=len(backfilled_market_ids)
            )

            # Fetch closed events with pagination
            events = await _fetch_all_closed_events(client)

            if not events:
                logger.info("no_closed_events_found")
                return

            logger.info("closed_events_fetched", total_events=len(events))

            # Process each event
            for event_data in events:
                try:
                    stats["events_processed"] += 1

                    # Upsert event to database
                    event = await writer.upsert_event(event_data)
                    markets_data = event_data.get("markets", [])

                    if not markets_data:
                        logger.warning(
                            "event_has_no_markets",
                            pm_event_id=event.pm_event_id,
                            title=event.title,
                        )
                        continue

                    # Upsert markets
                    markets = await writer.upsert_markets(
                        pm_event_id=event.id, markets_data=markets_data, event_title=event.title
                    )

                    # Determine which market types to process
                    types_to_process = [PolymarketMarketType.MONEYLINE]
                    if include_spreads:
                        types_to_process.append(PolymarketMarketType.SPREAD)
                    if include_totals:
                        types_to_process.append(PolymarketMarketType.TOTAL)

                    # Process markets by type
                    for market_type in types_to_process:
                        markets_of_type = [m for m in markets if m.market_type == market_type]

                        for market in markets_of_type:
                            result = await _backfill_market_history(
                                client=client,
                                writer=writer,
                                market=market,
                                commence_time=event.start_date,
                                backfilled_market_ids=backfilled_market_ids,
                                dry_run=dry_run,
                            )

                            if result["status"] == "success":
                                stats["markets_backfilled"] += 1
                                stats["total_price_points"] += result["points"]
                            elif result["status"] == "skipped":
                                stats["markets_skipped"] += 1
                            elif result["status"] == "no_data":
                                stats["events_with_no_data"] += 1
                            elif result["status"] == "error":
                                stats["errors"] += 1

                except Exception as e:
                    logger.error(
                        "event_processing_failed",
                        pm_event_id=event_data.get("id"),
                        title=event_data.get("title"),
                        error=str(e),
                        exc_info=True,
                    )
                    stats["errors"] += 1
                    # Continue to next event rather than failing entire job

            # Commit changes if not dry run
            if not dry_run:
                await session.commit()
                logger.info("polymarket_backfill_committed")

        # Log fetch operation
        fetch_log = PolymarketFetchLog(
            job_type="backfill-polymarket",
            events_count=stats["events_processed"],
            markets_count=stats["markets_backfilled"],
            snapshots_stored=stats["total_price_points"],
            success=True,
            error_message=None,
        )

        if not dry_run:
            async with async_session_maker() as log_session:
                log_writer = PolymarketWriter(log_session)
                await log_writer.log_fetch(fetch_log)
                await log_session.commit()

        logger.info(
            "polymarket_backfill_completed",
            events_processed=stats["events_processed"],
            markets_backfilled=stats["markets_backfilled"],
            markets_skipped=stats["markets_skipped"],
            total_price_points=stats["total_price_points"],
            events_with_no_data=stats["events_with_no_data"],
            errors=stats["errors"],
        )

    except Exception as e:
        logger.error("polymarket_backfill_failed", error=str(e), exc_info=True)

        # Log failed fetch
        fetch_log = PolymarketFetchLog(
            job_type="backfill-polymarket",
            events_count=stats["events_processed"],
            markets_count=stats["markets_backfilled"],
            snapshots_stored=stats["total_price_points"],
            success=False,
            error_message=str(e),
        )

        if not dry_run:
            async with async_session_maker() as log_session:
                log_writer = PolymarketWriter(log_session)
                await log_writer.log_fetch(fetch_log)
                await log_session.commit()

        raise


async def _fetch_all_closed_events(client: PolymarketClient) -> list[dict]:
    """
    Fetch all closed NBA events from Gamma API with pagination.

    Args:
        client: PolymarketClient instance

    Returns:
        List of all closed event dicts
    """
    all_events = []
    offset = 0
    limit = 100

    while True:
        events = await client.get_nba_events(active=False, closed=True, limit=limit, offset=offset)

        if not events:
            # Empty response means we've exhausted pagination
            break

        all_events.extend(events)
        offset += len(events)

        logger.info("pagination_batch_fetched", offset=offset, batch_size=len(events))

        # If we got fewer than limit, we've reached the end
        if len(events) < limit:
            break

    return all_events


async def _backfill_market_history(
    client: PolymarketClient,
    writer: PolymarketWriter,
    market: PolymarketMarket,
    commence_time: datetime,
    backfilled_market_ids: set[int],
    dry_run: bool,
) -> dict:
    """
    Backfill price history for a single market.

    Args:
        client: PolymarketClient instance
        writer: PolymarketWriter instance
        market: PolymarketMarket instance
        commence_time: Event commence time
        backfilled_market_ids: Set of market IDs already backfilled
        dry_run: Whether this is a dry run

    Returns:
        Dict with status and points count: {"status": "success|skipped|no_data|error", "points": int}
    """
    # Skip if already backfilled
    if market.id in backfilled_market_ids:
        logger.info(
            "market_history_skipped",
            market_id=market.id,
            question=market.question,
            reason="already_backfilled",
        )
        return {"status": "skipped", "points": 0}

    # Get token_id (use first outcome token - typically the "Yes" side for moneyline)
    if not market.clob_token_ids:
        logger.warning(
            "market_has_no_tokens",
            market_id=market.id,
            question=market.question,
        )
        return {"status": "error", "points": 0}

    token_id = market.clob_token_ids[0]

    try:
        # Fetch price history from CLOB
        history = await client.get_price_history(
            token_id=token_id,
            interval="max",
            fidelity=5,  # 5-minute resolution
        )

        # Rate-limit delay after CLOB API call (applies to both dry-run and normal mode)
        await asyncio.sleep(CLOB_DELAY_MS / 1000)

        if not history:
            logger.info(
                "market_history_empty",
                market_id=market.id,
                question=market.question,
                reason="no_data_available",
            )
            return {"status": "no_data", "points": 0}

        # Store history if not dry run
        if not dry_run:
            inserted = await writer.bulk_store_price_history(
                market=market, history=history, commence_time=commence_time
            )

            logger.info(
                "market_history_backfilled",
                market_id=market.id,
                question=market.question,
                points_fetched=len(history),
                points_inserted=inserted,
            )

            return {"status": "success", "points": inserted}
        else:
            logger.info(
                "market_history_dry_run",
                market_id=market.id,
                question=market.question,
                points_would_fetch=len(history),
            )
            return {"status": "success", "points": len(history)}

    except Exception as e:
        logger.error(
            "market_history_fetch_failed",
            market_id=market.id,
            question=market.question,
            error=str(e),
        )
        return {"status": "error", "points": 0}


if __name__ == "__main__":
    asyncio.run(main())
