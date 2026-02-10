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
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.polymarket_models import PolymarketFetchLog

from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.polymarket_ingestion import PolymarketIngestionResult, PolymarketIngestionService
from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.storage.polymarket_reader import PolymarketReader
from odds_lambda.storage.polymarket_writer import PolymarketWriter
from odds_lambda.tier_utils import calculate_tier

logger = structlog.get_logger()


def build_ingestion_service(
    client: PolymarketClient,
    reader: PolymarketReader,
    writer: PolymarketWriter,
) -> PolymarketIngestionService:
    """Factory to create ingestion service; exposed for test patching."""
    return PolymarketIngestionService(
        client=client,
        reader=reader,
        writer=writer,
        settings=get_settings(),
    )


async def _schedule(job_name: str, next_time: datetime, *, dry_run: bool) -> None:
    """Schedule next execution via the configured backend."""
    backend = get_scheduler_backend(dry_run=dry_run)
    await backend.schedule_next_execution(job_name=job_name, next_time=next_time)


async def _handle_no_api_events(
    reader: PolymarketReader,
    *,
    price_poll_interval: int,
    dry_run: bool,
) -> None:
    """When API returns no events, schedule based on DB state or daily fallback."""
    db_active_events = await reader.get_active_events()
    if db_active_events:
        closest_game = min(event.start_date for event in db_active_events)
        now = datetime.now(UTC)
        hours_until = (closest_game - now).total_seconds() / 3600
        fetch_tier = calculate_tier(hours_until)
        next_execution = now + timedelta(seconds=price_poll_interval)
        logger.info(
            "fetch_polymarket_scheduled_from_db",
            closest_game=closest_game.isoformat(),
            hours_until=round(hours_until, 2),
            tier=fetch_tier.value,
        )
    else:
        next_execution = datetime.now(UTC) + timedelta(hours=24)
        logger.info("fetch_polymarket_scheduled_daily_check")

    await _schedule("fetch-polymarket", next_execution, dry_run=dry_run)


async def _log_fetch(
    result: PolymarketIngestionResult, *, success: bool, error_message: str | None = None
) -> None:
    """Persist a PolymarketFetchLog on a dedicated session."""
    fetch_log = PolymarketFetchLog(
        job_type="fetch-polymarket",
        events_count=result.events_processed,
        markets_count=result.markets_discovered,
        snapshots_stored=result.total_snapshots,
        success=success,
        error_message=error_message,
    )
    async with async_session_maker() as log_session:
        log_writer = PolymarketWriter(log_session)
        await log_writer.log_fetch(fetch_log)
        await log_session.commit()


async def main() -> None:
    """Main job execution flow."""
    app_settings = get_settings()

    logger.info("fetch_polymarket_job_started", backend=app_settings.scheduler.backend)

    if not app_settings.polymarket.enabled:
        logger.info("fetch_polymarket_skipped", reason="polymarket.enabled=False")
        return

    result = PolymarketIngestionResult()

    try:
        async with PolymarketClient() as client, async_session_maker() as session:
            reader = PolymarketReader(session)
            writer = PolymarketWriter(session)
            service = build_ingestion_service(client, reader, writer)

            # Discover active NBA events from Gamma API
            logger.info("discovering_active_polymarket_events")
            events_data = await client.get_nba_events(active=True, closed=False)

            if not events_data:
                logger.info(
                    "fetch_polymarket_skipped",
                    reason="no_active_polymarket_events_from_api",
                )
                await _handle_no_api_events(
                    reader,
                    price_poll_interval=app_settings.polymarket.price_poll_interval,
                    dry_run=app_settings.scheduler.dry_run,
                )
                return

            logger.info("active_polymarket_events_discovered", count=len(events_data))

            # Phase 1: Upsert events + markets
            discovery_result = await service.discover_and_upsert_events(events_data)
            result.events_processed = discovery_result.events_processed
            result.markets_discovered = discovery_result.markets_discovered
            result.errors = discovery_result.errors

            # Phase 2: Collect price/orderbook snapshots
            snapshot_result = await service.collect_snapshots()
            result.price_snapshots_stored = snapshot_result.price_snapshots_stored
            result.orderbook_snapshots_stored = snapshot_result.orderbook_snapshots_stored
            result.errors += snapshot_result.errors
            result.fetch_tier = snapshot_result.fetch_tier

            if result.fetch_tier is None:
                # No active events after upsert — schedule daily check
                next_execution = datetime.now(UTC) + timedelta(hours=24)
                await _schedule(
                    "fetch-polymarket", next_execution, dry_run=app_settings.scheduler.dry_run
                )
                return

        logger.info(
            "fetch_polymarket_completed",
            events_processed=result.events_processed,
            markets_discovered=result.markets_discovered,
            price_snapshots_stored=result.price_snapshots_stored,
            orderbook_snapshots_stored=result.orderbook_snapshots_stored,
            errors=result.errors,
        )

        await _log_fetch(result, success=True)

    except Exception as e:
        logger.error("fetch_polymarket_failed", error=str(e), exc_info=True)

        await _log_fetch(result, success=False, error_message=str(e))

        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            await send_critical(f"🚨 Fetch Polymarket job failed: {type(e).__name__}: {str(e)}")

        raise

    # Self-schedule next execution
    if result.fetch_tier:
        try:
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
                tier=result.fetch_tier.value,
            )
        except Exception as e:
            logger.error("fetch_polymarket_scheduling_failed", error=str(e), exc_info=True)

            if app_settings.alerts.alert_enabled:
                from odds_cli.alerts.base import send_error

                await send_error(
                    f"Fetch Polymarket scheduling failed: {type(e).__name__}: {str(e)}"
                )


if __name__ == "__main__":
    asyncio.run(main())
