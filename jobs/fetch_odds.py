"""
Fetch odds job - standalone executable or Lambda handler.

This job:
1. Checks if odds fetch should execute (game-aware gating)
2. Fetches current odds for all upcoming games
3. Stores raw snapshots and normalized data
4. Self-schedules next execution based on game proximity
"""

import asyncio

import structlog

from core.config import settings
from core.data_fetcher import TheOddsAPIClient
from core.database import async_session_maker
from core.models import FetchLog
from core.scheduling.backends import get_scheduler_backend
from core.scheduling.intelligence import SchedulingIntelligence
from storage.writers import OddsWriter

logger = structlog.get_logger()


async def main():
    """
    Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: fetch and store odds
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    logger.info("fetch_odds_job_started", backend=settings.scheduler_backend)

    # Smart execution gating
    intelligence = SchedulingIntelligence(lookahead_days=settings.scheduling_lookahead_days)
    decision = await intelligence.should_execute_fetch()

    if not decision.should_execute:
        logger.info(
            "fetch_odds_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )

        # Still schedule next check even if not executing
        if decision.next_execution:
            backend = get_scheduler_backend(dry_run=settings.scheduler_dry_run)
            await backend.schedule_next_execution(
                job_name="fetch-odds", next_time=decision.next_execution
            )

        return

    # Execute fetch logic
    logger.info(
        "fetch_odds_executing",
        reason=decision.reason,
        tier=decision.tier.value if decision.tier else None,
    )

    try:
        await _fetch_and_store_odds()
        logger.info("fetch_odds_completed")

    except Exception as e:
        logger.error("fetch_odds_failed", error=str(e), exc_info=True)
        # Don't schedule next run if we failed - let manual intervention happen
        raise

    # Self-schedule next execution
    if decision.next_execution:
        try:
            backend = get_scheduler_backend(dry_run=settings.scheduler_dry_run)
            await backend.schedule_next_execution(
                job_name="fetch-odds", next_time=decision.next_execution
            )
            logger.info(
                "fetch_odds_next_scheduled",
                next_time=decision.next_execution.isoformat(),
                backend=backend.get_backend_name(),
                tier=decision.tier.value if decision.tier else None,
            )
        except Exception as e:
            logger.error("fetch_odds_scheduling_failed", error=str(e), exc_info=True)
            # Don't fail the job if scheduling fails - the fetch itself succeeded


async def _fetch_and_store_odds():
    """
    Core fetch and storage logic.

    Adapted from scheduler/jobs.py:fetch_odds_job()
    """
    # Fetch odds for each configured sport
    for sport_key in settings.sports:
        logger.info("fetching_odds", sport=sport_key)

        async with TheOddsAPIClient() as client:
            # Fetch current odds - API client returns parsed Event instances
            response = await client.get_odds(
                sport=sport_key,
                regions=settings.regions,
                markets=settings.markets,
                bookmakers=settings.bookmakers,
            )

            # Process each event in its own transaction
            processed_count = 0
            for event, event_data in zip(response.events, response.raw_events_data, strict=True):
                try:
                    async with async_session_maker() as event_session:
                        event_writer = OddsWriter(event_session)

                        # Upsert event - already parsed by API client
                        await event_writer.upsert_event(event)

                        # Flush to ensure event exists before quality logging
                        await event_session.flush()

                        # Store odds snapshot with raw data
                        await event_writer.store_odds_snapshot(
                            event_id=event.id,
                            raw_data=event_data,
                            snapshot_time=response.timestamp,
                            validate=settings.enable_validation,
                        )

                        await event_session.commit()
                        processed_count += 1

                except Exception as e:
                    logger.error(
                        "event_processing_failed",
                        event_id=event.id,
                        error=str(e),
                    )
                    continue

            # Log successful fetch in separate transaction
            async with async_session_maker() as log_session:
                log_writer = OddsWriter(log_session)
                fetch_log = FetchLog(
                    sport_key=sport_key,
                    events_count=len(response.events),
                    bookmakers_count=len(settings.bookmakers),
                    success=True,
                    api_quota_remaining=response.quota_remaining,
                    response_time_ms=response.response_time_ms,
                )
                await log_writer.log_fetch(fetch_log)
                await log_session.commit()

            logger.info(
                "sport_fetch_completed",
                sport=sport_key,
                events=processed_count,
                quota_remaining=response.quota_remaining,
            )

            # Warn if quota is running low
            if response.quota_remaining and response.quota_remaining < (
                settings.odds_api_quota * 0.2
            ):
                logger.warning(
                    "api_quota_low",
                    remaining=response.quota_remaining,
                    quota=settings.odds_api_quota,
                    percentage=round(response.quota_remaining / settings.odds_api_quota * 100, 1),
                )


if __name__ == "__main__":
    asyncio.run(main())
