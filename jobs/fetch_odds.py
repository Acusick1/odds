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

from core.config import Settings, get_settings
from core.ingestion import OddsIngestionService
from core.scheduling.backends import get_scheduler_backend
from core.scheduling.intelligence import SchedulingIntelligence

logger = structlog.get_logger()


def build_ingestion_service(settings: Settings) -> OddsIngestionService:
    """Factory to create ingestion service; exposed for test patching."""
    return OddsIngestionService(settings=settings)


async def main():
    """
    Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: fetch and store odds
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    app_settings = get_settings()

    logger.info("fetch_odds_job_started", backend=app_settings.scheduler_backend)

    # Smart execution gating
    intelligence = SchedulingIntelligence(lookahead_days=app_settings.scheduling_lookahead_days)
    decision = await intelligence.should_execute_fetch()

    if not decision.should_execute:
        logger.info(
            "fetch_odds_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )

        # Still schedule next check even if not executing
        if decision.next_execution:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler_dry_run)
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

    ingestion_service = build_ingestion_service(app_settings)

    try:
        results = await ingestion_service.ingest_sports(
            app_settings.sports,
            fetch_tier=decision.tier,
        )

        for sport_result in results.sport_results:
            for failure in sport_result.failures:
                logger.warning(
                    "event_processing_failed",
                    sport=sport_result.sport_key,
                    event_id=failure.event_id,
                    error=failure.error,
                )

            logger.info(
                "sport_fetch_completed",
                sport=sport_result.sport_key,
                processed_events=sport_result.processed_events,
                total_events=sport_result.total_events,
                quota_remaining=sport_result.quota_remaining,
            )

            if sport_result.quota_remaining is not None and sport_result.quota_remaining < (
                app_settings.odds_api_quota * 0.2
            ):
                percentage_remaining = round(
                    sport_result.quota_remaining / app_settings.odds_api_quota * 100,
                    1,
                )
                logger.warning(
                    "api_quota_low",
                    sport=sport_result.sport_key,
                    remaining=sport_result.quota_remaining,
                    quota=app_settings.odds_api_quota,
                    percentage=percentage_remaining,
                )

        logger.info(
            "fetch_odds_completed",
            total_processed=results.total_processed,
            total_events=results.total_events,
            total_failures=results.total_failures,
        )

    except Exception as e:
        logger.error("fetch_odds_failed", error=str(e), exc_info=True)
        # Don't schedule next run if we failed - let manual intervention happen
        raise

    # Self-schedule next execution
    if decision.next_execution:
        try:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler_dry_run)
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


if __name__ == "__main__":
    asyncio.run(main())
