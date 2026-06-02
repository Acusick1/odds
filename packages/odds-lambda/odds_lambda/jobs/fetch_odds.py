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
from odds_core.config import Settings, get_settings

from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.event_sync import EventSyncService
from odds_lambda.ingestion import OddsIngestionService
from odds_lambda.scheduling.decision import (
    CadenceConfig,
    ScheduleDecision,
    decide_forward_resilient,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()

# Proximity-based polling cadence (hours) for the cheap /odds API. Matches the
# canonical FetchTier intervals odds fetch has always used.
CADENCE = CadenceConfig(
    closing=0.5,
    pregame=3.0,
    sharp=12.0,
    early=24.0,
    opening=48.0,
    no_game=24.0,
)


def build_ingestion_service(client: TheOddsAPIClient, settings: Settings) -> OddsIngestionService:
    """Factory to create ingestion service; exposed for test patching."""
    return OddsIngestionService(client=client, settings=settings)


def build_event_sync_service(client: TheOddsAPIClient) -> EventSyncService:
    """Factory to create event sync service; exposed for test patching."""
    return EventSyncService(client=client)


async def main(ctx: JobContext) -> None:
    """Main job execution flow.

    Flow:
    1. Sync upcoming events from free /events endpoint (discovery)
    2. Check if we should execute odds fetch (smart gating)
    3. If yes: fetch and store odds
    4. Calculate next execution time
    5. Schedule next run via backend
    """
    app_settings = get_settings()
    sport = ctx.sport
    sports = [sport] if sport else app_settings.data_collection.sports

    logger.info(
        "fetch_odds_job_started",
        backend=app_settings.scheduler.backend,
        sport=sport,
        sports=sports,
    )

    if not sports:
        logger.warning("fetch_odds_no_sports_configured")
        return

    # Sync upcoming events first (free, 0 quota units).
    # Failures here should not block odds ingestion.
    try:
        async with TheOddsAPIClient() as client:
            event_sync = build_event_sync_service(client)
            sync_results = await event_sync.sync_sports(sports)

        for sync_result in sync_results:
            logger.info(
                "event_sync_result",
                sport=sync_result.sport_key,
                inserted=sync_result.inserted,
                updated=sync_result.updated,
            )
    except Exception as e:
        logger.warning("event_sync_failed", error=str(e), exc_info=True)

    # Self-scheduling uses the sport-suffixed job name to match Terraform rules
    schedule_job_name = make_compound_job_name("fetch-odds", sport)

    async def _schedule(decision: ScheduleDecision, *, label: str) -> None:
        if not decision.next_execution:
            return
        try:
            await self_schedule(
                job_name=schedule_job_name,
                next_time=decision.next_execution,
                dry_run=app_settings.scheduler.dry_run,
                sport=sport,
                interval_hours=None,
                reason=f"{label}: {decision.reason}",
            )
        except Exception as e:
            logger.error("fetch_odds_scheduling_failed", error=str(e), exc_info=True)
            from odds_core.alerts import send_error

            await send_error(f"Fetch odds scheduling failed: {type(e).__name__}: {str(e)}")

    # Smart execution gating, scoped to this job's sport(s) — fixes the prior
    # all-sports bug where another sport's fixtures kept this job alive. The
    # resilient path keys off the soonest kickoff across this job's sports and
    # falls back to db_fallback cadence (should_execute=True, burning one cheap
    # /odds call) if the kickoff query fails, so the chain never breaks.
    decision = await decide_forward_resilient(
        sports,
        CADENCE,
        lookahead_days=app_settings.scheduler.lookahead_days,
    )

    # Pre-schedule before any work so the chain survives crashes.
    await _schedule(decision, label="pre-schedule")

    if not decision.should_execute:
        logger.info(
            "fetch_odds_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )
        return

    # Execute fetch logic
    logger.info(
        "fetch_odds_executing",
        reason=decision.reason,
        tier=decision.tier.value if decision.tier else None,
    )

    try:
        async with TheOddsAPIClient() as client:
            ingestion_service = build_ingestion_service(client, app_settings)
            results = await ingestion_service.ingest_sports(sports)

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

            # Check quota thresholds and send tiered alerts
            if sport_result.quota_remaining is not None:
                quota_fraction = sport_result.quota_remaining / app_settings.api.quota
                percentage_remaining = round(quota_fraction * 100, 1)

                # Critical threshold (default 10%)
                if quota_fraction < app_settings.alerts.quota_critical_threshold:
                    logger.error(
                        "api_quota_critical",
                        sport=sport_result.sport_key,
                        remaining=sport_result.quota_remaining,
                        quota=app_settings.api.quota,
                        percentage=percentage_remaining,
                    )

                    # Send critical alert
                    from odds_core.alerts import send_critical

                    await send_critical(
                        f"🚨 API quota critical: {sport_result.quota_remaining} requests remaining "
                        f"({percentage_remaining}% of {app_settings.api.quota})"
                    )

                # Warning threshold (default 20%)
                elif quota_fraction < app_settings.alerts.quota_warning_threshold:
                    logger.warning(
                        "api_quota_low",
                        sport=sport_result.sport_key,
                        remaining=sport_result.quota_remaining,
                        quota=app_settings.api.quota,
                        percentage=percentage_remaining,
                    )

                    # Send warning alert
                    from odds_core.alerts import send_warning

                    await send_warning(
                        f"⚠️ API quota low: {sport_result.quota_remaining} requests remaining "
                        f"({percentage_remaining}% of {app_settings.api.quota})"
                    )

        logger.info(
            "fetch_odds_completed",
            total_processed=results.total_processed,
            total_events=results.total_events,
            total_failures=results.total_failures,
        )

    except Exception as e:
        logger.error("fetch_odds_failed", error=str(e), exc_info=True)

        # Send critical alert
        from odds_core.alerts import send_critical

        await send_critical(f"🚨 Fetch odds job failed: {type(e).__name__}: {str(e)}")

        # Pre-schedule already ran, so the chain survives. Re-raise for visibility.
        raise

    # Re-query and reschedule — ingestion may have created new events that move
    # the proximity tier.
    post_decision = await decide_forward_resilient(
        sports,
        CADENCE,
        lookahead_days=app_settings.scheduler.lookahead_days,
    )
    await _schedule(post_decision, label="post-fetch")
    logger.info(
        "fetch_odds_next_scheduled",
        next_time=post_decision.next_execution.isoformat()
        if post_decision.next_execution
        else None,
        tier=post_decision.tier.value if post_decision.tier else None,
    )


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
