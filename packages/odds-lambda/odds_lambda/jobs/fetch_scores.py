"""
Fetch scores job - update final scores for completed games.

This job:
1. Checks if score updates are needed
2. Fetches scores from API for last 3 days
3. Updates events with final scores and status
4. Self-schedules next execution
"""

import asyncio

import structlog
from odds_core.api_models import parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import EventStatus

from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.scheduling.intelligence import SchedulingIntelligence
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()


async def main():
    """
    Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: fetch and update scores
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    app_settings = get_settings()

    logger.info("fetch_scores_job_started", backend=app_settings.scheduler.backend)

    # Smart execution gating
    intelligence = SchedulingIntelligence(lookahead_days=app_settings.scheduler.lookahead_days)
    decision = await intelligence.should_execute_scores()

    if not decision.should_execute:
        logger.info(
            "fetch_scores_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )

        # Still schedule next check even if not executing
        if decision.next_execution:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
            await backend.schedule_next_execution(
                job_name="fetch-scores", next_time=decision.next_execution
            )

        return

    # Execute fetch logic
    logger.info("fetch_scores_executing", reason=decision.reason)

    try:
        await _fetch_and_update_scores(app_settings)
        logger.info("fetch_scores_completed")

    except Exception as e:
        logger.error("fetch_scores_failed", error=str(e), exc_info=True)

        # Send critical alert
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            await send_critical(f"🚨 Fetch scores job failed: {type(e).__name__}: {str(e)}")

        raise

    # Self-schedule next execution
    if decision.next_execution:
        try:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
            await backend.schedule_next_execution(
                job_name="fetch-scores", next_time=decision.next_execution
            )
            logger.info(
                "fetch_scores_next_scheduled",
                next_time=decision.next_execution.isoformat(),
                backend=backend.get_backend_name(),
            )
        except Exception as e:
            logger.error("fetch_scores_scheduling_failed", error=str(e), exc_info=True)

            # Send error alert
            if app_settings.alerts.alert_enabled:
                from odds_cli.alerts.base import send_error

                await send_error(f"Fetch scores scheduling failed: {type(e).__name__}: {str(e)}")


async def _fetch_and_update_scores(app_settings):
    """
    Core score fetching and update logic.

    Adapted from scheduler/jobs.py:fetch_scores_job()
    """
    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for sport_key in app_settings.data_collection.sports:
            logger.info("fetching_scores", sport=sport_key)

            async with TheOddsAPIClient() as client:
                # Fetch scores from last 3 days
                response = await client.get_scores(sport=sport_key, days_from=3)

            updated_count = 0
            for score_data in response.scores_data:
                try:
                    event_id = score_data.get("id")
                    completed = score_data.get("completed", False)

                    if completed and event_id:
                        # Extract home and away scores using helper
                        home_score, away_score = parse_scores_from_api_dict(score_data)

                        if home_score is not None and away_score is not None:
                            await writer.update_event_status(
                                event_id=event_id,
                                status=EventStatus.FINAL,
                                home_score=home_score,
                                away_score=away_score,
                            )
                            updated_count += 1

                except Exception as e:
                    logger.error(
                        "score_update_failed",
                        event_id=score_data.get("id"),
                        error=str(e),
                    )
                    continue

            await session.commit()

            logger.info(
                "sport_scores_completed",
                sport=sport_key,
                updated=updated_count,
            )


if __name__ == "__main__":
    asyncio.run(main())
