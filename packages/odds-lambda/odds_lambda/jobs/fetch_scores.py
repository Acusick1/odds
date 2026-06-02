"""
Fetch scores job - update final scores for completed games.

This job:
1. Checks if score updates are needed
2. Fetches scores from API for last 3 days
3. Updates events with final scores and status
4. Self-schedules next execution
"""

import asyncio
from datetime import timedelta

import structlog
from odds_core.api_models import parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import EventStatus

from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.scheduling.decision import ScheduleDecision, decide_backward_resilient
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Recent-games window and cadence (hours) for score backfill.
SCORES_WINDOW = timedelta(days=3)
SCORES_ACTIVE_INTERVAL = 6.0
SCORES_IDLE_INTERVAL = 12.0


async def _scores_decision(sports: list[str]) -> ScheduleDecision:
    """Run if any configured sport has a recent game still lacking a final score.

    Resilient to DB failure: a recent-games query error falls back to running at
    the db_fallback cadence so the self-scheduling chain survives a DB outage.
    """
    return await decide_backward_resilient(
        sports,
        window=SCORES_WINDOW,
        active_interval=SCORES_ACTIVE_INTERVAL,
        idle_interval=SCORES_IDLE_INTERVAL,
    )


async def main(ctx: JobContext) -> None:
    """Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: fetch and update scores
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    app_settings = get_settings()
    sport = ctx.sport
    sports = [sport] if sport else app_settings.data_collection.sports

    logger.info(
        "fetch_scores_job_started",
        backend=app_settings.scheduler.backend,
        sport=sport,
        sports=sports,
    )

    if not sports:
        logger.warning("fetch_scores_no_sports_configured")
        return

    # Self-scheduling uses the sport-suffixed job name to match Terraform rules
    schedule_job_name = make_compound_job_name("fetch-scores", sport)

    async def _schedule(decision: ScheduleDecision, *, label: str) -> None:
        if not decision.next_execution:
            return
        try:
            await self_schedule(
                job_name=schedule_job_name,
                next_time=decision.next_execution,
                dry_run=app_settings.scheduler.dry_run,
                sport=sport,
                reason=f"{label}: {decision.reason}",
            )
        except Exception as e:
            logger.error("fetch_scores_scheduling_failed", error=str(e), exc_info=True)
            from odds_core.alerts import send_error

            await send_error(f"Fetch scores scheduling failed: {type(e).__name__}: {str(e)}")

    # Smart execution gating, scoped to this job's sport(s).
    decision = await _scores_decision(sports)

    # Pre-schedule before any work so the chain survives crashes.
    await _schedule(decision, label="pre-schedule")

    if not decision.should_execute:
        logger.info(
            "fetch_scores_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )
        return

    # Execute fetch logic
    logger.info("fetch_scores_executing", reason=decision.reason)

    try:
        await _fetch_and_update_scores(sports)
        logger.info("fetch_scores_completed")

    except Exception as e:
        logger.error("fetch_scores_failed", error=str(e), exc_info=True)

        # Send critical alert
        from odds_core.alerts import send_critical

        await send_critical(f"🚨 Fetch scores job failed: {type(e).__name__}: {str(e)}")

        raise

    # Re-evaluate and reschedule after fetching scores.
    post_decision = await _scores_decision(sports)
    await _schedule(post_decision, label="post-fetch")
    logger.info(
        "fetch_scores_next_scheduled",
        next_time=post_decision.next_execution.isoformat()
        if post_decision.next_execution
        else None,
    )


async def _fetch_and_update_scores(sports: list[str]) -> None:
    """Core score fetching and update logic."""
    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for sport_key in sports:
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
    asyncio.run(main(JobContext()))
