"""
Fetch scores job - update final scores for completed games.

This job:
1. Checks if score updates are needed
2. Fetches scores from API for last 3 days
3. Updates events with final scores and status
4. Self-schedules next execution
"""

from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import structlog
from odds_core.alerts import job_alert_context
from odds_core.api_models import parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import EventStatus

from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.odds_api_events import find_event_by_odds_api_id
from odds_lambda.scheduling.decision import ScheduleDecision, decide_backward_resilient
from odds_lambda.scheduling.helpers import book_next_run
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
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
    sport = require_sport(ctx, "fetch-scores")
    sports = [sport]

    logger.info(
        "fetch_scores_job_started",
        backend=app_settings.scheduler.backend,
        sport=sport,
        sports=sports,
    )

    # Self-scheduling uses the sport-suffixed job name to match Terraform rules
    schedule_job_name = make_compound_job_name("fetch-scores", sport)

    schedule = partial(
        book_next_run,
        job_name=schedule_job_name,
        sport=sport,
        dry_run=app_settings.scheduler.dry_run,
    )

    # Smart execution gating, scoped to this job's sport(s).
    decision = await _scores_decision(sports)

    # Pre-schedule before any work so the chain survives crashes.
    pre_booked = await schedule(decision, label="pre-schedule", final=not decision.should_execute)

    async with job_alert_context(schedule_job_name):
        if not decision.should_execute:
            logger.info(
                "fetch_scores_skipped",
                reason=decision.reason,
                next_check=decision.next_execution.isoformat() if decision.next_execution else None,
            )
            return

        logger.info("fetch_scores_executing", reason=decision.reason)

        try:
            await update_scores(sport)
            logger.info("fetch_scores_completed")

        except Exception as e:
            logger.error("fetch_scores_failed", error=str(e), exc_info=True)
            raise

    # Re-evaluate and reschedule after fetching scores.
    post_decision = await _scores_decision(sports)
    await schedule(post_decision, label="post-fetch", final=not pre_booked)
    logger.info(
        "fetch_scores_next_scheduled",
        next_time=post_decision.next_execution.isoformat()
        if post_decision.next_execution
        else None,
    )


@dataclass(frozen=True, slots=True)
class ScoresUpdate:
    """Events the Odds API listed with scores, and how many of them were stored as final."""

    listed: int
    updated: int


async def update_scores(sport: str, *, days: int = SCORES_WINDOW.days) -> ScoresUpdate:
    """Store the final score of every game of ``sport`` the Odds API lists as completed in the
    last ``days`` days.

    A game whose Odds API id names no stored event, or whose update fails, is logged and skipped.
    """
    logger.info("fetching_scores", sport=sport, days=days)
    async with TheOddsAPIClient() as client:
        response = await client.get_scores(sport=sport, days_from=days)

    updated = 0
    async with async_session_maker() as session:
        writer = OddsWriter(session)
        for score_data in response.scores_data:
            odds_api_id = score_data.get("id")
            if not (score_data.get("completed", False) and odds_api_id):
                continue
            try:
                home_score, away_score = parse_scores_from_api_dict(score_data)
                if home_score is None or away_score is None:
                    continue
                event = await find_event_by_odds_api_id(session, odds_api_id)
                if event is None:
                    logger.warning("score_event_not_found", odds_api_id=odds_api_id)
                    continue
                await writer.update_event_status(
                    event_id=event.id,
                    status=EventStatus.FINAL,
                    home_score=home_score,
                    away_score=away_score,
                )
                updated += 1
            except Exception as e:
                logger.error("score_update_failed", odds_api_id=odds_api_id, error=str(e))
        await session.commit()

    logger.info("sport_scores_completed", sport=sport, updated=updated)
    return ScoresUpdate(listed=len(response.scores_data), updated=updated)
