"""
Update event status job - mark games as live/final.

This job:
1. Checks if status updates are needed
2. Finds scheduled games that may have started
3. Updates event status (SCHEDULED → LIVE)
4. Self-schedules next execution
"""

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import EventStatus

from odds_lambda.scheduling.decision import ScheduleDecision, decide_backward_resilient
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Recent-games window and cadence (hours) for status flips (SCHEDULED -> LIVE).
STATUS_WINDOW = timedelta(hours=4)
STATUS_ACTIVE_INTERVAL = 1.0
STATUS_IDLE_INTERVAL = 6.0


async def _status_decision(sports: list[str]) -> ScheduleDecision:
    """Run if any configured sport has a recently-started SCHEDULED game.

    Resilient to DB failure: a recent-games query error falls back to running at
    the db_fallback cadence so the self-scheduling chain survives a DB outage.
    """
    return await decide_backward_resilient(
        sports,
        window=STATUS_WINDOW,
        active_interval=STATUS_ACTIVE_INTERVAL,
        idle_interval=STATUS_IDLE_INTERVAL,
        statuses_needing_update={EventStatus.SCHEDULED},
    )


async def main(ctx: JobContext) -> None:
    """Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: update event statuses
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    app_settings = get_settings()
    sport = ctx.sport
    sports = [sport] if sport else app_settings.data_collection.sports

    logger.info("update_status_job_started", backend=app_settings.scheduler.backend, sport=sport)

    if not sports:
        logger.warning("update_status_no_sports_configured")
        return

    schedule_job_name = make_compound_job_name("update-status", sport)

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
            logger.error("update_status_scheduling_failed", error=str(e), exc_info=True)
            from odds_core.alerts import send_error

            await send_error(f"Update status scheduling failed: {type(e).__name__}: {e}")

    # Smart execution gating, scoped to this job's sport(s).
    decision = await _status_decision(sports)

    # Pre-schedule before any work so the chain survives crashes.
    await _schedule(decision, label="pre-schedule")

    # ``respect_gate=False`` (deploy smoke) forces the full body even when not due.
    if not decision.should_execute and ctx.policy.respect_gate:
        logger.info(
            "update_status_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )
        return

    # Execute update logic
    from odds_core.alerts import job_alert_context

    logger.info("update_status_executing", reason=decision.reason)

    async with job_alert_context(schedule_job_name):
        await _update_event_statuses()
        logger.info("update_status_completed")

    # Re-evaluate and reschedule after the status flip.
    post_decision = await _status_decision(sports)
    await _schedule(post_decision, label="post-update")
    logger.info(
        "update_status_next_scheduled",
        next_time=post_decision.next_execution.isoformat()
        if post_decision.next_execution
        else None,
    )


async def _update_event_statuses():
    """
    Core status update logic.

    Adapted from scheduler/jobs.py:update_event_status_job()
    """
    async with async_session_maker() as session:
        writer = OddsWriter(session)
        reader = OddsReader(session)

        now = datetime.now(UTC)

        # Find events that should be live (within last 4 hours)
        start_range = now - timedelta(hours=4)
        end_range = now

        events = await reader.get_events_by_date_range(
            start_date=start_range,
            end_date=end_range,
            status=EventStatus.SCHEDULED,
        )

        updated_count = 0
        for event in events:
            # If commence time has passed, mark as live
            if event.commence_time <= now:
                await writer.update_event_status(
                    event_id=event.id,
                    status=EventStatus.LIVE,
                )
                updated_count += 1

        await session.commit()

        logger.info(
            "event_statuses_updated",
            updated_to_live=updated_count,
        )


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
