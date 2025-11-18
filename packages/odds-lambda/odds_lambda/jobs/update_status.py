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

from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.scheduling.intelligence import SchedulingIntelligence
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()


async def main():
    """
    Main job execution flow.

    Flow:
    1. Check if we should execute (smart gating)
    2. If yes: update event statuses
    3. Calculate next execution time
    4. Schedule next run via backend
    """
    app_settings = get_settings()

    logger.info("update_status_job_started", backend=app_settings.scheduler.backend)

    # Smart execution gating
    intelligence = SchedulingIntelligence(lookahead_days=app_settings.scheduler.lookahead_days)
    decision = await intelligence.should_execute_status_update()

    if not decision.should_execute:
        logger.info(
            "update_status_skipped",
            reason=decision.reason,
            next_check=decision.next_execution.isoformat() if decision.next_execution else None,
        )

        # Still schedule next check even if not executing
        if decision.next_execution:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
            await backend.schedule_next_execution(
                job_name="update-status", next_time=decision.next_execution
            )

        return

    # Execute update logic
    logger.info("update_status_executing", reason=decision.reason)

    try:
        await _update_event_statuses()
        logger.info("update_status_completed")

    except Exception as e:
        logger.error("update_status_failed", error=str(e), exc_info=True)

        # Send critical alert
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            await send_critical(
                f"🚨 Update status job failed: {type(e).__name__}: {str(e)}"
            )

        raise

    # Self-schedule next execution
    if decision.next_execution:
        try:
            backend = get_scheduler_backend(dry_run=app_settings.scheduler.dry_run)
            await backend.schedule_next_execution(
                job_name="update-status", next_time=decision.next_execution
            )
            logger.info(
                "update_status_next_scheduled",
                next_time=decision.next_execution.isoformat(),
                backend=backend.get_backend_name(),
            )
        except Exception as e:
            logger.error("update_status_scheduling_failed", error=str(e), exc_info=True)

            # Send error alert
            if app_settings.alerts.alert_enabled:
                from odds_cli.alerts.base import send_error

                await send_error(
                    f"Update status scheduling failed: {type(e).__name__}: {str(e)}"
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
    asyncio.run(main())
