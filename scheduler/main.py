"""APScheduler orchestration for periodic data collection."""

import asyncio
import sys

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.config import settings
from scheduler.jobs import fetch_odds_job, fetch_scores_job, update_event_status_job

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def create_scheduler() -> AsyncIOScheduler:
    """
    Create and configure the APScheduler instance.

    Returns:
        Configured AsyncIOScheduler
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Primary job: Fetch odds
    if settings.sampling_mode == "fixed":
        # Fixed interval sampling
        interval_minutes = settings.fixed_interval_minutes
        scheduler.add_job(
            fetch_odds_job,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="fetch_odds",
            name="Fetch odds data",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
        )
        logger.info(
            "scheduled_fetch_odds",
            mode="fixed",
            interval_minutes=interval_minutes,
        )
    else:
        # Adaptive sampling (future implementation)
        # For now, use medium interval
        interval_minutes = settings.adaptive_intervals.get("medium", 15)
        scheduler.add_job(
            fetch_odds_job,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="fetch_odds",
            name="Fetch odds data (adaptive)",
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "scheduled_fetch_odds",
            mode="adaptive",
            interval_minutes=interval_minutes,
        )

    # Secondary job: Fetch scores every 6 hours
    scheduler.add_job(
        fetch_scores_job,
        trigger=IntervalTrigger(hours=6),
        id="fetch_scores",
        name="Fetch game scores",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduled_fetch_scores", interval_hours=6)

    # Tertiary job: Update event statuses hourly
    scheduler.add_job(
        update_event_status_job,
        trigger=IntervalTrigger(hours=1),
        id="update_event_status",
        name="Update event statuses",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("scheduled_update_event_status", interval_hours=1)

    return scheduler


async def main():
    """Main entry point for scheduler."""
    logger.info(
        "scheduler_starting",
        sports=settings.sports,
        bookmakers=settings.bookmakers,
        markets=settings.markets,
        sampling_mode=settings.sampling_mode,
    )

    # Create scheduler
    scheduler = create_scheduler()

    try:
        # Start scheduler
        scheduler.start()
        logger.info("scheduler_started")

        # Run initial fetch immediately
        logger.info("running_initial_fetch")
        try:
            await fetch_odds_job()
        except Exception as e:
            logger.error("initial_fetch_failed", error=str(e))

        # Keep running
        while True:
            await asyncio.sleep(60)

    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler_shutdown_requested")
        scheduler.shutdown()
        logger.info("scheduler_stopped")
        sys.exit(0)

    except Exception as e:
        logger.error("scheduler_error", error=str(e), exc_info=True)
        scheduler.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
