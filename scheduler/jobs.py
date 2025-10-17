"""Scheduled job definitions for data collection."""

from datetime import datetime, timedelta

import structlog

from core.config import settings
from core.data_fetcher import TheOddsAPIClient
from core.database import async_session_maker
from core.models import EventStatus, FetchLog
from storage.readers import OddsReader
from storage.writers import OddsWriter

logger = structlog.get_logger()


async def fetch_odds_job():
    """
    Primary job: Fetch current odds for all upcoming games.

    Frequency: Based on sampling_mode configuration
    - Fixed: Every N minutes (default 30)
    - Adaptive: Varies by proximity to game time
    """
    logger.info("fetch_odds_job_started")

    try:
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
                for event, event_data in zip(
                    response.events, response.raw_events_data, strict=True
                ):
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
                    "fetch_odds_job_completed",
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
                        percentage=round(
                            response.quota_remaining / settings.odds_api_quota * 100, 1
                        ),
                    )

    except Exception as e:
        logger.error("fetch_odds_job_failed", error=str(e), exc_info=True)

        # Log failed fetch
        try:
            async with async_session_maker() as fail_session:
                fail_writer = OddsWriter(fail_session)
                fetch_log = FetchLog(
                    sport_key=settings.sports[0] if settings.sports else "unknown",
                    events_count=0,
                    bookmakers_count=0,
                    success=False,
                    error_message=str(e),
                )
                await fail_writer.log_fetch(fetch_log)
                await fail_session.commit()
        except Exception:
            pass

        raise


async def fetch_scores_job():
    """
    Secondary job: Fetch scores for completed games.

    Frequency: Every 6 hours
    Updates event results and status to FINAL
    """
    logger.info("fetch_scores_job_started")

    async with async_session_maker() as session:
        try:
            writer = OddsWriter(session)

            for sport_key in settings.sports:
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
                                scores = score_data.get("scores", [])

                                # Extract home and away scores
                                home_score = None
                                away_score = None

                                for score in scores:
                                    if score.get("name") == score_data.get("home_team"):
                                        home_score = score.get("score")
                                    if score.get("name") == score_data.get("away_team"):
                                        away_score = score.get("score")

                                if home_score is not None and away_score is not None:
                                    await writer.update_event_status(
                                        event_id=event_id,
                                        status=EventStatus.FINAL,
                                        home_score=int(home_score),
                                        away_score=int(away_score),
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
                        "fetch_scores_job_completed",
                        sport=sport_key,
                        updated=updated_count,
                    )

        except Exception as e:
            logger.error("fetch_scores_job_failed", error=str(e), exc_info=True)
            raise


async def update_event_status_job():
    """
    Tertiary job: Update event statuses based on commence time.

    Frequency: Every hour
    - SCHEDULED → LIVE (when game starts)
    - LIVE → Check if completed
    """
    logger.info("update_event_status_job_started")

    async with async_session_maker() as session:
        try:
            writer = OddsWriter(session)
            reader = OddsReader(session)

            now = datetime.utcnow()

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
                "update_event_status_job_completed",
                updated_to_live=updated_count,
            )

        except Exception as e:
            logger.error("update_event_status_job_failed", error=str(e), exc_info=True)
            raise


async def cleanup_old_data_job():
    """
    Maintenance job: Archive or delete old data.

    Frequency: Daily
    Note: Not enabled by default, implement when needed
    """
    logger.info("cleanup_old_data_job_started")

    # Future implementation:
    # - Archive odds snapshots older than 90 days
    # - Clean up completed events older than 1 year
    # - Rotate logs

    logger.info("cleanup_old_data_job_completed (no-op)")
