"""Fetch ESPN fixtures job — refresh current-season all-competition schedules.

Flow:
1. Fetch current-season fixtures across EPL + FA Cup + League Cup + European
   competitions from the ESPN Site API (free, unauthenticated).
2. Upsert into ``espn_fixtures`` via ``EspnFixtureWriter`` (idempotent on
   ``(date, team, competition)``).
3. Self-schedule next execution for +24h.
4. Alerts on failure via ``job_alert_context``.

Scheduler cadence is daily at 06:00 UTC (pre-matchday), with ESPN being a free
unauthenticated API there is no quota concern that would require smart gating.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker

from odds_lambda.espn_fixture_fetcher import EspnFixtureFetcher, current_season
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter

logger = structlog.get_logger()

# Sport this job serves. ESPN fixture coverage here is EPL-only.
SPORT_KEY = "soccer_epl"

# Default interval between runs. Fixture data moves slowly enough that daily is
# plenty even in heavy cup weeks.
DEFAULT_INTERVAL_HOURS = 24.0


async def main(ctx: JobContext) -> None:
    """Fetch ESPN fixtures for the current season and self-schedule."""
    from odds_core.alerts import job_alert_context

    app_settings = get_settings()
    season = current_season()

    logger.info(
        "fetch_espn_fixtures_started",
        backend=app_settings.scheduler.backend,
        season=season,
        sport=SPORT_KEY,
    )

    async with job_alert_context("fetch-espn-fixtures"):
        count = await _fetch_and_upsert_current_season(season)
        logger.info("fetch_espn_fixtures_completed", season=season, upserted=count)

    # Self-schedule next execution (24h from now). Uses the sport-suffixed job
    # name so the Lambda EventBridge rule matches Terraform.
    try:
        next_time = datetime.now(UTC) + timedelta(hours=DEFAULT_INTERVAL_HOURS)
        compound_name = make_compound_job_name("fetch-espn-fixtures", SPORT_KEY)
        await self_schedule(
            job_name=compound_name,
            next_time=next_time,
            dry_run=app_settings.scheduler.dry_run,
            sport=SPORT_KEY,
            interval_hours=DEFAULT_INTERVAL_HOURS,
            reason="fetch-espn-fixtures daily cadence",
        )
    except Exception as e:
        logger.error("fetch_espn_fixtures_scheduling_failed", error=str(e), exc_info=True)
        from odds_core.alerts import send_error

        await send_error(f"fetch-espn-fixtures scheduling failed: {type(e).__name__}: {e}")


async def _fetch_and_upsert_current_season(season: int) -> int:
    """Fetch the given season from ESPN and upsert into the database.

    Returns the number of records upserted.
    """
    async with EspnFixtureFetcher() as fetcher:
        records = await fetcher.fetch_season(season)

    if not records:
        logger.warning("fetch_espn_fixtures_empty", season=season)
        return 0

    async with async_session_maker() as session:
        writer = EspnFixtureWriter(session)
        count = await writer.upsert_fixtures(records)
        await session.commit()

    return count


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
