"""Fetch ESPN fixtures job — refresh current-season all-competition schedules.

Flow:
1. Fetch current-season fixtures across EPL + FA Cup + League Cup + European
   competitions from the ESPN Site API (free, unauthenticated).
2. Upsert into ``espn_fixtures`` via ``EspnFixtureWriter`` (idempotent on
   ``(date, team, competition)``).
3. Alerts on failure via ``job_alert_context``.

Scheduler cadence is a fixed daily cron (06:00 UTC, pre-matchday) managed in
Terraform — the ESPN endpoint is free and unauthenticated so no proximity-based
gating is needed.
"""

from __future__ import annotations

import asyncio

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker

from odds_lambda.espn_fixture_fetcher import EspnFixtureFetcher, current_season
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter

logger = structlog.get_logger()

# Sport this job serves. ESPN fixture coverage here is EPL-only.
SPORT_KEY = "soccer_epl"


async def main(ctx: JobContext) -> None:
    """Fetch ESPN fixtures for the current season.

    Raises:
        ValueError: If ``ctx.sport`` is provided and does not match the sport
            this job serves.
    """
    from odds_core.alerts import job_alert_context

    if ctx.sport is not None and ctx.sport != SPORT_KEY:
        raise ValueError(f"fetch-espn-fixtures only supports {SPORT_KEY}, got sport={ctx.sport!r}")

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
