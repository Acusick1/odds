"""Fetch ESPN fixtures job — refresh current-season all-competition schedules.

Flow:
1. Fetch current-season fixtures across EPL + FA Cup + League Cup + European
   competitions from the ESPN Site API (free, unauthenticated).
2. Upsert into ``espn_fixtures`` via ``EspnFixtureWriter`` (idempotent on
   ``(date, team, competition)``).
3. Alerts on failure via ``job_alert_context``; the run fails when every
   upcoming request failed, or ESPN lists no upcoming fixture while EPL events
   are scheduled.

Scheduler cadence is a fixed daily cron (06:00 UTC, pre-matchday) managed in
Terraform — the ESPN endpoint is free and unauthenticated so no proximity-based
gating is needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.epl_data_models import EspnFixtureRecord

from odds_lambda.collection_calendar import EmptyCollectionError, fixture_calendar
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
        past_count, upcoming_count, total = await _fetch_and_upsert(season)
        logger.info(
            "fetch_espn_fixtures_completed",
            season=season,
            past_count=past_count,
            upcoming_count=upcoming_count,
            total_upserted=total,
        )


# How many days ahead the upcoming fetch reaches. 30 days covers the agent's
# rotation-risk window (next ~5 PL rounds plus any midweek cup fixtures).
_UPCOMING_DAYS_AHEAD = 30


async def _raise_if_upcoming_unexpectedly_empty() -> None:
    """Raise ``EmptyCollectionError`` if EPL events are scheduled in the window ESPN returned empty."""
    horizon = timedelta(days=_UPCOMING_DAYS_AHEAD)
    async with async_session_maker() as session:
        calendar = await fixture_calendar(
            session, SPORT_KEY, now=datetime.now(UTC), horizon=horizon
        )
    if calendar.kickoffs_in_horizon:
        raise EmptyCollectionError(
            f"ESPN returned no upcoming fixtures with {calendar.kickoffs_in_horizon} "
            f"{SPORT_KEY} event(s) scheduled within {horizon} "
            f"(next kickoff {calendar.next_kickoff})"
        )


async def _fetch_and_upsert(season: int) -> tuple[int, int, int]:
    """Fetch past + upcoming fixtures from ESPN and upsert into the database.

    Returns ``(past_count, upcoming_count, total_upserted)``.
    """
    async with EspnFixtureFetcher() as fetcher:
        past_records = await fetcher.fetch_season(season)
        upcoming_records = await fetcher.fetch_upcoming(days_ahead=_UPCOMING_DAYS_AHEAD)

    if not upcoming_records:
        await _raise_if_upcoming_unexpectedly_empty()

    past_count = len(past_records)
    upcoming_count = len(upcoming_records)

    # Dedup across past + upcoming on (date, team, opponent). The team-schedule
    # endpoint (past) and the scoreboard endpoint (upcoming) can overlap for an
    # in-progress-today match. Iterate scoreboard first so it wins any conflict:
    # team-schedule can lag for matches that just kicked off, while the scoreboard
    # has the freshest ``state``, which is the reason this job fetches both.
    seen: set[tuple[str, str, str]] = set()
    combined: list[EspnFixtureRecord] = []
    for record in (*upcoming_records, *past_records):
        key = (record.date.isoformat(), record.team, record.opponent)
        if key in seen:
            continue
        seen.add(key)
        combined.append(record)

    if not combined:
        logger.warning("fetch_espn_fixtures_empty", season=season)
        return past_count, upcoming_count, 0

    async with async_session_maker() as session:
        writer = EspnFixtureWriter(session)
        total = await writer.upsert_fixtures(combined)
        await session.commit()

    return past_count, upcoming_count, total


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
