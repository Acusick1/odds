"""Fetch ESPN fixtures job — refresh current-season all-competition schedules.

Flow:
1. Fetch current-season fixtures across EPL + FA Cup + League Cup + European
   competitions from the ESPN Site API (free, unauthenticated).
2. Upsert into ``espn_fixtures`` via ``EspnFixtureWriter`` (idempotent on
   ``(date, team, competition)``).
3. Marks events of postponed Premier League matches POSTPONED.
4. Alerts on failure via ``job_alert_context``; the run fails when the Premier
   League upcoming request failed (or every one did), the Premier League
   schedule request failed for every team, or ESPN lists no upcoming Premier League fixture while EPL
   events are scheduled. Rows fetched before a failure are upserted first.

Scheduler cadence is a fixed daily cron (06:00 UTC, pre-matchday) in
``_JOB_CRON_MAP`` — the ESPN endpoint is free and unauthenticated so no
proximity-based gating is needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.epl_data_models import EspnFixtureRecord
from odds_core.models import Event, EventStatus

from odds_lambda.collection_calendar import EmptyCollectionError, event_calendar
from odds_lambda.espn_fixture_fetcher import (
    LEAGUE_SLUGS,
    PRIMARY_LEAGUE_SLUG,
    EspnFixtureFetcher,
    EspnSeasonFetchError,
    EspnUpcomingFetchError,
    current_season,
)
from odds_lambda.scheduling.jobs import JobContext, require_sport
from odds_lambda.storage.espn_fixture_reader import EspnFixtureReader
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter
from odds_lambda.storage.event_merge import EVENT_MATCH_WINDOW, find_match_id
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Sport this job serves. ESPN fixture coverage here is EPL-only.
SPORT_KEY = "soccer_epl"


async def main(ctx: JobContext) -> None:
    """Fetch ESPN fixtures for the current season.

    Raises:
        ValueError: If ``ctx.sport`` is unset or does not match the sport
            this job serves.
    """
    from odds_core.alerts import job_alert_context

    sport = require_sport(ctx, "fetch-espn-fixtures")
    if sport != SPORT_KEY:
        raise ValueError(f"fetch-espn-fixtures only supports {SPORT_KEY}, got sport={sport!r}")

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
        postponed = await _mark_postponed_events()
        logger.info("fetch_espn_fixtures_postponed_marked", events=postponed)


# How many days ahead the upcoming fetch reaches. 30 days covers the agent's
# rotation-risk window (next ~5 PL rounds plus any midweek cup fixtures).
_UPCOMING_DAYS_AHEAD = 30


async def _raise_if_upcoming_unexpectedly_empty() -> None:
    """Raise ``EmptyCollectionError`` if EPL events are scheduled in the window ESPN returned empty."""
    horizon = timedelta(days=_UPCOMING_DAYS_AHEAD)
    async with async_session_maker() as session:
        calendar = await event_calendar(session, SPORT_KEY, now=datetime.now(UTC), horizon=horizon)
    if calendar.kickoffs_in_horizon:
        raise EmptyCollectionError(
            f"ESPN returned no upcoming Premier League fixtures with "
            f"{calendar.kickoffs_in_horizon} {SPORT_KEY} event(s) scheduled within {horizon} "
            f"(next kickoff {calendar.next_kickoff})"
        )


async def _mark_postponed_events() -> int:
    """Set the event of each ESPN-postponed Premier League match POSTPONED; returns how many."""
    marked = 0
    async with async_session_maker() as session:
        postponed = await EspnFixtureReader(session).get_postponed_home_fixtures(
            LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG]
        )
        writer = OddsWriter(session)
        reader = EspnFixtureReader(session)
        for fixture in postponed:
            if await reader.has_unpostponed_fixture(
                LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG],
                fixture.team,
                fixture.opponent,
                fixture.date,
                EVENT_MATCH_WINDOW,
            ):
                continue
            event_id = await find_match_id(
                session, fixture.team, fixture.opponent, fixture.date, SPORT_KEY
            )
            event = await session.get(Event, event_id) if event_id is not None else None
            if event is None or event.status not in (EventStatus.SCHEDULED, EventStatus.LIVE):
                continue
            await writer.update_event_status(event.id, EventStatus.POSTPONED)
            marked += 1
        await session.commit()
    return marked


async def _upsert(records: list[EspnFixtureRecord]) -> int:
    if not records:
        return 0
    async with async_session_maker() as session:
        total = await EspnFixtureWriter(session).upsert_fixtures(records)
        await session.commit()
    return total


async def _fetch_and_upsert(season: int) -> tuple[int, int, int]:
    """Fetch past + upcoming fixtures from ESPN and upsert into the database.

    Fetched rows are upserted before any failure is raised. Returns
    ``(past_count, upcoming_count, total_upserted)``.
    """
    async with EspnFixtureFetcher() as fetcher:
        try:
            past_records = await fetcher.fetch_season(season)
            season_error: EspnSeasonFetchError | None = None
        except EspnSeasonFetchError as e:
            past_records, season_error = [], e
        try:
            upcoming = await fetcher.fetch_upcoming(days_ahead=_UPCOMING_DAYS_AHEAD)
        except EspnUpcomingFetchError:
            await _upsert(past_records)
            raise

    if season_error is not None:
        await _upsert(upcoming.records)
        raise season_error

    if upcoming.failed_leagues:
        await send_job_warning(
            "espn_upcoming_partial:fetch-espn-fixtures",
            f"fetch-espn-fixtures: upcoming request failed for {', '.join(upcoming.failed_leagues)}",
        )

    # Dedup across past + upcoming on (date, team, opponent). The team-schedule
    # endpoint (past) and the scoreboard endpoint (upcoming) can overlap for an
    # in-progress-today match. Iterate scoreboard first so it wins any conflict:
    # team-schedule can lag for matches that just kicked off, while the scoreboard
    # has the freshest ``state``, which is the reason this job fetches both.
    seen: set[tuple[str, str, str]] = set()
    combined: list[EspnFixtureRecord] = []
    for record in (*upcoming.records, *past_records):
        key = (record.date.isoformat(), record.team, record.opponent)
        if key in seen:
            continue
        seen.add(key)
        combined.append(record)

    if not combined:
        logger.warning("fetch_espn_fixtures_empty", season=season)
    total = await _upsert(combined)

    if not any(r.competition == LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG] for r in upcoming.records):
        await _raise_if_upcoming_unexpectedly_empty()

    return len(past_records), len(upcoming.records), total


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
