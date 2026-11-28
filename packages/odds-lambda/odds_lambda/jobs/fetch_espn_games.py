"""Fetch ESPN games job — refresh current-season all-competition schedules.

For NBA the job instead syncs ESPN's scoreboard into events (``odds_lambda.espn_nba_schedule``),
and for each league of ``odds_lambda.espn_football_schedule.LEAGUES`` (every football league
but the EPL) it does the same with that league's football scoreboard.

Flow:
1. Fetch current-season games across EPL + FA Cup + League Cup + European
   competitions from the ESPN Site API (free, unauthenticated).
2. Upsert into ``espn_games`` via ``EspnGameWriter`` (idempotent on
   ``(date, team, competition)``).
3. Marks events of postponed Premier League matches POSTPONED.
4. Alerts on failure via ``job_alert_context``; the run fails when the Premier
   League upcoming request failed (or every one did), the Premier League
   schedule request failed for every team, or ESPN lists no upcoming Premier League game while EPL
   events are scheduled. Rows fetched before a failure are upserted first.

Scheduler cadence is a fixed daily cron (06:00 UTC, pre-matchday) in
``JobSpec.cron`` — the ESPN endpoint is free and unauthenticated so no
proximity-based gating is needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.epl_data_models import EspnGameRecord
from odds_core.job_specs import heartbeat_name
from odds_core.models import Event, EventStatus
from odds_core.sports import SportKey

from odds_lambda.collection_calendar import EmptyCollectionError, event_calendar
from odds_lambda.espn_football_schedule import LEAGUES, FootballLeague
from odds_lambda.espn_game_fetcher import (
    LEAGUE_SLUGS,
    PRIMARY_LEAGUE_SLUG,
    EspnGameFetcher,
    EspnSeasonFetchError,
    EspnUpcomingFetchError,
    current_season,
)
from odds_lambda.live_events import alert_dropped_games
from odds_lambda.scheduling.jobs import JobContext, require_sport
from odds_lambda.storage.espn_game_reader import EspnGameReader
from odds_lambda.storage.espn_game_writer import EspnGameWriter
from odds_lambda.storage.event_merge import EVENT_MATCH_WINDOW, find_match_id
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# EPL keeps its games in ``espn_games``; NBA's schedule is written as events directly.
SPORT_KEY = "soccer_epl"
NBA_SPORT_KEY: SportKey = "basketball_nba"


async def main(ctx: JobContext) -> None:
    """Fetch ESPN games for the current season.

    Raises:
        ValueError: If ``ctx.sport`` is unset or does not match the sport
            this job serves.
    """
    from odds_core.alerts import job_alert_context

    sport = require_sport(ctx, "fetch-espn-games")
    if sport == NBA_SPORT_KEY:
        await _sync_nba_schedule()
        return
    if sport in LEAGUES:
        await _sync_football_schedule(LEAGUES[sport])
        return
    if sport != SPORT_KEY:
        raise ValueError(
            f"fetch-espn-games supports {SPORT_KEY}, {NBA_SPORT_KEY} and "
            f"{', '.join(LEAGUES)}, got sport={sport!r}"
        )

    app_settings = get_settings()
    season = current_season()

    logger.info(
        "fetch_espn_games_started",
        backend=app_settings.scheduler.backend,
        season=season,
        sport=SPORT_KEY,
    )

    async with job_alert_context(heartbeat_name("fetch-espn-games", sport)):
        past_count, upcoming_count, total = await _fetch_and_upsert(season)
        logger.info(
            "fetch_espn_games_completed",
            season=season,
            past_count=past_count,
            upcoming_count=upcoming_count,
            total_upserted=total,
        )
        postponed = await _mark_postponed_events()
        logger.info("fetch_espn_games_postponed_marked", events=postponed)


async def _warn_if_listing_unexpectedly_empty(sport_key: SportKey, days: int) -> None:
    """Warn when ESPN lists no game of the sport although events are scheduled in the next
    ``days``; the off-season is silent."""
    async with async_session_maker() as session:
        calendar = await event_calendar(
            session, sport_key, now=datetime.now(UTC), horizon=timedelta(days=days)
        )
    if calendar.kickoffs_in_horizon:
        job = heartbeat_name("fetch-espn-games", sport_key)
        await send_job_warning(
            f"espn_empty:{job}",
            f"{job}: ESPN listed no {sport_key} game with "
            f"{calendar.kickoffs_in_horizon} event(s) scheduled in the next {days} days",
        )


async def _sync_nba_schedule() -> None:
    """Create the NBA events of the next weeks and settle the finished ones; raises when any
    game failed, after the others are stored."""
    from odds_core.alerts import job_alert_context

    from odds_lambda.espn_nba_schedule import UPCOMING_DAYS, sync_upcoming
    from odds_lambda.scheduling.jobs import make_compound_job_name

    async with job_alert_context(make_compound_job_name("fetch-espn-games", NBA_SPORT_KEY)):
        report = await sync_upcoming(datetime.now(UTC))
        logger.info(
            "fetch_espn_nba_schedule_completed",
            games=report.games,
            created=report.created,
            matched=report.matched,
            finalised=report.finalised,
            rescheduled=report.rescheduled,
            errors=len(report.errors),
        )
        if report.errors:
            raise RuntimeError(
                f"{len(report.errors)} NBA games failed to sync: " + "; ".join(report.errors[:3])
            )
        if report.games == 0:
            await _warn_if_listing_unexpectedly_empty(NBA_SPORT_KEY, UPCOMING_DAYS)


async def _sync_football_schedule(league: FootballLeague) -> None:
    """Create the events of the league's next matches; raises when any match failed, after the
    others are stored, and alerts on matches dropped for a team outside the league's clubs."""
    from odds_core.alerts import job_alert_context

    from odds_lambda.espn_football_schedule import UPCOMING_DAYS, sync_upcoming
    from odds_lambda.scheduling.jobs import make_compound_job_name

    job_name = make_compound_job_name("fetch-espn-games", league.sport_key)
    async with job_alert_context(job_name):
        report = await sync_upcoming(league, datetime.now(UTC))
        logger.info(
            "fetch_espn_football_schedule_completed",
            sport=league.sport_key,
            games=report.games,
            created=report.created,
            matched=report.matched,
            skipped=report.skipped,
            rescheduled=report.rescheduled,
            dropped=len(report.dropped),
            errors=len(report.errors),
        )
        await alert_dropped_games(job_name, report.dropped, stored="event")
        if report.errors:
            raise RuntimeError(
                f"{len(report.errors)} {league.sport_title} matches failed to sync: "
                + "; ".join(report.errors[:3])
            )
        if report.games == 0 and not report.dropped:
            await _warn_if_listing_unexpectedly_empty(league.sport_key, UPCOMING_DAYS)


# How many days ahead the upcoming fetch reaches. 30 days covers the agent's
# rotation-risk window (next ~5 PL rounds plus any midweek cup games).
_UPCOMING_DAYS_AHEAD = 30


async def _raise_if_upcoming_unexpectedly_empty() -> None:
    """Raise ``EmptyCollectionError`` if EPL events are scheduled in the window ESPN returned empty."""
    horizon = timedelta(days=_UPCOMING_DAYS_AHEAD)
    async with async_session_maker() as session:
        calendar = await event_calendar(session, SPORT_KEY, now=datetime.now(UTC), horizon=horizon)
    if calendar.kickoffs_in_horizon:
        raise EmptyCollectionError(
            f"ESPN returned no upcoming Premier League games with "
            f"{calendar.kickoffs_in_horizon} {SPORT_KEY} event(s) scheduled within {horizon} "
            f"(next kickoff {calendar.next_kickoff})"
        )


async def _mark_postponed_events() -> int:
    """Set the event of each ESPN-postponed Premier League match POSTPONED; returns how many."""
    marked = 0
    async with async_session_maker() as session:
        postponed = await EspnGameReader(session).get_postponed_home_games(
            LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG]
        )
        writer = OddsWriter(session)
        reader = EspnGameReader(session)
        for game in postponed:
            if await reader.has_unpostponed_game(
                LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG],
                game.team,
                game.opponent,
                game.date,
                EVENT_MATCH_WINDOW,
            ):
                continue
            event_id = await find_match_id(session, game.team, game.opponent, game.date, SPORT_KEY)
            event = await session.get(Event, event_id) if event_id is not None else None
            if event is None or event.status not in (EventStatus.SCHEDULED, EventStatus.LIVE):
                continue
            await writer.update_event_status(event.id, EventStatus.POSTPONED)
            marked += 1
        await session.commit()
    return marked


async def _upsert(records: list[EspnGameRecord]) -> int:
    if not records:
        return 0
    async with async_session_maker() as session:
        total = await EspnGameWriter(session).upsert_games(records)
        await session.commit()
    return total


async def _fetch_and_upsert(season: int) -> tuple[int, int, int]:
    """Fetch past + upcoming games from ESPN and upsert into the database.

    Fetched rows are upserted before any failure is raised. Returns
    ``(past_count, upcoming_count, total_upserted)``.
    """
    async with EspnGameFetcher() as fetcher:
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
            "espn_upcoming_partial:fetch-espn-games",
            f"fetch-espn-games: upcoming request failed for {', '.join(upcoming.failed_leagues)}",
        )

    # Dedup across past + upcoming on (date, team, opponent). The team-schedule
    # endpoint (past) and the scoreboard endpoint (upcoming) can overlap for an
    # in-progress-today match. Iterate scoreboard first so it wins any conflict:
    # team-schedule can lag for matches that just kicked off, while the scoreboard
    # has the freshest ``state``, which is the reason this job fetches both.
    seen: set[tuple[str, str, str]] = set()
    combined: list[EspnGameRecord] = []
    for record in (*upcoming.records, *past_records):
        key = (record.date.isoformat(), record.team, record.opponent)
        if key in seen:
            continue
        seen.add(key)
        combined.append(record)

    if not combined:
        logger.warning("fetch_espn_games_empty", season=season)
    total = await _upsert(combined)

    if not any(r.competition == LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG] for r in upcoming.records):
        await _raise_if_upcoming_unexpectedly_empty()

    return len(past_records), len(upcoming.records), total


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
