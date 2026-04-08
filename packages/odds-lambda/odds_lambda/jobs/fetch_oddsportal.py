"""Fetch upcoming odds from OddsPortal via OddsHarvester.

Scrapes pre-match odds for configured leagues, converts to pipeline format,
and stores via OddsWriter. Self-schedules next execution based on scrape
outcome and game proximity.

This job:
1. Scrapes upcoming matches from OddsPortal (via OddsHarvester)
2. Converts fractional odds to pipeline raw_data format
3. Matches or creates Event records
4. Stores snapshots via OddsWriter.store_odds_snapshot()
5. Self-schedules next execution via scheduler backend
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.oddsportal_adapter import (
    MatchOdds,
    convert_upcoming_matches,
)
from odds_lambda.oddsportal_common import hours_to_tier, run_scraper_with_retry, team_abbrev
from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.scheduling.jobs import make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Self-scheduling constants
NORMAL_INTERVAL_HOURS = 1.0
RETRY_DELAY_MINUTES_MIN = 10
RETRY_DELAY_MINUTES_MAX = 15
MAX_FAST_RETRIES = 2
OVERNIGHT_RESUME_HOUR_UTC = 6
OVERNIGHT_START_HOUR_UTC = 22
GAME_LOOKAHEAD_HOURS = 8


@dataclass
class LeagueSpec:
    """Configuration for a single league to scrape."""

    sport: str  # OddsHarvester sport name (e.g. "football")
    league: str  # OddsHarvester league name (e.g. "england-premier-league")
    sport_key: str  # Pipeline sport key (e.g. "soccer_epl")
    sport_title: str  # Display name (e.g. "EPL")
    markets: list[str] = field(default_factory=lambda: ["1x2"])


LEAGUE_SPECS: list[LeagueSpec] = [
    LeagueSpec(
        sport="football",
        league="england-premier-league",
        sport_key="soccer_epl",
        sport_title="EPL",
        markets=["1x2", "over_under_2_5"],
    ),
]

# Index by sport_key for fast lookup from event payload
_LEAGUE_SPEC_BY_SPORT: dict[str, LeagueSpec] = {spec.sport_key: spec for spec in LEAGUE_SPECS}


@dataclass
class IngestionStats:
    """Tracks results of a single league ingestion run."""

    league: str = ""
    matches_scraped: int = 0
    matches_converted: int = 0
    events_matched: int = 0
    events_created: int = 0
    snapshots_stored: int = 0
    errors: list[str] = field(default_factory=list)


def _build_event_id(home_team: str, away_team: str, match_date: datetime) -> str:
    """Generate deterministic event ID for OddsPortal-sourced events."""
    home_abbrev = team_abbrev(home_team)
    away_abbrev = team_abbrev(away_team)
    date_str = match_date.strftime("%Y-%m-%d")
    return f"op_live_{home_abbrev}_{away_abbrev}_{date_str}"


async def run_harvester_upcoming(spec: LeagueSpec) -> list[dict[str, Any]]:
    """Scrape upcoming matches for a league via ``run_scraper_with_retry``."""
    from oddsharvester.utils.command_enum import CommandEnum

    result = await run_scraper_with_retry(
        command=CommandEnum.UPCOMING_MATCHES,
        sport=spec.sport,
        leagues=[spec.league],
        markets=spec.markets,
        headless=True,
    )
    return result.success


async def find_or_create_event(
    session: AsyncSession,
    match: MatchOdds,
    spec: LeagueSpec,
) -> tuple[str, bool]:
    """Find existing event or create a new one.

    Returns:
        (event_id, was_created)
    """
    window_start = match.match_date - timedelta(hours=24)
    window_end = match.match_date + timedelta(hours=24)

    query = select(Event.id).where(
        and_(
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            Event.home_team == match.home_team,
            Event.away_team == match.away_team,
        )
    )
    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        return candidates[0], False

    if len(candidates) > 1:
        logger.warning(
            "ambiguous_event_match",
            home=match.home_team,
            away=match.away_team,
            date=match.match_date.isoformat(),
            count=len(candidates),
        )
        return candidates[0], False

    # Create new event
    event_id = _build_event_id(match.home_team, match.away_team, match.match_date)

    # Check idempotency
    existing = await session.get(Event, event_id)
    if existing:
        return event_id, False

    event = Event(
        id=event_id,
        sport_key=spec.sport_key,
        sport_title=spec.sport_title,
        commence_time=match.match_date,
        home_team=match.home_team,
        away_team=match.away_team,
        status=EventStatus.SCHEDULED,
    )
    session.add(event)
    return event_id, True


async def ingest_league(
    spec: LeagueSpec,
    *,
    raw_matches: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> IngestionStats:
    """Scrape and ingest a single league.

    Args:
        spec: League configuration.
        raw_matches: Pre-scraped data (skips harvester call). For testing.
        dry_run: If True, convert but don't write to DB.

    Returns:
        Ingestion statistics.
    """
    stats = IngestionStats(league=spec.league)

    if raw_matches is None:
        logger.info("scraping_league", league=spec.league, sport=spec.sport)
        raw_matches = await run_harvester_upcoming(spec)

    stats.matches_scraped = len(raw_matches)

    if not raw_matches:
        logger.warning("no_matches_scraped", league=spec.league)
        return stats

    logger.info("matches_scraped", league=spec.league, count=len(raw_matches))

    # Convert all markets
    all_converted: list[tuple[str, MatchOdds]] = []
    for market in spec.markets:
        converted = convert_upcoming_matches(raw_matches, market)
        all_converted.extend((market, m) for m in converted)
        stats.matches_converted += len(converted)

    if dry_run:
        logger.info(
            "dry_run_complete",
            league=spec.league,
            matches_scraped=stats.matches_scraped,
            matches_converted=stats.matches_converted,
        )
        return stats

    # Ingest to database
    scraped_date = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    api_request_id = f"oddsportal_live_{scraped_date}"

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for market, match in all_converted:
            try:
                event_id, created = await find_or_create_event(session, match, spec)

                if created:
                    stats.events_created += 1
                else:
                    stats.events_matched += 1

                snapshot, _ = await writer.store_odds_snapshot(
                    event_id=event_id,
                    raw_data=match.raw_data,
                    snapshot_time=match.scraped_date,
                )

                snapshot.api_request_id = api_request_id
                hours_before = (match.match_date - match.scraped_date).total_seconds() / 3600
                snapshot.hours_until_commence = max(0.0, hours_before)
                snapshot.fetch_tier = hours_to_tier(hours_before)
                stats.snapshots_stored += 1

            except Exception as e:
                msg = f"{match.home_team} vs {match.away_team} ({market}): {e}"
                stats.errors.append(msg)
                logger.error("match_ingestion_failed", error=msg, exc_info=True)

        await session.commit()

    logger.info(
        "league_ingestion_complete",
        league=spec.league,
        matches_scraped=stats.matches_scraped,
        events_matched=stats.events_matched,
        events_created=stats.events_created,
        snapshots_stored=stats.snapshots_stored,
        errors=len(stats.errors),
    )

    return stats


async def _get_next_game_time(sport_key: str = LEAGUE_SPECS[0].sport_key) -> datetime | None:
    """Find the commence_time of the nearest upcoming scheduled game."""
    from odds_lambda.storage.readers import OddsReader

    async with async_session_maker() as session:
        reader = OddsReader(session)
        now = datetime.now(UTC)
        events = await reader.get_events_by_date_range(
            start_date=now,
            end_date=now + timedelta(days=14),
            sport_key=sport_key,
            status=EventStatus.SCHEDULED,
        )

    if not events:
        return None
    return min(e.commence_time for e in events)


def _calculate_next_execution(
    *,
    success: bool,
    retry_count: int,
    now: datetime | None = None,
) -> tuple[datetime, int]:
    """Determine next execution time and updated retry_count.

    Returns:
        (next_time, next_retry_count)
    """
    if now is None:
        now = datetime.now(UTC)

    if success:
        return now + timedelta(hours=NORMAL_INTERVAL_HOURS), 0

    # Fast failure path
    if retry_count < MAX_FAST_RETRIES:
        jitter_minutes = random.randint(RETRY_DELAY_MINUTES_MIN, RETRY_DELAY_MINUTES_MAX)
        return now + timedelta(minutes=jitter_minutes), retry_count + 1

    # Exhausted retries — back to normal cadence
    return now + timedelta(hours=NORMAL_INTERVAL_HOURS), 0


def _apply_overnight_skip(next_time: datetime, next_game_time: datetime | None) -> datetime:
    """Push next_time to morning if overnight and no imminent games."""
    hours_to_game = float("inf")
    if next_game_time is not None:
        hours_to_game = (next_game_time - next_time).total_seconds() / 3600

    is_overnight = (
        next_time.hour >= OVERNIGHT_START_HOUR_UTC or next_time.hour < OVERNIGHT_RESUME_HOUR_UTC
    )
    no_imminent_games = hours_to_game > GAME_LOOKAHEAD_HOURS

    if is_overnight and no_imminent_games:
        # Skip to 06:00 UTC the next morning (or same morning if before 06:00)
        resume = next_time.replace(
            hour=OVERNIGHT_RESUME_HOUR_UTC, minute=0, second=0, microsecond=0
        )
        if resume <= next_time:
            resume += timedelta(days=1)
        return resume

    return next_time


async def _self_schedule(
    *,
    job_name: str,
    next_time: datetime,
    next_retry_count: int,
    dry_run: bool,
    sport: str | None = None,
) -> None:
    """Schedule the next execution via the scheduler backend."""
    backend = get_scheduler_backend(dry_run=dry_run)

    payload: dict[str, object] = {}
    if next_retry_count > 0:
        payload["retry_count"] = next_retry_count
    if sport:
        payload["sport"] = sport

    await backend.schedule_next_execution(
        job_name=job_name,
        next_time=next_time,
        payload=payload or None,
    )

    logger.info(
        "fetch_oddsportal_next_scheduled",
        next_time=next_time.isoformat(),
        retry_count=next_retry_count,
        backend=backend.get_backend_name(),
    )


async def main(
    *,
    sport: str | None = None,
    retry_count: int = 0,
    **_kwargs: object,
) -> None:
    """Main job execution — scrapes configured league(s), then self-schedules.

    Args:
        sport: Sport key (e.g. "soccer_epl"). When provided, only the matching
            LeagueSpec is scraped. Falls back to all LEAGUE_SPECS.
        retry_count: Current retry attempt (passed via self-scheduling payload).
    """
    from odds_cli.alerts.base import alert_manager, job_alert_context
    from odds_core.config import get_settings

    settings = get_settings()

    # Resolve which leagues to scrape
    if sport:
        spec = _LEAGUE_SPEC_BY_SPORT.get(sport)
        if spec is None:
            logger.error("unknown_sport_key", sport=sport, available=list(_LEAGUE_SPEC_BY_SPORT))
            return
        specs = [spec]
    else:
        specs = LEAGUE_SPECS

    async with job_alert_context(make_compound_job_name("fetch-oddsportal", sport)):
        logger.info(
            "fetch_oddsportal_started",
            sport=sport,
            leagues=len(specs),
            retry_count=retry_count,
        )

        all_stats: list[IngestionStats] = []
        leagues_failed = 0

        for spec in specs:
            try:
                stats = await ingest_league(spec)
                all_stats.append(stats)
            except Exception as e:
                leagues_failed += 1
                logger.error(
                    "league_failed",
                    league=spec.league,
                    error=str(e),
                    exc_info=True,
                )

        total_scraped = sum(s.matches_scraped for s in all_stats)
        total_snapshots = sum(s.snapshots_stored for s in all_stats)
        total_errors = sum(len(s.errors) for s in all_stats)

        logger.info(
            "fetch_oddsportal_completed",
            leagues_succeeded=len(all_stats),
            leagues_failed=leagues_failed,
            total_matches_scraped=total_scraped,
            total_snapshots_stored=total_snapshots,
            total_errors=total_errors,
        )

        if total_scraped == 0:
            from odds_cli.alerts.base import _check_rate_limit, _record_to_alert_history

            alert_type = f"scrape_empty:{make_compound_job_name('fetch-oddsportal', sport)}"
            if alert_manager.enabled and await _check_rate_limit(alert_type):
                detail = (
                    f"{leagues_failed}/{len(specs)} leagues failed"
                    if leagues_failed
                    else "0 matches returned"
                )
                msg = f"⚠️ OddsPortal scrape empty ({detail}), retry #{retry_count}"
                await alert_manager.alert(msg, "warning")
                await _record_to_alert_history(alert_type, "warning", msg)

    # Self-schedule next execution (outside alert context)
    scrape_success = total_scraped > 0
    now = datetime.now(UTC)
    next_time, next_retry_count = _calculate_next_execution(
        success=scrape_success,
        retry_count=retry_count,
        now=now,
    )

    # Apply overnight / no-games-soon skip
    sport_key_for_lookup = sport or LEAGUE_SPECS[0].sport_key
    next_game_time = await _get_next_game_time(sport_key=sport_key_for_lookup)
    next_time = _apply_overnight_skip(next_time, next_game_time)

    try:
        await _self_schedule(
            job_name=make_compound_job_name("fetch-oddsportal", sport),
            next_time=next_time,
            next_retry_count=next_retry_count,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
        )
    except Exception as e:
        logger.error("fetch_oddsportal_scheduling_failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
