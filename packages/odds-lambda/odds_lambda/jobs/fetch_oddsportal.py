"""Fetch upcoming odds from OddsPortal via OddsHarvester.

Scrapes pre-match odds for configured leagues, converts to pipeline format,
and stores via OddsWriter. Self-schedules next execution based on scrape
outcome.

Self-scheduling strategy: defensively schedule at retry cadence before any
work (no DB needed), then reschedule at normal cadence on success. If
anything fails — DB, browser, Lambda timeout — the retry schedule is already
set and the chain survives.

This job:
1. Pre-schedules next run at retry cadence (no DB, survives any failure)
2. Scrapes upcoming matches from OddsPortal (via OddsHarvester)
3. Converts fractional odds to pipeline raw_data format
4. Matches or creates Event records
5. Stores snapshots via OddsWriter.store_odds_snapshot()
6. On success, reschedules at normal cadence (1h) with overnight skip
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.database import async_session_maker

from odds_lambda.oddsportal_adapter import (
    MatchOdds,
    convert_upcoming_matches,
)
from odds_lambda.oddsportal_common import hours_to_tier, run_scraper_with_retry
from odds_lambda.scheduling.backends import get_scheduler_backend
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Self-scheduling constants
NORMAL_INTERVAL_HOURS = 1.0
RETRY_DELAY_MINUTES_MIN = 10
RETRY_DELAY_MINUTES_MAX = 15
MAX_FAST_RETRIES = 2


@dataclass
class LeagueSpec:
    """Configuration for a single league to scrape."""

    sport: str  # OddsHarvester sport name (e.g. "football")
    league: str  # OddsHarvester league name (e.g. "england-premier-league")
    sport_key: str  # Pipeline sport key (e.g. "soccer_epl")
    sport_title: str  # Display name (e.g. "EPL")
    markets: list[str] = field(default_factory=lambda: ["1x2"])
    primary_market: str = "1x2"
    num_outcomes: int = 3
    overnight_start_utc: int = 22
    overnight_resume_utc: int = 6


LEAGUE_SPECS: list[LeagueSpec] = [
    LeagueSpec(
        sport="football",
        league="england-premier-league",
        sport_key="soccer_epl",
        sport_title="EPL",
        markets=["1x2", "over_under_2_5"],
    ),
    LeagueSpec(
        sport="baseball",
        league="mlb",
        sport_key="baseball_mlb",
        sport_title="MLB",
        markets=["home_away"],
        primary_market="home_away",
        num_outcomes=2,
        overnight_start_utc=5,
        overnight_resume_utc=14,
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
                event_id, created = await writer.find_or_create_event(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    match_date=match.match_date,
                    sport_key=spec.sport_key,
                    sport_title=spec.sport_title,
                )

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


def _apply_overnight_skip(
    next_time: datetime,
    *,
    overnight_start_utc: int = 22,
    overnight_resume_utc: int = 6,
) -> datetime:
    """Push next_time to resume hour if it falls in the overnight window."""
    if overnight_start_utc > overnight_resume_utc:
        # Window wraps midnight (e.g. 22:00-06:00)
        is_overnight = (
            next_time.hour >= overnight_start_utc or next_time.hour < overnight_resume_utc
        )
    else:
        # Window within same day (e.g. 05:00-14:00)
        is_overnight = overnight_start_utc <= next_time.hour < overnight_resume_utc

    if is_overnight:
        resume = next_time.replace(hour=overnight_resume_utc, minute=0, second=0, microsecond=0)
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


async def main(ctx: JobContext) -> None:
    """Main job execution — scrapes configured league(s), then self-schedules.

    Scheduling strategy: defensively pre-schedule at retry cadence (no DB
    needed) so the chain survives any failure including Lambda timeouts. On
    success, reschedule at normal cadence with overnight skip.
    """
    from odds_core.alerts import job_alert_context, send_job_warning
    from odds_core.config import get_settings

    settings = get_settings()
    sport = ctx.sport
    retry_count = ctx.retry_count

    # Resolve which leagues to scrape
    if sport:
        spec = _LEAGUE_SPEC_BY_SPORT.get(sport)
        if spec is None:
            logger.error("unknown_sport_key", sport=sport, available=list(_LEAGUE_SPEC_BY_SPORT))
            return
        specs = [spec]
    else:
        specs = LEAGUE_SPECS

    compound_job_name = make_compound_job_name("fetch-oddsportal", sport)
    primary_spec = specs[0]

    # Defensive pre-schedule at retry cadence BEFORE any DB or browser work.
    # No DB query needed — just now + retry delay. If anything fails (DB down,
    # browser crash, Lambda timeout), this schedule is already set.
    defensive_next_time, defensive_retry_count = _calculate_next_execution(
        success=False,
        retry_count=retry_count,
        now=datetime.now(UTC),
    )
    defensive_next_time = _apply_overnight_skip(
        defensive_next_time,
        overnight_start_utc=primary_spec.overnight_start_utc,
        overnight_resume_utc=primary_spec.overnight_resume_utc,
    )
    try:
        await _self_schedule(
            job_name=compound_job_name,
            next_time=defensive_next_time,
            next_retry_count=defensive_retry_count,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
        )
    except Exception as e:
        logger.error("fetch_oddsportal_scheduling_failed", error=str(e), exc_info=True)
        raise

    async with job_alert_context(compound_job_name):
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
            detail = (
                f"{leagues_failed}/{len(specs)} leagues failed"
                if leagues_failed
                else "0 matches returned"
            )
            await send_job_warning(
                alert_type=f"scrape_empty:{compound_job_name}",
                message=f"⚠️ OddsPortal scrape empty ({detail}), retry #{retry_count}",
            )

    # On success, push schedule out to normal cadence
    if total_scraped > 0:
        success_next_time = _apply_overnight_skip(
            datetime.now(UTC) + timedelta(hours=NORMAL_INTERVAL_HOURS),
            overnight_start_utc=primary_spec.overnight_start_utc,
            overnight_resume_utc=primary_spec.overnight_resume_utc,
        )
        try:
            await _self_schedule(
                job_name=compound_job_name,
                next_time=success_next_time,
                next_retry_count=0,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
            )
        except Exception as e:
            logger.error("fetch_oddsportal_success_scheduling_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
