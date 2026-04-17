"""Fetch upcoming odds from OddsPortal via OddsHarvester.

Scrapes pre-match odds for configured leagues, converts to pipeline format,
and stores via OddsWriter. Self-schedules based on proximity to next kickoff.

Scheduling strategy: query DB for next kickoff, compute interval from game
proximity, pre-schedule before any browser work so the chain survives Lambda
timeouts. After a successful scrape, re-query (scrape may have created new
events) and reschedule at the updated interval.

Interval table:
  < 3h  (CLOSING)   → 30 min
  3–12h (PREGAME)   → 1h
  12h+  or no games → 2h
  DB unreachable    → 1h (fallback)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.database import async_session_maker

from odds_lambda.fetch_tier import FetchTier
from odds_lambda.oddsportal_adapter import (
    MatchOdds,
    convert_upcoming_matches,
)
from odds_lambda.oddsportal_common import hours_to_tier, run_scraper_with_retry
from odds_lambda.scheduling.helpers import apply_overnight_skip, get_next_kickoff, self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Proximity-based scheduling intervals (hours)
CLOSING_INTERVAL_HOURS = 0.5  # < 3h to kickoff
PREGAME_INTERVAL_HOURS = 1.0  # 3–12h to kickoff
FAR_INTERVAL_HOURS = 2.0  # 12h+ or no upcoming games
DB_FALLBACK_INTERVAL_HOURS = 1.0  # DB unreachable


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

# Index by league name for fast lookup from scrape commands / MCP tools
LEAGUE_SPEC_BY_NAME: dict[str, LeagueSpec] = {spec.league: spec for spec in LEAGUE_SPECS}


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
    """Scrape upcoming matches for a league via ``run_scraper_with_retry``.

    Acquires the Playwright semaphore so only one browser instance runs at a time.
    """
    from oddsharvester.utils.command_enum import CommandEnum

    from odds_lambda.browser_lock import playwright_semaphore

    async with playwright_semaphore:
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


def _interval_for_kickoff(
    next_kickoff: datetime | None,
    *,
    now: datetime | None = None,
) -> float:
    """Compute scheduling interval (hours) based on proximity to next kickoff."""
    if next_kickoff is None:
        return FAR_INTERVAL_HOURS

    if now is None:
        now = datetime.now(UTC)

    hours_until = (next_kickoff - now).total_seconds() / 3600

    if hours_until < FetchTier.CLOSING.max_hours:
        return CLOSING_INTERVAL_HOURS
    if hours_until < FetchTier.PREGAME.max_hours:
        return PREGAME_INTERVAL_HOURS
    return FAR_INTERVAL_HOURS


async def main(ctx: JobContext) -> None:
    """Main job execution — scrapes configured league(s), then self-schedules.

    Scheduling strategy: query DB for next kickoff, compute proximity-based
    interval, pre-schedule before browser work. After successful scrape,
    re-query (new events may have appeared) and reschedule.
    """
    from odds_core.alerts import job_alert_context, send_job_warning
    from odds_core.config import get_settings

    settings = get_settings()
    sport = ctx.sport

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
    # Combined job (no --sport) uses first spec's overnight window.
    # Production invokes per-sport; only relevant for local multi-league runs.
    primary_spec = specs[0]

    # Query DB for next kickoff to determine scheduling interval.
    # On failure, fall back to 1h so we don't block on DB availability.
    pre_interval = DB_FALLBACK_INTERVAL_HOURS
    try:
        next_kickoff = await get_next_kickoff(primary_spec.sport_key)
        pre_interval = _interval_for_kickoff(next_kickoff)
        logger.info(
            "proximity_schedule",
            next_kickoff=next_kickoff.isoformat() if next_kickoff else None,
            interval_hours=pre_interval,
        )
    except Exception as e:
        logger.warning("next_kickoff_query_failed", error=str(e), exc_info=True)

    # Pre-schedule before any browser work so the chain survives Lambda
    # timeouts or browser crashes.
    pre_next_time = apply_overnight_skip(
        datetime.now(UTC) + timedelta(hours=pre_interval),
        overnight_start_utc=primary_spec.overnight_start_utc,
        overnight_resume_utc=primary_spec.overnight_resume_utc,
    )
    try:
        await self_schedule(
            job_name=compound_job_name,
            next_time=pre_next_time,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            interval_hours=pre_interval,
        )
    except Exception as e:
        logger.error("fetch_oddsportal_scheduling_failed", error=str(e), exc_info=True)
        raise

    async with job_alert_context(compound_job_name):
        logger.info(
            "fetch_oddsportal_started",
            sport=sport,
            leagues=len(specs),
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
                message=f"⚠️ OddsPortal scrape empty ({detail})",
            )

    # On success, re-query next kickoff (scrape may have created new events)
    # and reschedule at the updated interval.
    if total_scraped > 0:
        post_interval = DB_FALLBACK_INTERVAL_HOURS
        try:
            post_kickoff = await get_next_kickoff(primary_spec.sport_key)
            post_interval = _interval_for_kickoff(post_kickoff)
        except Exception as e:
            logger.warning("post_scrape_kickoff_query_failed", error=str(e), exc_info=True)

        success_next_time = apply_overnight_skip(
            datetime.now(UTC) + timedelta(hours=post_interval),
            overnight_start_utc=primary_spec.overnight_start_utc,
            overnight_resume_utc=primary_spec.overnight_resume_utc,
        )
        try:
            await self_schedule(
                job_name=compound_job_name,
                next_time=success_next_time,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                interval_hours=post_interval,
            )
        except Exception as e:
            logger.error("fetch_oddsportal_success_scheduling_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
