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
from datetime import UTC, datetime
from typing import Any

import structlog
from odds_core.database import async_session_maker

from odds_lambda.oddsportal_adapter import (
    MatchOdds,
    convert_upcoming_matches,
)
from odds_lambda.oddsportal_common import run_scraper_with_retry
from odds_lambda.scheduling.decision import (
    CadenceConfig,
    ScheduleDecision,
    decide_forward_resilient,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Proximity-based polling cadence (hours): browser scrape — moderate cadence.
# 12h+ to kickoff (and no-game) all poll at 2h, matching the previous "far" band.
CADENCE = CadenceConfig(
    closing=0.5,
    pregame=1.0,
    sharp=2.0,
    early=2.0,
    opening=2.0,
    no_game=2.0,
    db_fallback=1.0,
)


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
    overnight = (primary_spec.overnight_start_utc, primary_spec.overnight_resume_utc)
    sport_keys = [s.sport_key for s in specs]
    # Most conservative lead across targeted sports (combined runs wake earliest).
    lead_days = min(settings.scheduler.lead_days_for(sk) for sk in sport_keys)

    # Season-gated scraper: the decision's ``should_execute`` says whether the
    # next fixture is within the lead window. When it is not we self-schedule the
    # precise wake and skip all browser work (offseason / long mid-season break).
    async def _reschedule(label: str) -> ScheduleDecision:
        decision = await decide_forward_resilient(
            sport_keys, CADENCE, overnight=overnight, lookahead_days=lead_days
        )
        assert decision.next_execution is not None  # noqa: S101
        try:
            await self_schedule(
                job_name=compound_job_name,
                next_time=decision.next_execution,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                reason=f"{label}: {decision.reason}",
            )
        except Exception as e:
            logger.error("fetch_oddsportal_scheduling_failed", error=str(e), exc_info=True)
            if label == "pre-schedule":
                raise
        return decision

    # Pre-schedule before any browser work so the chain survives Lambda
    # timeouts or browser crashes.
    decision = await _reschedule("pre-schedule")

    # Season gate: no fixture within the lead window — skip the 2 GB browser run.
    if not decision.should_execute:
        next_execution = decision.next_execution
        assert next_execution is not None  # noqa: S101
        logger.info(
            "fetch_oddsportal_season_gated",
            sport=sport,
            next_execution=next_execution.isoformat(),
            reason=decision.reason,
        )
        return

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

    # Outside job_alert_context so scoring failures don't fire scrape alerts.
    try:
        from odds_lambda.model_loader import model_supports_sport

        if model_supports_sport(ctx.sport):
            from odds_lambda.jobs.score_predictions import score_events

            score_stats = await score_events(sport=ctx.sport)
            logger.info("fetch_oddsportal_scoring_complete", **score_stats)
        else:
            logger.info("fetch_oddsportal_scoring_skipped", sport=sport)
    except Exception as e:
        logger.error(
            "fetch_oddsportal_scoring_failed",
            sport=sport,
            error=str(e),
            exc_info=True,
        )

    # On success, re-query next kickoff (scrape may have created new events)
    # and reschedule at the updated interval.
    if total_scraped > 0:
        await _reschedule("post-scrape")


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
