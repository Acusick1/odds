"""Fetch results and closing odds from OddsPortal historical pages.

Scrapes recent match results, updates SCHEDULED events to FINAL with scores,
and stores one closing odds snapshot per event. Designed for daily Lambda
execution on the scraper Lambda (requires Playwright/Chromium).

Sport-aware: resolves league configuration from the shared ``LeagueSpec``
registry in ``fetch_oddsportal``. Derives market key, num_outcomes, and
harvester params from the spec.

Idempotency: only queries SCHEDULED/LIVE events with commence_time in the past,
so re-runs after events are marked FINAL find nothing to process.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT, LeagueSpec
from odds_lambda.oddsportal_common import (
    build_raw_data,
    parse_match_date,
    run_scraper_with_retry,
)
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Default sport key when no --sport is provided (backward-compatible)
DEFAULT_SPORT_KEY = "soccer_epl"
API_REQUEST_ID = "oddsportal_closing"


def _market_key_for_spec(spec: LeagueSpec) -> str:
    """Derive the OddsHarvester record key for the spec's primary market.

    OddsHarvester stores market data under ``{market}_market`` keys, e.g.
    ``"1x2_market"`` or ``"home_away_market"``.
    """
    return f"{spec.primary_market}_market"


def _db_market_for_spec(spec: LeagueSpec) -> str:
    """Derive the pipeline ``db_market`` value from the spec's primary market.

    Maps OddsPortal market names to the pipeline's canonical market key used
    in ``raw_data["bookmakers"][*]["markets"][*]["key"]``.
    """
    if spec.primary_market == "home_away":
        return "h2h"
    if spec.primary_market == "1x2":
        return "1x2"
    raise ValueError(f"Unrecognized primary_market: {spec.primary_market!r}")


@dataclass
class ResultsStats:
    """Tracks results of a single results-collection run."""

    matches_scraped: int = 0
    events_updated: int = 0
    snapshots_stored: int = 0
    events_not_matched: int = 0
    errors: list[str] = field(default_factory=list)


async def run_harvester_historic(spec: LeagueSpec) -> list[dict[str, Any]]:
    """Scrape historical results for the given league via ``run_scraper_with_retry``."""
    from oddsharvester.utils.command_enum import CommandEnum

    result = await run_scraper_with_retry(
        command=CommandEnum.HISTORIC,
        sport=spec.sport,
        leagues=[spec.league],
        season=None,
        max_pages=1,
        markets=[spec.primary_market],
        headless=True,
    )
    return result.success


async def get_pending_events(
    session: AsyncSession, sport_key: str = DEFAULT_SPORT_KEY
) -> list[Event]:
    """Get SCHEDULED or LIVE events with commence_time in the past for the given sport."""
    query = select(Event).where(
        and_(
            Event.sport_key == sport_key,
            Event.status.in_([EventStatus.SCHEDULED, EventStatus.LIVE]),
            Event.commence_time < datetime.now(UTC),
        )
    )
    result = await session.execute(query)
    return list(result.scalars().all())


def _parse_score(record: dict[str, Any]) -> tuple[int, int] | None:
    """Extract home and away scores from a scraped record."""
    home_str = str(record.get("home_score", ""))
    away_str = str(record.get("away_score", ""))
    if home_str.isdigit() and away_str.isdigit():
        return int(home_str), int(away_str)
    return None


def _match_record_to_event(
    record: dict[str, Any],
    pending_events: list[Event],
) -> Event | None:
    """Match a scraped record to a pending event by team name and date window."""
    home_team = record.get("home_team", "").strip()
    away_team = record.get("away_team", "").strip()
    match_date_str = record.get("match_date", "")

    if not home_team or not away_team or not match_date_str:
        return None

    try:
        match_dt = parse_match_date(match_date_str)
    except (ValueError, TypeError):
        return None

    for event in pending_events:
        if event.home_team != home_team or event.away_team != away_team:
            continue
        if event.commence_time is None:
            continue
        delta = abs((event.commence_time - match_dt).total_seconds())
        if delta <= 24 * 3600:
            return event

    return None


async def process_results(
    raw_matches: list[dict[str, Any]] | None = None,
    sport: str | None = None,
) -> ResultsStats:
    """Process scraped results: update scores and store closing snapshots.

    Args:
        raw_matches: Pre-scraped data (skips harvester call). For testing.
        sport: Sport key to filter (e.g. "soccer_epl"). Defaults to DEFAULT_SPORT_KEY.
    """
    stats = ResultsStats()
    sport_key = sport or DEFAULT_SPORT_KEY

    spec = _LEAGUE_SPEC_BY_SPORT.get(sport_key)
    if spec is None:
        logger.error(
            "unknown_sport_key",
            sport_key=sport_key,
            available=list(_LEAGUE_SPEC_BY_SPORT),
        )
        return stats

    market_key = _market_key_for_spec(spec)
    db_market = _db_market_for_spec(spec)

    async with async_session_maker() as session:
        pending_events = await get_pending_events(session, sport_key=sport_key)

        if not pending_events:
            logger.info("no_pending_events", sport_key=sport_key)
            return stats

        logger.info("pending_events_found", count=len(pending_events))

        if raw_matches is None:
            raw_matches = await run_harvester_historic(spec)

        stats.matches_scraped = len(raw_matches)

        if not raw_matches:
            logger.warning("no_matches_scraped")
            return stats

        logger.info("matches_scraped", count=len(raw_matches))

        writer = OddsWriter(session)

        for record in raw_matches:
            try:
                event = _match_record_to_event(record, pending_events)
                if event is None:
                    stats.events_not_matched += 1
                    continue

                scores = _parse_score(record)
                if scores is None:
                    logger.warning(
                        "no_scores_in_record",
                        home=record.get("home_team"),
                        away=record.get("away_team"),
                    )
                    continue

                home_score, away_score = scores

                # Update event status
                await writer.update_event_status(
                    event_id=event.id,
                    status=EventStatus.FINAL,
                    home_score=home_score,
                    away_score=away_score,
                )
                stats.events_updated += 1

                # Remove from pending list so it can't match again
                pending_events.remove(event)

                # Store closing odds snapshot
                market_data = record.get(market_key, [])
                if not market_data:
                    continue

                match_dt = parse_match_date(record["match_date"])
                raw_data = build_raw_data(
                    market_data,
                    event.home_team,
                    event.away_team,
                    use_opening=False,
                    match_dt=match_dt,
                    num_outcomes=spec.num_outcomes,
                    db_market=db_market,
                )
                if raw_data is None:
                    continue

                snapshot_time_str = raw_data.pop("_snapshot_time", None)
                if not snapshot_time_str:
                    continue

                snapshot_time = datetime.fromisoformat(snapshot_time_str.replace("Z", "+00:00"))

                snapshot, _ = await writer.store_odds_snapshot(
                    event_id=event.id,
                    raw_data=raw_data,
                    snapshot_time=snapshot_time,
                )
                snapshot.api_request_id = API_REQUEST_ID
                stats.snapshots_stored += 1

            except Exception as e:
                msg = f"{record.get('home_team', '?')} vs {record.get('away_team', '?')}: {e}"
                stats.errors.append(msg)
                logger.error("result_processing_failed", error=msg, exc_info=True)

        await session.commit()

    logger.info(
        "results_collection_complete",
        matches_scraped=stats.matches_scraped,
        events_updated=stats.events_updated,
        snapshots_stored=stats.snapshots_stored,
        events_not_matched=stats.events_not_matched,
        errors=len(stats.errors),
    )

    return stats


async def main(ctx: JobContext) -> None:
    """Main job entry point — runs results collection, then self-schedules for tomorrow."""
    from odds_core.alerts import job_alert_context
    from odds_core.config import get_settings

    settings = get_settings()
    sport = ctx.sport

    async with job_alert_context(make_compound_job_name("fetch-oddsportal-results", sport)):
        logger.info("fetch_oddsportal_results_started", sport=sport)
        await process_results(sport=sport)

    # Self-schedule: run again tomorrow at 08:00 UTC
    now = datetime.now(UTC)
    tomorrow_8am = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

    schedule_payload: dict[str, object] | None = None
    if sport:
        schedule_payload = {"sport": sport}

    try:
        from odds_lambda.scheduling.backends import get_scheduler_backend

        backend = get_scheduler_backend(dry_run=settings.scheduler.dry_run)
        schedule_job_name = make_compound_job_name("fetch-oddsportal-results", sport)
        await backend.schedule_next_execution(
            job_name=schedule_job_name,
            next_time=tomorrow_8am,
            payload=schedule_payload,
        )
        logger.info(
            "fetch_oddsportal_results_next_scheduled",
            next_time=tomorrow_8am.isoformat(),
            backend=backend.get_backend_name(),
        )
    except Exception as e:
        logger.error("fetch_oddsportal_results_scheduling_failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
