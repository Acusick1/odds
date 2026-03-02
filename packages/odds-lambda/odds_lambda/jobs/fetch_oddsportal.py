"""Fetch upcoming odds from OddsPortal via OddsHarvester.

Scrapes pre-match odds for configured leagues, converts to pipeline format,
and stores via OddsWriter. Designed for hourly Lambda execution or manual
CLI invocation.

This job:
1. Scrapes upcoming matches from OddsPortal (via OddsHarvester)
2. Converts fractional odds to pipeline raw_data format
3. Matches or creates Event records
4. Stores snapshots via OddsWriter.store_odds_snapshot()
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from odds_lambda.oddsportal_common import team_abbrev
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

FORK_SPEC = (
    "git+https://github.com/Acusick1/OddsHarvester.git@fix/fractional-odds-and-unknown-bookmaker"
)


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


def run_harvester_upcoming(spec: LeagueSpec) -> list[dict[str, Any]]:
    """Run OddsHarvester upcoming command and return parsed match list.

    Tries direct import first, falls back to subprocess via uvx.

    Raises:
        RuntimeError: If both import and subprocess paths fail.
    """
    try:
        from OddsHarvester.scraper import OddsScraper

        scraper = OddsScraper(headless=True, odds_format="Fractional Odds")
        raw = scraper.upcoming(sport=spec.sport, league=spec.league)
        return raw if isinstance(raw, list) else []
    except ImportError:
        pass

    return _run_harvester_subprocess(spec)


def _run_harvester_subprocess(spec: LeagueSpec) -> list[dict[str, Any]]:
    """Run oddsharvester via uvx subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        tmp_path = Path(tf.name)

    try:
        cmd = [
            "uvx",
            "--from",
            FORK_SPEC,
            "oddsharvester",
            "upcoming",
            "-s",
            spec.sport,
            "-l",
            spec.league,
            "--headless",
            "--odds-format",
            "Fractional Odds",
            "-f",
            "json",
            "-o",
            str(tmp_path),
        ]

        logger.info("running_harvester", cmd=" ".join(cmd[-8:]))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"OddsHarvester exited with code {result.returncode}: {result.stderr[:500]}"
            )

        data = json.loads(tmp_path.read_text())
        return data if isinstance(data, list) else []

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"OddsHarvester subprocess failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


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
        raw_matches = run_harvester_upcoming(spec)

    stats.matches_scraped = len(raw_matches)

    if not raw_matches:
        logger.warning("no_matches_scraped", league=spec.league)
        return stats

    logger.info("matches_scraped", league=spec.league, count=len(raw_matches))

    # Convert all markets
    all_converted: list[tuple[str, MatchOdds]] = []
    for market in spec.markets:
        converted = convert_upcoming_matches(raw_matches, market)
        for m in converted:
            all_converted.append((market, m))
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

                # Tag with api_request_id for querying
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


async def main() -> None:
    """Main job execution — iterates all configured leagues."""
    logger.info("fetch_oddsportal_started", leagues=len(LEAGUE_SPECS))

    all_stats: list[IngestionStats] = []

    for spec in LEAGUE_SPECS:
        try:
            stats = await ingest_league(spec)
            all_stats.append(stats)
        except Exception as e:
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
        leagues=len(all_stats),
        total_matches_scraped=total_scraped,
        total_snapshots_stored=total_snapshots,
        total_errors=total_errors,
    )


if __name__ == "__main__":
    asyncio.run(main())
