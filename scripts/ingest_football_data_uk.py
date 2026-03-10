#!/usr/bin/env python3
"""Ingest football-data.co.uk EPL CSVs into the database.

Downloads CSVs from football-data.co.uk and writes Event + OddsSnapshot records
following the same raw_data format as ingest_oddsportal.py, so the training
pipeline reads them via extract_odds_from_snapshot() without changes.

Usage:
    # Download and ingest all seasons (dry-run first)
    uv run python scripts/ingest_football_data_uk.py --all --dry-run

    # Ingest specific seasons
    uv run python scripts/ingest_football_data_uk.py --seasons 2024-2025

    # Ingest from already-downloaded CSVs
    uv run python scripts/ingest_football_data_uk.py --all --no-download

    # Download only (no database writes)
    uv run python scripts/ingest_football_data_uk.py --all --download-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.oddsportal_common import decimal_to_american, hours_to_tier, team_abbrev
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_football_data_uk")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "football_data_uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
SOURCE_TAG = "football_data_uk"
SPORT_KEY = "soccer_epl"
SPORT_TITLE = "EPL"

# Season label → URL path segment
SEASONS: dict[str, str] = {
    "2015-2016": "1516",
    "2016-2017": "1617",
    "2017-2018": "1718",
    "2018-2019": "1819",
    "2019-2020": "1920",
    "2020-2021": "2021",
    "2021-2022": "2122",
    "2022-2023": "2223",
    "2023-2024": "2324",
    "2024-2025": "2425",
    "2025-2026": "2526",
}

# football-data.co.uk team name → OddsPortal / pipeline canonical name.
# Only entries that differ need mapping; matching names are passed through.
# Verified against OddsPortal team names used in existing DB records.
# Teams not listed here pass through unchanged (e.g. Arsenal, Chelsea,
# Liverpool, Everton, Bournemouth, Brentford, Crystal Palace, Fulham,
# Aston Villa, Southampton, Burnley, Watford, Luton, Ipswich).
TEAM_NAME_MAP: dict[str, str] = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Tottenham": "Tottenham Hotspur",
    "Brighton": "Brighton and Hove Albion",
    "West Brom": "West Bromwich Albion",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Huddersfield": "Huddersfield Town",
    "Cardiff": "Cardiff City",
    "Hull": "Hull City",
    "Sunderland": "Sunderland AFC",
    "Middlesbrough": "Middlesbrough FC",
}

# Bookmaker column prefixes → pipeline bookmaker key.
# Each entry maps (home_col, draw_col, away_col) derived from prefix + H/D/A.
BOOKMAKER_PREFIXES: dict[str, str] = {
    "PS": "pinnacle",
    "B365": "bet365",
    "BW": "bwin",
    "BFE": "betfair_exchange",
    "IW": "interwetten",
    "WH": "williamhill",
    "VC": "betvictor",
    "LB": "ladbrokes",
    "BFD": "betfair_sportsbook",
    "BMGM": "betmgm",
    "BV": "betvictor",
    "CL": "coral",
    "1XB": "onexbet",
}

# Aggregate/market columns — stored as pseudo-bookmakers for analysis.
AGGREGATE_PREFIXES: dict[str, str] = {
    "Max": "market_max",
    "Avg": "market_avg",
}


@dataclass
class IngestionStats:
    games_loaded: int = 0
    games_skipped: int = 0
    events_matched: int = 0
    events_created: int = 0
    snapshots_inserted: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_team(name: str) -> str:
    """Normalize football-data.co.uk team name to pipeline canonical form."""
    return TEAM_NAME_MAP.get(name, name)


def _season_url(season: str) -> str:
    """Build CSV download URL for a season."""
    code = SEASONS[season]
    return f"{BASE_URL}/{code}/E0.csv"


def _csv_path(season: str) -> Path:
    """Local CSV storage path for a season."""
    code = SEASONS[season]
    return DATA_DIR / f"E0_{code}.csv"


def download_csv(season: str) -> Path:
    """Download a season CSV from football-data.co.uk."""
    url = _season_url(season)
    path = _csv_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=30) as resp:  # noqa: S310
        path.write_bytes(resp.read())

    log.info(f"  Downloaded {url} → {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a football-data.co.uk CSV file."""
    content = path.read_bytes().decode("latin-1")
    reader = csv.DictReader(io.StringIO(content))
    return [row for row in reader if row.get("HomeTeam")]


def _parse_date(row: dict[str, str]) -> datetime:
    """Parse match date + time from CSV row to UTC datetime.

    Date format is DD/MM/YYYY. Time column may or may not exist.
    UK kickoff times are in local time (GMT/BST); we approximate as UTC
    since the exact timezone offset per date isn't worth the complexity
    for historical ingestion (max 1h error).
    """
    date_str = row["Date"]
    time_str = row.get("Time", "15:00")
    if not time_str:
        time_str = "15:00"
    return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M").replace(tzinfo=UTC)


def _safe_float(value: str | None) -> float | None:
    """Parse a float from CSV, returning None for empty/invalid values."""
    if not value or not value.strip():
        return None
    try:
        f = float(value)
        return f if not math.isnan(f) else None
    except ValueError:
        return None


def _safe_int(value: str | None) -> int | None:
    """Parse an int from CSV, returning None for empty/invalid values."""
    if not value or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _extract_h2h_odds(
    row: dict[str, str],
    prefix: str,
    bk_key: str,
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    """Extract 1x2 h2h odds for a bookmaker from a CSV row."""
    home_odds = _safe_float(row.get(f"{prefix}H"))
    draw_odds = _safe_float(row.get(f"{prefix}D"))
    away_odds = _safe_float(row.get(f"{prefix}A"))

    if home_odds is None or draw_odds is None or away_odds is None:
        return None
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return None

    return {
        "key": bk_key,
        "title": bk_key,
        "last_update": "",  # placeholder, set at snapshot level
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": decimal_to_american(home_odds)},
                    {"name": "Draw", "price": decimal_to_american(draw_odds)},
                    {"name": away_team, "price": decimal_to_american(away_odds)},
                ],
            }
        ],
    }


def _build_snapshot_raw_data(
    row: dict[str, str],
    home_team: str,
    away_team: str,
    *,
    use_closing: bool,
) -> dict[str, Any] | None:
    """Build raw_data dict from a CSV row for one timing (opening or closing).

    Opening odds: columns like PSH, PSD, PSA, B365H, B365D, B365A
    Closing odds: columns like PSCH, PSCD, PSCA, B365CH, B365CD, B365CA
    """
    bookmakers: list[dict[str, Any]] = []

    all_prefixes = {**BOOKMAKER_PREFIXES, **AGGREGATE_PREFIXES}

    for prefix, bk_key in all_prefixes.items():
        if use_closing:
            col_prefix = f"{prefix}C"
        else:
            col_prefix = prefix

        bk_data = _extract_h2h_odds(row, col_prefix, bk_key, home_team, away_team)
        if bk_data:
            bookmakers.append(bk_data)

    if not bookmakers:
        return None

    return {
        "bookmakers": bookmakers,
        "source": SOURCE_TAG,
    }


async def find_existing_event(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> str | None:
    """Find an existing Event matching the given game within a +/-24h window."""
    window_start = commence_time - timedelta(hours=24)
    window_end = commence_time + timedelta(hours=24)

    query = select(Event.id).where(
        and_(
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            Event.home_team == home_team,
            Event.away_team == away_team,
        )
    )
    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning(
            f"Ambiguous match for {away_team} @ {home_team} on "
            f"{commence_time.date()}: {len(candidates)} candidates"
        )
    return None


def build_event_id(
    season: str,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> str:
    """Generate deterministic event ID for football-data.co.uk events."""
    home_abbrev = team_abbrev(home_team)
    away_abbrev = team_abbrev(away_team)
    return f"fduk_{season}_{home_abbrev}_{away_abbrev}_{commence_time.date().isoformat()}"


async def ingest_season(
    season: str,
    *,
    dry_run: bool = False,
    no_download: bool = False,
) -> IngestionStats:
    """Ingest one season of football-data.co.uk data."""
    stats = IngestionStats()

    path = _csv_path(season)
    if not no_download:
        try:
            path = download_csv(season)
        except OSError as e:
            log.error(f"  {season}: download failed — {e}")
            stats.errors.append(f"{season}: download failed — {e}")
            return stats
    elif not path.exists():
        log.error(f"  {season}: CSV not found at {path}")
        stats.errors.append(f"{season}: CSV not found at {path}")
        return stats

    rows = read_csv(path)
    if not rows:
        log.error(f"  {season}: no data rows in CSV")
        return stats

    stats.games_loaded = len(rows)
    log.info(f"  {season}: loaded {len(rows)} matches")

    if dry_run:
        _report_dry_run(season, rows)
        return stats

    async with async_session_maker() as session:
        for row in rows:
            try:
                await _ingest_one_match(session, row, season, stats)
            except Exception as e:
                home = row.get("HomeTeam", "?")
                away = row.get("AwayTeam", "?")
                msg = f"{home} vs {away}: {e}"
                stats.errors.append(msg)
                log.error(f"  Error: {msg}")

        await session.commit()

    return stats


def _report_dry_run(season: str, rows: list[dict[str, str]]) -> None:
    """Log dry-run summary for a season."""
    has_pinnacle_opening = sum(1 for r in rows if _safe_float(r.get("PSH")))
    has_pinnacle_closing = sum(1 for r in rows if _safe_float(r.get("PSCH")))
    has_bet365_closing = sum(1 for r in rows if _safe_float(r.get("B365CH")))
    has_betfair = sum(1 for r in rows if _safe_float(r.get("BFEH")))

    log.info(
        f"  {season}: Pinnacle opening={has_pinnacle_opening}, "
        f"Pinnacle closing={has_pinnacle_closing}, "
        f"bet365 closing={has_bet365_closing}, "
        f"Betfair={has_betfair}"
    )


async def _ingest_one_match(
    session: AsyncSession,
    row: dict[str, str],
    season: str,
    stats: IngestionStats,
) -> None:
    """Ingest a single CSV row as Event + OddsSnapshots."""
    home_raw = row.get("HomeTeam", "")
    away_raw = row.get("AwayTeam", "")
    if not home_raw or not away_raw:
        stats.games_skipped += 1
        return

    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)
    commence_time = _parse_date(row)

    # --- Match or create Event ---
    event_id = await find_existing_event(session, home_team, away_team, commence_time)

    if event_id:
        stats.events_matched += 1
    else:
        event_id = build_event_id(season, home_team, away_team, commence_time)

        existing = await session.get(Event, event_id)
        if existing:
            stats.events_matched += 1
        else:
            home_score = _safe_int(row.get("FTHG"))
            away_score = _safe_int(row.get("FTAG"))
            is_final = home_score is not None and away_score is not None

            event = Event(
                id=event_id,
                sport_key=SPORT_KEY,
                sport_title=SPORT_TITLE,
                commence_time=commence_time,
                home_team=home_team,
                away_team=away_team,
                status=EventStatus.FINAL if is_final else EventStatus.SCHEDULED,
                home_score=home_score,
                away_score=away_score,
                completed_at=commence_time if is_final else None,
            )
            session.add(event)
            stats.events_created += 1

    # --- Check for existing snapshots (idempotency) ---
    existing_check = await session.execute(
        select(OddsSnapshot.id).where(
            and_(
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.api_request_id == SOURCE_TAG,
            )
        )
    )
    if existing_check.scalars().first() is not None:
        return

    # --- Build opening snapshot ---
    opening_data = _build_snapshot_raw_data(row, home_team, away_team, use_closing=False)
    if opening_data:
        # Opening odds: assume ~24h before match
        opening_time = commence_time - timedelta(hours=24)
        hours_before = 24.0
        tier = hours_to_tier(hours_before)

        for bk in opening_data["bookmakers"]:
            bk["last_update"] = opening_time.isoformat().replace("+00:00", "Z")

        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=opening_time,
            raw_data=opening_data,
            bookmaker_count=len(opening_data["bookmakers"]),
            api_request_id=SOURCE_TAG,
            fetch_tier=tier,
            hours_until_commence=hours_before,
        )
        session.add(snapshot)
        stats.snapshots_inserted += 1

    # --- Build closing snapshot ---
    closing_data = _build_snapshot_raw_data(row, home_team, away_team, use_closing=True)
    if closing_data:
        # Closing odds: ~1h before match
        closing_time = commence_time - timedelta(hours=1)
        hours_before = 1.0
        tier = hours_to_tier(hours_before)

        for bk in closing_data["bookmakers"]:
            bk["last_update"] = closing_time.isoformat().replace("+00:00", "Z")

        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=closing_time,
            raw_data=closing_data,
            bookmaker_count=len(closing_data["bookmakers"]),
            api_request_id=SOURCE_TAG,
            fetch_tier=tier,
            hours_until_commence=hours_before,
        )
        session.add(snapshot)
        stats.snapshots_inserted += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest football-data.co.uk EPL data into the database"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to ingest (e.g. 2024-2025)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available seasons (2015-2026)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and preview without writing to DB",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use already-downloaded CSVs (skip HTTP requests)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download CSVs without database ingestion",
    )
    args = parser.parse_args()

    if args.all:
        seasons = list(SEASONS.keys())
    elif args.seasons:
        for s in args.seasons:
            if s not in SEASONS:
                parser.error(f"Unknown season: {s}. Available: {list(SEASONS.keys())}")
        seasons = args.seasons
    else:
        parser.error("Specify --seasons or --all")

    asyncio.run(
        _run(
            seasons,
            dry_run=args.dry_run,
            no_download=args.no_download,
            download_only=args.download_only,
        )
    )


async def _run(
    seasons: list[str],
    *,
    dry_run: bool,
    no_download: bool,
    download_only: bool,
) -> None:
    totals = IngestionStats()

    for i, season in enumerate(seasons):
        log.info(f"[{i + 1}/{len(seasons)}] {season}")

        if download_only:
            try:
                download_csv(season)
                totals.games_loaded += 1
            except OSError as e:
                log.error(f"  {season}: download failed — {e}")
                totals.errors.append(f"{season}: {e}")
            continue

        stats = await ingest_season(
            season,
            dry_run=dry_run,
            no_download=no_download,
        )

        totals.games_loaded += stats.games_loaded
        totals.games_skipped += stats.games_skipped
        totals.events_matched += stats.events_matched
        totals.events_created += stats.events_created
        totals.snapshots_inserted += stats.snapshots_inserted
        totals.errors.extend(stats.errors)

        if not dry_run and not download_only:
            log.info(
                f"  → matched={stats.events_matched}, created={stats.events_created}, "
                f"snapshots={stats.snapshots_inserted}, skipped={stats.games_skipped}"
            )

    log.info("")
    log.info("=== Summary ===")
    if download_only:
        log.info(f"Seasons downloaded: {totals.games_loaded}")
    else:
        log.info(f"Matches loaded:     {totals.games_loaded}")
        log.info(f"Matches skipped:    {totals.games_skipped}")
        log.info(f"Events matched:     {totals.events_matched}")
        log.info(f"Events created:     {totals.events_created}")
        log.info(f"Snapshots inserted: {totals.snapshots_inserted}")

    if totals.errors:
        log.info(f"Errors:             {len(totals.errors)}")
        for err in totals.errors[:10]:
            log.error(f"  {err}")


if __name__ == "__main__":
    main()
