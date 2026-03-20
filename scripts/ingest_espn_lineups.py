#!/usr/bin/env python3
"""Fetch starting XIs and squad lists from ESPN Summary API.

Downloads rosters for every EPL match (2020-21 through 2025-26) via the ESPN
scoreboard and summary endpoints, writing one CSV per season to
data/espn_lineups/.

Usage:
    # Fetch all seasons (2020-2025)
    uv run python scripts/ingest_espn_lineups.py

    # Fetch a single season
    uv run python scripts/ingest_espn_lineups.py --season 2024
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
from team_names import normalize_team as _normalize_team_central

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_espn_lineups")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "espn_lineups"

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "espn_fixtures"

SEASONS: dict[int, str] = {
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}

# EPL season date ranges (start year -> (first_date, last_date)).
SEASON_DATE_RANGES: dict[int, tuple[date, date]] = {
    2020: (date(2020, 9, 12), date(2021, 5, 23)),
    2021: (date(2021, 8, 13), date(2022, 5, 22)),
    2022: (date(2022, 8, 5), date(2023, 5, 28)),
    2023: (date(2023, 8, 11), date(2024, 5, 19)),
    2024: (date(2024, 8, 16), date(2025, 5, 25)),
    2025: (date(2025, 8, 16), date(2026, 5, 24)),
}

CSV_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "team",
    "player_id",
    "player_name",
    "position",
    "starter",
    "formation_place",
]

REQUEST_DELAY = 0.5  # seconds between HTTP requests

# Position abbreviation mapping from ESPN full names.
POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "G",
    "Defender": "D",
    "Midfielder": "M",
    "Forward": "F",
}


def _normalize_team(espn_name: str) -> str:
    """Convert ESPN displayName to pipeline canonical name."""
    return _normalize_team_central(espn_name)


def _fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    """Fetch JSON from ESPN API with retry."""
    for attempt in range(3):
        try:
            resp = client.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt == 2:
                raise
            log.warning(f"  Retry {attempt + 1} for {url}: {e}")
            time.sleep(1)
    raise RuntimeError("unreachable")


def _load_match_dates_from_fixtures(season: int) -> list[str] | None:
    """Load unique match dates from the ESPN fixtures CSV for a season.

    Returns sorted list of YYYYMMDD strings, or None if the CSV doesn't exist.
    """
    label = SEASONS[season]
    csv_path = FIXTURES_DIR / f"fixtures_{label}.csv"
    if not csv_path.exists():
        return None

    dates: set[str] = set()
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Only EPL matches (fixture CSV includes cups/European competitions)
            if row.get("competition") != "Premier League":
                continue
            date_str = row.get("date", "")
            if date_str:
                # Parse ISO date and convert to YYYYMMDD
                day = date_str[:10].replace("-", "")
                if len(day) == 8:
                    dates.add(day)

    return sorted(dates) if dates else None


def _discover_game_ids(client: httpx.Client, season: int) -> list[tuple[str, str]]:
    """Discover EPL game IDs for a season via the scoreboard endpoint.

    If the ESPN fixtures CSV exists for this season, only query match dates
    (~38 requests) instead of every calendar day (~270 requests).

    Returns list of (game_id, date_iso) tuples.
    """
    match_dates = _load_match_dates_from_fixtures(season)
    if match_dates is not None:
        log.info(f"  Using {len(match_dates)} match dates from fixtures CSV")
        dates_to_query = match_dates
    else:
        log.info("  Fixtures CSV not found, scanning all season dates")
        start_date, end_date = SEASON_DATE_RANGES[season]
        dates_to_query = []
        current = start_date
        while current <= end_date:
            dates_to_query.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

    games: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    for date_str in dates_to_query:
        url = f"{BASE_URL}/scoreboard?dates={date_str}"
        time.sleep(REQUEST_DELAY)

        try:
            data = _fetch_json(client, url)
        except Exception as e:
            log.warning(f"  Failed scoreboard for {date_str}: {e}")
            continue

        for event in data.get("events", []):
            game_id = event.get("id", "")
            event_date = event.get("date", "")
            if game_id and game_id not in seen_ids:
                seen_ids.add(game_id)
                games.append((game_id, event_date))

    log.info(f"  Discovered {len(games)} games from scoreboard")
    return games


def _fetch_lineup(client: httpx.Client, game_id: str, event_date: str) -> list[dict[str, str]]:
    """Fetch lineup rows for a single game via the summary endpoint."""
    url = f"{BASE_URL}/summary?event={game_id}"
    time.sleep(REQUEST_DELAY)
    data = _fetch_json(client, url)

    # Extract home/away team names from boxscore or header
    header = data.get("header", {})
    competitions = header.get("competitions", [])
    if not competitions:
        return []

    competitors = competitions[0].get("competitors", [])
    if len(competitors) != 2:
        return []

    # Build team info: ESPN uses homeAway field
    team_info: dict[str, dict[str, str]] = {}
    for comp in competitors:
        ha = comp.get("homeAway", "")
        team_name = _normalize_team(comp.get("team", {}).get("displayName", ""))
        team_id = comp.get("id", "")
        team_info[team_id] = {"name": team_name, "home_away": ha}

    home_team = ""
    away_team = ""
    for info in team_info.values():
        if info["home_away"] == "home":
            home_team = info["name"]
        else:
            away_team = info["name"]

    if not home_team or not away_team:
        return []

    # Extract rosters
    rosters = data.get("rosters", [])
    rows: list[dict[str, str]] = []

    for roster_entry in rosters:
        # Determine which team this roster belongs to
        roster_team_id = roster_entry.get("team", {}).get("id", "")
        info = team_info.get(roster_team_id)
        if info is None:
            # Try matching by displayName
            roster_team_name = roster_entry.get("team", {}).get("displayName", "")
            team_name = _normalize_team(roster_team_name)
        else:
            team_name = info["name"]

        for player in roster_entry.get("roster", []):
            athlete = player.get("athlete", {})
            player_id = str(athlete.get("id", ""))
            player_name = athlete.get("displayName", "")
            is_starter = player.get("starter", False)
            formation_place = player.get("formationPlace", 0)

            # Position
            position_raw = player.get("position", {})
            if isinstance(position_raw, dict):
                pos_name = position_raw.get("displayName", "")
            else:
                pos_name = str(position_raw)
            position = POSITION_MAP.get(pos_name, pos_name)

            rows.append(
                {
                    "date": event_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team_name,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "starter": str(is_starter),
                    "formation_place": str(formation_place),
                }
            )

    return rows


def fetch_season(client: httpx.Client, season: int) -> list[dict[str, str]]:
    """Fetch all lineup data for a season."""
    games = _discover_game_ids(client, season)
    all_rows: list[dict[str, str]] = []

    for i, (game_id, event_date) in enumerate(games):
        try:
            rows = _fetch_lineup(client, game_id, event_date)
            all_rows.extend(rows)
            if (i + 1) % 50 == 0:
                log.info(f"  [{i + 1}/{len(games)}] {len(all_rows)} player rows so far")
        except Exception as e:
            log.warning(f"  Failed summary for game {game_id}: {e}")
            continue

    log.info(f"  Total: {len(all_rows)} player rows from {len(games)} games")
    return all_rows


def write_csv(rows: list[dict[str, str]], season: int) -> Path:
    """Write lineup rows to CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label = SEASONS[season]
    path = DATA_DIR / f"lineups_{label}.csv"

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


async def write_db(rows: list[dict[str, str]], season: int) -> int:
    """Write lineup rows to database via EspnLineupWriter."""
    from datetime import UTC, datetime

    from odds_core.database import async_session_maker
    from odds_core.epl_data_models import EspnLineupRecord
    from odds_lambda.storage.espn_lineup_writer import EspnLineupWriter

    label = SEASONS[season]
    records: list[EspnLineupRecord] = []
    for r in rows:
        date_str = r["date"]
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        records.append(
            EspnLineupRecord(
                date=dt,
                home_team=r["home_team"],
                away_team=r["away_team"],
                team=r["team"],
                player_id=r["player_id"],
                player_name=r["player_name"],
                position=r.get("position", ""),
                starter=r["starter"].lower() == "true"
                if isinstance(r["starter"], str)
                else bool(r["starter"]),
                formation_place=int(r.get("formation_place", 0)),
                season=label,
            )
        )

    async with async_session_maker() as session:
        writer = EspnLineupWriter(session)
        count = await writer.upsert_lineups(records)
        await session.commit()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ESPN starting XIs for EPL matches")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single season start year (e.g. 2024 for 2024-25). Default: all seasons.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip writing to database (CSV only).",
    )
    args = parser.parse_args()

    if args.season is not None:
        if args.season not in SEASONS:
            parser.error(f"Unknown season: {args.season}. Available: {sorted(SEASONS)}")
        seasons = [args.season]
    else:
        seasons = sorted(SEASONS.keys())

    client = httpx.Client()
    total_rows = 0
    season_data: list[tuple[list[dict[str, str]], int]] = []

    try:
        for season in seasons:
            label = SEASONS[season]
            log.info(f"[{label}] Fetching lineups...")
            rows = fetch_season(client, season)
            path = write_csv(rows, season)
            total_rows += len(rows)
            log.info(f"[{label}] Wrote {len(rows)} player rows to {path}")
            season_data.append((rows, season))
    finally:
        client.close()

    if not args.skip_db and season_data:
        import asyncio

        async def _write_all() -> None:
            for rows, season in season_data:
                count = await write_db(rows, season)
                log.info(f"[{SEASONS[season]}] Upserted {count} lineup rows to database")

        asyncio.run(_write_all())

    log.info(f"\nTotal: {total_rows} player rows across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
