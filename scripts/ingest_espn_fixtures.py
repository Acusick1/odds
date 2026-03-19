#!/usr/bin/env python3
"""Fetch all-competition fixture dates from ESPN Site API.

Downloads fixture schedules for every EPL team across all relevant competitions
(Premier League, FA Cup, League Cup, Champions League, Europa League, Conference
League) and writes one CSV per season to data/espn_fixtures/.

These CSVs are loaded as a pandas DataFrame by the training pipeline to provide
accurate rest-day and congestion features that account for midweek European and
cup matches.

Usage:
    # Fetch all seasons (2015-2025)
    uv run python scripts/ingest_espn_fixtures.py

    # Fetch a single season
    uv run python scripts/ingest_espn_fixtures.py --season 2024
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from team_names import normalize_team as _normalize_team_central

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_espn_fixtures")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "espn_fixtures"

BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN league slugs for all competitions an EPL team might play in.
LEAGUE_SLUGS: dict[str, str] = {
    "eng.1": "Premier League",
    "eng.fa": "FA Cup",
    "eng.league_cup": "League Cup",
    "uefa.champions": "Champions League",
    "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League",
}

# European competition names for tagging.
EUROPEAN_COMPETITIONS = frozenset({"Champions League", "Europa League", "Conference League"})

# Seasons: start year -> display label (matches FDUK convention).
SEASONS: dict[int, str] = {
    2015: "2015-16",
    2016: "2016-17",
    2017: "2017-18",
    2018: "2018-19",
    2019: "2019-20",
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}

CSV_COLUMNS = [
    "date",
    "team",
    "opponent",
    "competition",
    "match_round",
    "home_away",
    "score_team",
    "score_opponent",
    "status",
]

REQUEST_DELAY = 0.5  # seconds between HTTP requests


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


def _extract_score(competitor: dict[str, Any]) -> str:
    """Extract score from a competitor entry. Returns empty string if unavailable."""
    score = competitor.get("score")
    if score is None:
        return ""
    if isinstance(score, dict):
        return score.get("displayValue", "")
    return str(score)


def fetch_teams(client: httpx.Client, season: int) -> list[dict[str, str]]:
    """Fetch EPL teams for a season. Returns list of {id, name}."""
    url = f"{BASE_URL}/eng.1/teams?season={season}"
    data = _fetch_json(client, url)
    teams = data["sports"][0]["leagues"][0]["teams"]
    return [{"id": t["team"]["id"], "name": t["team"]["displayName"]} for t in teams]


def fetch_team_schedule(
    client: httpx.Client,
    league_slug: str,
    team_id: str,
    season: int,
) -> list[dict[str, str]]:
    """Fetch a team's schedule for one competition/season.

    Returns list of fixture dicts matching CSV_COLUMNS.
    """
    url = f"{BASE_URL}/{league_slug}/teams/{team_id}/schedule?season={season}"
    data = _fetch_json(client, url)
    events = data.get("events", [])
    competition = LEAGUE_SLUGS[league_slug]
    fixtures: list[dict[str, str]] = []

    for event in events:
        date_str = event.get("date", "")
        if not date_str:
            continue

        comps = event.get("competitions", [])
        if not comps:
            continue

        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Identify team and opponent from competitors
        team_entry = None
        opponent_entry = None
        for c in competitors:
            if c["team"]["id"] == team_id:
                team_entry = c
            else:
                opponent_entry = c

        if team_entry is None or opponent_entry is None:
            continue

        team_name = _normalize_team(team_entry["team"]["displayName"])
        opponent_name = _normalize_team(opponent_entry["team"]["displayName"])

        # Status
        status_type = comp.get("status", {}).get("type", {})
        status = status_type.get("description", "")

        # Round/phase
        round_name = event.get("seasonType", {}).get("name", "")

        fixtures.append(
            {
                "date": date_str,
                "team": team_name,
                "opponent": opponent_name,
                "competition": competition,
                "match_round": round_name,
                "home_away": team_entry["homeAway"],
                "score_team": _extract_score(team_entry),
                "score_opponent": _extract_score(opponent_entry),
                "status": status,
            }
        )

    return fixtures


def fetch_season(client: httpx.Client, season: int) -> list[dict[str, str]]:
    """Fetch all fixtures for all EPL teams in a season across all competitions."""
    teams = fetch_teams(client, season)
    log.info(f"  Found {len(teams)} teams")

    all_fixtures: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()  # (date, team, opponent) for dedup

    for i, team in enumerate(teams):
        team_id = team["id"]
        team_name = _normalize_team(team["name"])
        team_fixtures = 0

        for league_slug in LEAGUE_SLUGS:
            time.sleep(REQUEST_DELAY)
            try:
                fixtures = fetch_team_schedule(client, league_slug, team_id, season)
            except Exception as e:
                log.warning(f"  Failed {team_name} {league_slug}: {e}")
                continue

            for f in fixtures:
                # Dedup: each match appears twice (once per team)
                key = (f["date"], f["team"], f["opponent"])
                if key not in seen:
                    seen.add(key)
                    all_fixtures.append(f)
                    team_fixtures += 1

        log.info(f"  [{i + 1}/{len(teams)}] {team_name}: {team_fixtures} new fixtures")

    # Sort by date
    all_fixtures.sort(key=lambda f: f["date"])
    return all_fixtures


def write_csv(fixtures: list[dict[str, str]], season: int) -> Path:
    """Write fixtures to CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label = SEASONS[season]
    path = DATA_DIR / f"fixtures_{label}.csv"

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(fixtures)

    return path


async def write_db(fixtures: list[dict[str, str]], season: int) -> int:
    """Write fixtures to database via EspnFixtureWriter."""
    from datetime import UTC, datetime

    from odds_core.database import async_session_maker
    from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter

    label = SEASONS[season]
    records = []
    for f in fixtures:
        date_str = f["date"]
        # Parse ISO datetime, ensure UTC
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        records.append(
            {
                "date": dt,
                "team": f["team"],
                "opponent": f["opponent"],
                "competition": f["competition"],
                "match_round": f.get("match_round", ""),
                "home_away": f["home_away"],
                "score_team": f.get("score_team", ""),
                "score_opponent": f.get("score_opponent", ""),
                "status": f.get("status", ""),
                "season": label,
            }
        )

    async with async_session_maker() as session:
        writer = EspnFixtureWriter(session)
        count = await writer.upsert_fixtures(records)
        await session.commit()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ESPN fixture schedules for EPL teams")
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
    total_fixtures = 0

    try:
        for season in seasons:
            label = SEASONS[season]
            log.info(f"[{label}] Fetching fixtures...")
            fixtures = fetch_season(client, season)
            path = write_csv(fixtures, season)
            total_fixtures += len(fixtures)
            log.info(f"[{label}] Wrote {len(fixtures)} fixtures to {path}")

            if not args.skip_db:
                import asyncio

                count = asyncio.run(write_db(fixtures, season))
                log.info(f"[{label}] Upserted {count} fixtures to database")
    finally:
        client.close()

    log.info(f"\nTotal: {total_fixtures} fixtures across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
