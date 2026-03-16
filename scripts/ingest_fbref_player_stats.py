#!/usr/bin/env python3
"""Fetch per-player season stats from FBref via direct HTML scraping.

Downloads standard player stats for EPL seasons (2020-21 through 2025-26) by
fetching FBref HTML pages and parsing tables with pandas.read_html(). Writes
one CSV per season to data/fbref_player_stats/.

FBref aggressively rate-limits — a 5-second delay is used between requests.

Usage:
    # Fetch all seasons
    uv run python scripts/ingest_fbref_player_stats.py

    # Fetch a single season
    uv run python scripts/ingest_fbref_player_stats.py --season 2024
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_fbref_player_stats")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fbref_player_stats"

# FBref Premier League competition ID is 9.
FBREF_BASE = "https://fbref.com/en/comps/9"

SEASONS: dict[int, str] = {
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}

SEASON_URL_SLUGS: dict[int, str] = {
    2020: "2020-2021",
    2021: "2021-2022",
    2022: "2022-2023",
    2023: "2023-2024",
    2024: "2024-2025",
    2025: "2025-2026",
}

# FBref team name -> pipeline canonical name.
FBREF_TEAM_NAME_MAP: dict[str, str] = {
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Manchester Utd": "Manchester Utd",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'ham Forest": "Nottingham",
    "Nottingham Forest": "Nottingham",
    "Sheffield Utd": "Sheffield Utd",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Tottenham": "Tottenham",
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
}

REQUEST_DELAY = 5.0  # FBref rate-limits aggressively

OUTPUT_COLUMNS = [
    "season",
    "player",
    "team",
    "games",
    "games_starts",
    "minutes",
    "goals",
    "assists",
    "xg",
    "xa",
]

# Mapping from FBref HTML table headers to output column names.
COLUMN_RENAMES: dict[str, str] = {
    "Player": "player",
    "Squad": "team",
    "MP": "games",
    "Starts": "games_starts",
    "Min": "minutes",
    "Gls": "goals",
    "Ast": "assists",
    "xG": "xg",
    "xAG": "xa",
}


def _normalize_team(fbref_name: str) -> str:
    """Convert FBref team name to pipeline canonical name."""
    return FBREF_TEAM_NAME_MAP.get(fbref_name, fbref_name)


def _fetch_html(client: httpx.Client, url: str) -> str:
    """Fetch HTML from FBref with retry."""
    for attempt in range(3):
        try:
            resp = client.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt == 2:
                raise
            wait = REQUEST_DELAY * (attempt + 1)
            log.warning(f"  Retry {attempt + 1} for {url}: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _find_standard_stats_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Find the main standard stats table from parsed HTML tables.

    FBref pages contain multiple tables. The standard stats table is the largest
    one containing 'Player' and 'MP' columns after flattening MultiIndex headers.
    """
    for df in tables:
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[-1] if col[-1] != "" else col[-2] for col in df.columns]

        col_set = set(df.columns)
        if "Player" in col_set and "MP" in col_set:
            return df

    return None


def fetch_season(client: httpx.Client, season: int) -> pd.DataFrame:
    """Fetch player stats for a single season from FBref."""
    slug = SEASON_URL_SLUGS[season]
    url = f"{FBREF_BASE}/{slug}/stats/{slug}-Premier-League-Stats"
    log.info(f"  Fetching {url}")

    html = _fetch_html(client, url)
    tables = pd.read_html(html)

    if not tables:
        raise ValueError(f"No tables found on FBref page for {SEASONS[season]}")

    df = _find_standard_stats_table(tables)
    if df is None:
        raise ValueError(
            f"Could not find standard stats table for {SEASONS[season]}. "
            f"Found {len(tables)} tables with columns: "
            f"{[list(t.columns[:5]) for t in tables[:5]]}"
        )

    # Flatten MultiIndex columns if not already done
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[-1] if col[-1] != "" else col[-2] for col in df.columns]

    # Rename columns to output names
    df = df.rename(columns=COLUMN_RENAMES)

    # Drop separator/header rows that FBref inserts (Player == "Player")
    if "player" in df.columns:
        df = df[df["player"] != "Player"].copy()

    # Normalize team names
    if "team" in df.columns:
        df["team"] = df["team"].apply(_normalize_team)

    # Add season
    df["season"] = SEASONS[season]

    # Ensure all output columns exist (fill missing with NaN)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            log.warning(f"  Missing column (will be NaN): {col}")
            df[col] = None

    df = df[OUTPUT_COLUMNS].copy()

    # Convert numeric columns
    for col in ["games", "games_starts", "minutes", "goals", "assists", "xg", "xa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no player name (residual header/footer rows)
    df = df.dropna(subset=["player"])

    return df


def write_csv(df: pd.DataFrame, season: int) -> Path:
    """Write player stats to CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label = SEASONS[season]
    path = DATA_DIR / f"player_stats_{label}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FBref player season stats for EPL")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single season start year (e.g. 2024 for 2024-25). Default: all seasons.",
    )
    args = parser.parse_args()

    if args.season is not None:
        if args.season not in SEASONS:
            parser.error(f"Unknown season: {args.season}. Available: {sorted(SEASONS)}")
        seasons = [args.season]
    else:
        seasons = sorted(SEASONS.keys())

    client = httpx.Client(
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    )
    total_players = 0

    try:
        for i, season in enumerate(seasons):
            label = SEASONS[season]
            log.info(f"[{label}] Fetching player stats...")
            try:
                df = fetch_season(client, season)
                path = write_csv(df, season)
                total_players += len(df)
                log.info(f"[{label}] Wrote {len(df)} player rows to {path}")
            except Exception as e:
                log.error(f"[{label}] Failed: {e}")
                continue

            # Rate limit between seasons (not after the last one)
            if i < len(seasons) - 1:
                log.info(f"  Waiting {REQUEST_DELAY}s for rate limit...")
                time.sleep(REQUEST_DELAY)
    finally:
        client.close()

    log.info(f"\nTotal: {total_players} player rows across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
