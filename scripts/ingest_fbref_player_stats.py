#!/usr/bin/env python3
"""Parse per-player season stats from locally saved FBref HTML pages.

Reads saved HTML files from data/external/fbref/ and extracts player standard
stats tables. Writes one CSV per season to data/fbref_player_stats/.

FBref is behind Cloudflare and blocks headless browsers, so HTML files must be
saved manually from a real browser. Expected filename pattern:
    FBREF_EPL_{YY}_{YY}.html  (e.g. FBREF_EPL_24_25.html)

Usage:
    # Parse all HTML files in data/external/fbref/
    uv run python scripts/ingest_fbref_player_stats.py

    # Parse a single season
    uv run python scripts/ingest_fbref_player_stats.py --season 2024
"""

from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
from team_names import normalize_team as _normalize_team_central

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_fbref_player_stats")

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data" / "fbref_player_stats"
HTML_DIR = SCRIPT_DIR / "data" / "external" / "fbref"

SEASONS: dict[int, str] = {
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}

# Map season start year to expected HTML filename.
SEASON_HTML_FILES: dict[int, str] = {
    2020: "FBREF_EPL_20_21.html",
    2021: "FBREF_EPL_21_22.html",
    2022: "FBREF_EPL_22_23.html",
    2023: "FBREF_EPL_23_24.html",
    2024: "FBREF_EPL_24_25.html",
    2025: "FBREF_EPL_25_26.html",
}

# Mapping from flattened column names to clean output names. Unique leaf names
# keep their original name (e.g. "MP"), duplicates get group-prefixed
# (e.g. "Performance_Gls" vs "Per 90 Minutes_Gls").
COLUMN_RENAMES: dict[str, str] = {
    "Player": "player",
    "Nation": "nation",
    "Pos": "position",
    "Squad": "team",
    "Age": "age",
    "Born": "born",
    "MP": "games",
    "Starts": "games_starts",
    "Min": "minutes",
    "90s": "nineties",
    "Performance_Gls": "goals",
    "Performance_Ast": "assists",
    "Performance_G+A": "goals_assists",
    "Performance_G-PK": "goals_non_penalty",
    "PK": "penalty_goals",
    "PKatt": "penalty_attempts",
    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
    "Per 90 Minutes_Gls": "goals_per90",
    "Per 90 Minutes_Ast": "assists_per90",
    "Per 90 Minutes_G+A": "goals_assists_per90",
    "Per 90 Minutes_G-PK": "goals_non_penalty_per90",
    "G+A-PK": "goals_assists_non_penalty_per90",
}

# Columns to drop from output (internal FBref fields).
DROP_COLUMNS = {"Rk", "Matches"}


def _normalize_team(fbref_name: str) -> str:
    """Convert FBref team name to pipeline canonical name."""
    return _normalize_team_central(fbref_name)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, prefixing group name for duplicate leaf names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    seen: dict[str, int] = {}
    for _, leaf in df.columns:
        seen[leaf] = seen.get(leaf, 0) + 1

    flat_names: list[str] = []
    for group, leaf in df.columns:
        if leaf == "":
            leaf = group
        # Prefix with group name if the leaf appears more than once
        if seen.get(leaf, 1) > 1 and not group.startswith("Unnamed"):
            flat_names.append(f"{group}_{leaf}")
        else:
            flat_names.append(leaf)

    df = df.copy()
    df.columns = pd.Index(flat_names)
    return df


def _find_standard_stats_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Find the main standard stats table from parsed HTML tables.

    FBref pages contain multiple tables. The standard stats table is the largest
    one containing 'Player' and 'MP' columns after flattening MultiIndex headers.
    """
    for df in tables:
        df = _flatten_columns(df)
        col_set = set(df.columns)
        if "Player" in col_set and "MP" in col_set and "Min" in col_set:
            return df

    return None


def parse_season(html_path: Path, season: int) -> pd.DataFrame:
    """Parse player stats for a single season from a saved FBref HTML file."""
    log.info(f"  Reading {html_path}")
    html = html_path.read_text(encoding="utf-8")
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise ValueError(f"No tables found in {html_path}")

    df = _find_standard_stats_table(tables)
    if df is None:
        raise ValueError(
            f"Could not find standard stats table in {html_path}. "
            f"Found {len(tables)} tables with columns: "
            f"{[list(t.columns[:5]) for t in tables[:5]]}"
        )

    # Rename columns using our mapping
    df = df.rename(columns=COLUMN_RENAMES)

    # Drop internal FBref columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Drop separator/header rows that FBref inserts (Player == "Player")
    if "player" in df.columns:
        df = df[df["player"] != "Player"].copy()

    # Normalize team names
    if "team" in df.columns:
        df["team"] = df["team"].apply(_normalize_team)

    # Add season
    df["season"] = SEASONS[season]

    # Convert numeric columns (everything except identifiers)
    non_numeric = {"season", "player", "nation", "position", "team", "age"}
    for col in df.columns:
        if col not in non_numeric:
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
    parser = argparse.ArgumentParser(
        description="Parse FBref player season stats from saved HTML files"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single season start year (e.g. 2024 for 2024-25). Default: all available.",
    )
    args = parser.parse_args()

    if args.season is not None:
        if args.season not in SEASONS:
            parser.error(f"Unknown season: {args.season}. Available: {sorted(SEASONS)}")
        seasons = [args.season]
    else:
        seasons = sorted(SEASONS.keys())

    total_players = 0

    for season in seasons:
        label = SEASONS[season]
        html_file = HTML_DIR / SEASON_HTML_FILES[season]

        if not html_file.exists():
            log.warning(f"[{label}] HTML file not found: {html_file} — skipping")
            continue

        log.info(f"[{label}] Parsing player stats...")
        try:
            df = parse_season(html_file, season)
            path = write_csv(df, season)
            total_players += len(df)
            log.info(f"[{label}] Wrote {len(df)} player rows to {path}")
        except Exception as e:
            log.error(f"[{label}] Failed: {e}")
            continue

    log.info(f"\nTotal: {total_players} player rows across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
