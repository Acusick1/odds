#!/usr/bin/env python3
"""Ingest FPL availability snapshots from Randdalf/fplcache.

Downloads compressed bootstrap-static JSON snapshots via GitHub raw URLs,
extracts player availability data, and writes one CSV per season to
data/fpl_availability/.

For each gameweek, selects the last snapshot before the gameweek deadline.
Also captures snapshots at ~12h and ~3h before first kickoff when available.

Usage:
    uv run python scripts/ingest_fpl_availability.py
    uv run python scripts/ingest_fpl_availability.py --season 2024-25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import lzma
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from team_names import normalize_team

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_fpl_availability")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fpl_availability"

REPO_RAW_BASE = "https://github.com/Randdalf/fplcache/raw/main/cache"
REPO_API_BASE = "https://api.github.com/repos/Randdalf/fplcache"

POSITION_MAP: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

SEASONS: dict[str, tuple[int, int, int, int]] = {
    # label -> (start_year, start_month, end_year, end_month)
    "2021-22": (2021, 8, 2022, 6),
    "2022-23": (2022, 8, 2023, 6),
    "2023-24": (2023, 8, 2024, 6),
    "2024-25": (2024, 8, 2025, 6),
    "2025-26": (2025, 8, 2026, 6),
}

CSV_COLUMNS = [
    "snapshot_time",
    "gameweek",
    "season",
    "player_code",
    "player_name",
    "team",
    "position",
    "chance_of_playing",
    "status",
]

REQUEST_DELAY = 0.3


def _normalize_team(fpl_name: str) -> str:
    return normalize_team(fpl_name, source="fpl")


def _snapshot_time_from_path(year: int, month: int, day: int, filename: str) -> datetime:
    hhmm = filename.replace(".json.xz", "")
    hour = int(hhmm[:2])
    minute = int(hhmm[2:4])
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _discover_snapshot_paths(
    client: httpx.Client,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[tuple[datetime, str]]:
    """Discover all snapshot file paths via GitHub Git Trees API (recursive).

    Uses a single recursive tree call per year instead of per-day calls,
    reducing API usage from ~300 calls/season to ~3 calls/season.
    """
    snapshots: list[tuple[datetime, str]] = []

    start_dt = datetime(start_year, start_month, 1, tzinfo=UTC)
    end_dt = datetime(end_year, end_month, 28, tzinfo=UTC)

    # Get top-level tree to find cache/ sha
    resp = client.get(f"{REPO_API_BASE}/git/trees/main", timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    cache_sha = None
    for item in resp.json()["tree"]:
        if item["path"] == "cache":
            cache_sha = item["sha"]
            break
    if cache_sha is None:
        raise RuntimeError("cache directory not found in repo")

    # Get year-level trees
    resp = client.get(f"{REPO_API_BASE}/git/trees/{cache_sha}", timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)

    years_needed = set()
    current = start_dt
    while current <= end_dt:
        years_needed.add(current.year)
        current += timedelta(days=32)
        current = current.replace(day=1)

    year_shas: dict[int, str] = {}
    for item in resp.json()["tree"]:
        y = int(item["path"])
        if y in years_needed:
            year_shas[y] = item["sha"]

    # One recursive call per year — returns all month/day/file entries
    for year in sorted(year_shas):
        log.info(f"  Fetching recursive tree for {year}...")
        resp = client.get(
            f"{REPO_API_BASE}/git/trees/{year_shas[year]}",
            params={"recursive": "1"},
            timeout=60,
        )
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)

        for item in resp.json()["tree"]:
            if item["type"] != "blob" or not item["path"].endswith(".json.xz"):
                continue
            # path format: "MM/DD/HHMM.json.xz"
            parts = item["path"].split("/")
            if len(parts) != 3:
                continue
            try:
                month = int(parts[0])
                day = int(parts[1])
            except ValueError:
                continue

            dt_check = datetime(year, month, 1, tzinfo=UTC)
            if dt_check < start_dt or dt_check > end_dt:
                continue

            snap_time = _snapshot_time_from_path(year, month, day, parts[2])
            url = f"{REPO_RAW_BASE}/{year}/{parts[0]}/{parts[1]}/{parts[2]}"
            snapshots.append((snap_time, url))

    snapshots.sort()
    log.info(f"  Discovered {len(snapshots)} snapshots using {2 + len(year_shas)} API calls")
    return snapshots


def _download_snapshot(client: httpx.Client, url: str) -> dict[str, Any]:
    for attempt in range(3):
        try:
            resp = client.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            decompressed = lzma.decompress(resp.content)
            return json.loads(decompressed)
        except (httpx.HTTPError, lzma.LZMAError) as e:
            if attempt == 2:
                raise
            log.warning(f"  Retry {attempt + 1} for {url}: {e}")
    raise RuntimeError("unreachable")


def _select_snapshots_for_gameweeks(
    snapshot_index: list[tuple[datetime, str]],
    gameweek_deadlines: dict[int, datetime],
) -> dict[int, list[tuple[datetime, str]]]:
    """For each gameweek, select all snapshots from 48h before the deadline.

    Multi-match gameweeks span several days (e.g. Saturday + Monday).
    Including all snapshots in the 48h window means later matches get
    more up-to-date availability data when compute_fpl_disruption_features
    does per-event matching.

    Returns dict of gameweek -> list of (snapshot_time, url) to download.
    """
    selected: dict[int, list[tuple[datetime, str]]] = {}

    for gw, deadline in sorted(gameweek_deadlines.items()):
        cutoff = deadline - timedelta(hours=48)
        candidates = [(t, u) for t, u in snapshot_index if cutoff <= t < deadline]
        if not candidates:
            # Fall back to last snapshot before deadline if nothing in 48h window
            all_before = [(t, u) for t, u in snapshot_index if t < deadline]
            if all_before:
                candidates = [all_before[-1]]

        if candidates:
            selected[gw] = candidates

    return selected


def _extract_players(
    data: dict[str, Any],
    snapshot_time: datetime,
    gameweek: int,
    season: str,
) -> list[dict[str, Any]]:
    team_map: dict[int, str] = {}
    for team in data.get("teams", []):
        team_map[team["id"]] = _normalize_team(team["name"])

    rows: list[dict[str, Any]] = []
    for el in data.get("elements", []):
        chance = el.get("chance_of_playing_this_round")
        if chance is None:
            chance = 100.0
        else:
            chance = float(chance)

        team_id = el.get("team", 0)
        team_name = team_map.get(team_id, f"unknown_{team_id}")
        position = POSITION_MAP.get(el.get("element_type", 0), "UNK")

        rows.append(
            {
                "snapshot_time": snapshot_time.isoformat(),
                "gameweek": gameweek,
                "season": season,
                "player_code": el["code"],
                "player_name": el.get("web_name", ""),
                "team": team_name,
                "position": position,
                "chance_of_playing": chance,
                "status": el.get("status", ""),
            }
        )

    return rows


def _extract_gameweek_deadlines(data: dict[str, Any]) -> dict[int, datetime]:
    deadlines: dict[int, datetime] = {}
    for event in data.get("events", []):
        gw_id = event.get("id")
        deadline_str = event.get("deadline_time")
        if gw_id and deadline_str:
            dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            deadlines[gw_id] = dt
    return deadlines


def ingest_season(
    client: httpx.Client,
    season_label: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[dict[str, Any]]:
    log.info(f"[{season_label}] Discovering snapshots...")
    snapshot_index = _discover_snapshot_paths(client, start_year, start_month, end_year, end_month)
    log.info(f"[{season_label}] Found {len(snapshot_index)} snapshots")

    if not snapshot_index:
        return []

    # Download one snapshot to get gameweek deadlines
    log.info(f"[{season_label}] Downloading reference snapshot for gameweek deadlines...")
    # Use the latest snapshot to get the most complete gameweek info
    ref_data = _download_snapshot(client, snapshot_index[-1][1])
    deadlines = _extract_gameweek_deadlines(ref_data)
    log.info(f"[{season_label}] Found {len(deadlines)} gameweeks")

    # Select snapshots per gameweek
    selected = _select_snapshots_for_gameweeks(snapshot_index, deadlines)
    log.info(f"[{season_label}] Selected snapshots for {len(selected)} gameweeks")

    # Download and extract
    all_rows: list[dict[str, Any]] = []
    for gw in sorted(selected):
        for snap_time, url in selected[gw]:
            log.info(f"[{season_label}] GW{gw}: downloading {snap_time.isoformat()}")
            try:
                data = _download_snapshot(client, url)
                rows = _extract_players(data, snap_time, gw, season_label)
                all_rows.extend(rows)
                log.info(f"[{season_label}] GW{gw}: {len(rows)} players")
            except Exception as e:
                log.warning(f"[{season_label}] GW{gw}: failed to download: {e}")

    return all_rows


def write_csv(rows: list[dict[str, Any]], season_label: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"fpl_availability_{season_label}.csv"

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FPL availability snapshots from fplcache")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season label (e.g. 2024-25). Default: all seasons.",
    )
    args = parser.parse_args()

    if args.season is not None:
        if args.season not in SEASONS:
            parser.error(f"Unknown season: {args.season}. Available: {sorted(SEASONS)}")
        season_list = [args.season]
    else:
        season_list = sorted(SEASONS.keys())

    headers = {"Accept": "application/vnd.github.v3+json"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
        log.info("Using authenticated GitHub API requests (5,000 req/hour)")
    else:
        log.warning("No GITHUB_TOKEN set — unauthenticated rate limit is 60 req/hour")

    client = httpx.Client(
        headers=headers,
        follow_redirects=True,
    )
    total_rows = 0

    try:
        for season_label in season_list:
            start_year, start_month, end_year, end_month = SEASONS[season_label]
            rows = ingest_season(client, season_label, start_year, start_month, end_year, end_month)
            if rows:
                path = write_csv(rows, season_label)
                total_rows += len(rows)
                log.info(f"[{season_label}] Wrote {len(rows)} rows to {path}")
            else:
                log.warning(f"[{season_label}] No data extracted")
    finally:
        client.close()

    log.info(f"Total: {total_rows} player rows across {len(season_list)} seasons")


if __name__ == "__main__":
    main()
