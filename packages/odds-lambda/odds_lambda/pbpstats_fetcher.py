"""Fetcher for PBPStats player season statistics API."""

from __future__ import annotations

from dataclasses import dataclass

import requests
import structlog

logger = structlog.get_logger(__name__)

PBPSTATS_BASE_URL = "https://api.pbpstats.com/get-totals/nba"

_SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV", "V"}


@dataclass(slots=True)
class PlayerSeasonRecord:
    """Parsed player season stats ready for database storage."""

    player_id: int
    player_name: str  # "Last, First" format
    team_id: int
    team_abbreviation: str
    season: str
    minutes: float
    games_played: int
    on_off_rtg: float | None
    on_def_rtg: float | None
    usage: float | None
    ts_pct: float | None
    efg_pct: float | None
    assists: int
    turnovers: int
    rebounds: int
    steals: int
    blocks: int
    points: int
    plus_minus: float


def convert_name(name: str) -> str:
    """Convert PBPStats "First Last" to "Last, First" format.

    Handles suffixes: "Michael Porter Jr." -> "Porter Jr., Michael"
    Handles multi-part firsts: "Karl-Anthony Towns" -> "Towns, Karl-Anthony"
    Single-word names returned as-is: "Nene" -> "Nene"
    """
    parts = name.split()
    if len(parts) <= 1:
        return name

    # Require 3+ words for suffix detection: "Michael Porter Jr." not "A Jr."
    if len(parts) >= 3 and parts[-1] in _SUFFIXES:
        last = f"{parts[-2]} {parts[-1]}"
        first = " ".join(parts[:-2])
    else:
        last = parts[-1]
        first = " ".join(parts[:-1])

    return f"{last}, {first}"


def _safe_float(value: object) -> float | None:
    """Convert to float, returning None for missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_player(raw: dict, season: str) -> PlayerSeasonRecord:
    """Parse a single player dict from the PBPStats API response."""
    return PlayerSeasonRecord(
        player_id=int(raw["EntityId"]),
        player_name=convert_name(raw["Name"]),
        team_id=int(raw["TeamId"]),
        team_abbreviation=raw["TeamAbbreviation"],
        season=season,
        minutes=float(raw["Minutes"]),
        games_played=int(raw["GamesPlayed"]),
        on_off_rtg=_safe_float(raw.get("OnOffRtg")),
        on_def_rtg=_safe_float(raw.get("OnDefRtg")),
        usage=_safe_float(raw.get("Usage")),
        ts_pct=_safe_float(raw.get("TsPct")),
        efg_pct=_safe_float(raw.get("EfgPct")),
        assists=int(raw.get("Assists", 0)),
        turnovers=int(raw.get("Turnovers", 0)),
        rebounds=int(raw.get("Rebounds", 0)),
        steals=int(raw.get("Steals", 0)),
        blocks=int(raw.get("Blocks", 0)),
        points=int(raw.get("Points", 0)),
        plus_minus=float(raw.get("PlusMinus", 0)),
    )


def fetch_player_season_stats(season: str) -> list[PlayerSeasonRecord]:
    """Fetch player season stats from PBPStats API (synchronous).

    Args:
        season: Season string e.g. '2024-25'.

    Returns:
        Parsed PlayerSeasonRecord instances (~500 per season).
    """
    params = {
        "Type": "Player",
        "Season": season,
        "SeasonType": "Regular Season",
    }

    response = requests.get(PBPSTATS_BASE_URL, params=params, timeout=90)
    response.raise_for_status()

    data = response.json()
    rows = data.get("multi_row_table_data", [])

    records: list[PlayerSeasonRecord] = []
    for raw in rows:
        try:
            records.append(_parse_player(raw, season))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                "pbpstats_parse_error",
                player=raw.get("Name"),
                season=season,
                error=str(e),
            )

    logger.info("pbpstats_fetched", season=season, count=len(records))
    return records
