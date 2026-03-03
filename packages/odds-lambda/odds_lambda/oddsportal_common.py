"""Shared utilities for OddsPortal data ingestion (historical and live)."""

from __future__ import annotations

import re
from datetime import UTC, datetime

DRAW_OUTCOME = "Draw"

# OddsPortal bookmaker name → pipeline key.
BOOKMAKER_KEY_MAP: dict[str, str] = {
    "10bet": "10bet",
    "bet365": "bet365",
    "BetMGM": "betmgm",
    "bwin": "bwin",
    "Betway": "betway",
    "BetVictor": "betvictor",
    "Betfred": "betfred",
    "BetUK": "betuk",
    "Midnite": "midnite",
    "Unibetuk": "unibet_uk",
    "Betano.uk": "betano",
    "AllBritishCasino": "allbritishcasino",
    "Pinnacle": "pinnacle",
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "Caesars": "williamhill_us",
    "PointsBet": "pointsbetus",
    "BetRivers": "betrivers",
    "Unibet": "unibet",
    "Bovada": "bovada",
    "Marathon Bet": "marathonbet",
    "1xBet": "onexbet",
    "Betfair Exchange": "betfair_exchange",
}


def slugify(name: str) -> str:
    """Convert bookmaker name to a lowercase slug key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalize_bookmaker_key(name: str) -> str:
    """Map OddsPortal bookmaker name to pipeline key."""
    return BOOKMAKER_KEY_MAP.get(name, slugify(name))


def decimal_to_american(d: float) -> int:
    """Convert decimal odds to American odds."""
    if d >= 2.0:
        return round((d - 1) * 100)
    elif d > 1.0:
        return round(-100 / (d - 1))
    return -10000


def parse_match_date(date_str: str) -> datetime:
    """Parse OddsPortal date string to UTC datetime.

    Format: '2024-10-04 16:00:00 UTC'
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=UTC)


def hours_to_tier(hours_before: float) -> str:
    """Map hours before game start to fetch tier name."""
    if hours_before < 3:
        return "closing"
    elif hours_before < 12:
        return "pregame"
    elif hours_before < 24:
        return "sharp"
    elif hours_before < 72:
        return "early"
    return "opening"


def team_abbrev(name: str) -> str:
    """Derive a short abbreviation from a team name.

    Single-word names get 3 chars (e.g. "Arsenal" -> "ARS").
    Multi-word names use first 3 chars of first + last word
    (e.g. "Manchester United" -> "MANUNI").
    """
    words = name.split()
    if len(words) == 1:
        return words[0][:3].upper()
    return (words[0][:3] + words[-1][:3]).upper()
