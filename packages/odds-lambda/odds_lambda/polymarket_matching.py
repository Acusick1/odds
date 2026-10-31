"""NBA team abbreviations: the id lookup from a Polymarket ticker abbreviation to the full name."""

from __future__ import annotations

# Polymarket ticker abbreviation → canonical team name
NBA_ABBREV_MAP: dict[str, str] = {
    "atl": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gsw": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "nop": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "uta": "Utah Jazz",
    "was": "Washington Wizards",
}

# Canonical team name → standard NBA abbreviation (used by PBPStats, nba_api)
CANONICAL_TO_ABBREV: dict[str, str] = {v: k.upper() for k, v in NBA_ABBREV_MAP.items()}
