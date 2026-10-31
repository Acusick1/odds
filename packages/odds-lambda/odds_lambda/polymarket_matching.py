"""NBA team abbreviations: the id lookup from a Polymarket ticker abbreviation to the full name."""

from __future__ import annotations

from odds_core.team import normalize_team_name

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

NBA_TEAMS: frozenset[str] = frozenset(NBA_ABBREV_MAP.values())


def canonical_nba_team(name: str) -> str | None:
    """The shared team normalizer's name for ``name`` when it is one of the 30 NBA teams, else
    ``None``: ESPN's NBA Cup placeholders and All-Star teams, or an unknown spelling."""
    canonical = normalize_team_name(name)
    return canonical if canonical in NBA_TEAMS else None
