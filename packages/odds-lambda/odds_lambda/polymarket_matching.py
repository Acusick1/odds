"""NBA team names: the aliases sources use for each canonical name, and NBA abbreviations."""

from __future__ import annotations

# Canonical team name → set of known aliases
TEAM_ALIASES: dict[str, set[str]] = {
    "Atlanta Hawks": {"Atlanta Hawks", "Hawks", "Atlanta"},
    "Boston Celtics": {"Boston Celtics", "Celtics", "Boston"},
    "Brooklyn Nets": {"Brooklyn Nets", "Nets", "Brooklyn"},
    "Charlotte Hornets": {"Charlotte Hornets", "Hornets", "Charlotte"},
    "Chicago Bulls": {"Chicago Bulls", "Bulls", "Chicago"},
    "Cleveland Cavaliers": {"Cleveland Cavaliers", "Cavaliers", "Cavs", "Cleveland"},
    "Dallas Mavericks": {"Dallas Mavericks", "Mavericks", "Mavs", "Dallas"},
    "Denver Nuggets": {"Denver Nuggets", "Nuggets", "Denver"},
    "Detroit Pistons": {"Detroit Pistons", "Pistons", "Detroit"},
    "Golden State Warriors": {"Golden State Warriors", "Warriors", "Golden State", "GSW"},
    "Houston Rockets": {"Houston Rockets", "Rockets", "Houston"},
    "Indiana Pacers": {"Indiana Pacers", "Pacers", "Indiana"},
    "Los Angeles Clippers": {"Los Angeles Clippers", "Clippers", "LA Clippers"},
    "Los Angeles Lakers": {"Los Angeles Lakers", "Lakers", "LA Lakers"},
    "Memphis Grizzlies": {"Memphis Grizzlies", "Grizzlies", "Memphis"},
    "Miami Heat": {"Miami Heat", "Heat", "Miami"},
    "Milwaukee Bucks": {"Milwaukee Bucks", "Bucks", "Milwaukee"},
    "Minnesota Timberwolves": {"Minnesota Timberwolves", "Timberwolves", "Wolves", "Minnesota"},
    "New Orleans Pelicans": {"New Orleans Pelicans", "Pelicans", "New Orleans"},
    "New York Knicks": {"New York Knicks", "Knicks", "New York"},
    "Oklahoma City Thunder": {"Oklahoma City Thunder", "Thunder", "Oklahoma City", "OKC"},
    "Orlando Magic": {"Orlando Magic", "Magic", "Orlando"},
    "Philadelphia 76ers": {"Philadelphia 76ers", "76ers", "Sixers", "Philadelphia"},
    "Phoenix Suns": {"Phoenix Suns", "Suns", "Phoenix"},
    "Portland Trail Blazers": {"Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland"},
    "Sacramento Kings": {"Sacramento Kings", "Kings", "Sacramento"},
    "San Antonio Spurs": {"San Antonio Spurs", "Spurs", "San Antonio"},
    "Toronto Raptors": {"Toronto Raptors", "Raptors", "Toronto"},
    "Utah Jazz": {"Utah Jazz", "Jazz", "Utah"},
    "Washington Wizards": {"Washington Wizards", "Wizards", "Washington"},
}

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

# Pre-computed lowercase alias → canonical name for O(1) lookups
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): canonical for canonical, aliases in TEAM_ALIASES.items() for alias in aliases
}

# Canonical team name → standard NBA abbreviation (used by PBPStats, nba_api)
CANONICAL_TO_ABBREV: dict[str, str] = {v: k.upper() for k, v in NBA_ABBREV_MAP.items()}


def normalize_team(name: str) -> str | None:
    """Normalize a team name to its canonical form.

    Returns the canonical name (e.g. "Dallas Mavericks") or None if unrecognized.
    """
    return _ALIAS_TO_CANONICAL.get(name.strip().lower())
