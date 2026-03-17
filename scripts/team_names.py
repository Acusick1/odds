"""Centralized EPL team name normalization.

Maps source-specific team names to pipeline canonical names (as stored in the
events table, sourced from The Odds API). Each data source has its own alias
dict; normalize_team() accepts an optional source parameter for source-specific
lookups, then falls back to the shared aliases.
"""

from __future__ import annotations

# Pipeline canonical names (34 teams across all EPL seasons in the events table):
#   Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Cardiff,
#   Chelsea, Crystal Palace, Everton, Fulham, Huddersfield, Hull, Ipswich,
#   Leeds, Leicester, Liverpool, Luton, Manchester City, Manchester Utd,
#   Middlesbrough, Newcastle, Norwich, Nottingham, Sheffield Utd, Southampton,
#   Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wolves

# Shared aliases that appear across multiple sources.
_COMMON_ALIASES: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'm Forest": "Nottingham",
    "Nott'ham Forest": "Nottingham",
    "Nottingham Forest": "Nottingham",
    "Sheffield United": "Sheffield Utd",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# Source-specific aliases (only entries not covered by _COMMON_ALIASES).
_SOURCE_ALIASES: dict[str, dict[str, str]] = {
    "fpl": {
        "Man Utd": "Manchester Utd",
    },
}


def normalize_team(name: str, source: str | None = None) -> str:
    """Normalize a team name to the pipeline canonical form.

    Args:
        name: Raw team name from the data source.
        source: Optional source identifier (e.g. "fpl", "espn", "fbref", "fduk")
                for source-specific overrides.
    """
    if source and source in _SOURCE_ALIASES:
        canonical = _SOURCE_ALIASES[source].get(name)
        if canonical:
            return canonical
    return _COMMON_ALIASES.get(name, name)
