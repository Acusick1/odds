"""Centralized team name normalization.

Maps source-specific team names to pipeline canonical names (Odds API forms
stored in the events table). Whitespace/case cleanup applied before alias
lookup. Passthrough for names that are already canonical.
"""

from __future__ import annotations

# Pipeline canonical EPL names (from The Odds API / events table):
#   Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Cardiff,
#   Chelsea, Crystal Palace, Everton, Fulham, Huddersfield, Hull, Ipswich,
#   Leeds, Leicester, Liverpool, Luton, Manchester City, Manchester Utd,
#   Middlesbrough, Newcastle, Norwich, Nottingham, Sheffield Utd, Southampton,
#   Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wolves

# Alias -> canonical. Title-cased before lookup so each variant needs one entry.
_TEAM_ALIASES: dict[str, str] = {
    # OddsPortal / football-data.co.uk / ESPN / FBref variants
    "Afc Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton And Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Man Utd": "Manchester Utd",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'M Forest": "Nottingham",
    "Nott'Ham Forest": "Nottingham",
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


def normalize_team_name(name: str) -> str:
    """Normalize a team name to pipeline canonical form.

    Applies whitespace/case cleanup then explicit alias lookup.
    Returns the canonical name, or the cleaned name if no alias matches.

    >>> normalize_team_name("Wolves")
    'Wolves'
    >>> normalize_team_name("  manchester   united ")
    'Manchester Utd'
    >>> normalize_team_name("Wolverhampton Wanderers")
    'Wolves'
    >>> normalize_team_name("Arsenal")
    'Arsenal'
    """
    # Collapse whitespace and strip
    cleaned = " ".join(name.split())
    # Title-case for consistent lookup
    titled = cleaned.title()
    return _TEAM_ALIASES.get(titled, titled)
