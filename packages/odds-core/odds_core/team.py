"""Centralized team name normalization.

Maps source-specific team names to pipeline canonical names (short forms
stored in the events table). Whitespace cleanup applied before alias lookup.
Passthrough for names that are already canonical.
"""

from __future__ import annotations

# Pipeline canonical EPL names (from The Odds API / events table):
#   Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Cardiff,
#   Chelsea, Crystal Palace, Everton, Fulham, Huddersfield, Hull, Ipswich,
#   Leeds, Leicester, Liverpool, Luton, Manchester City, Manchester Utd,
#   Middlesbrough, Newcastle, Norwich, Nottingham, Sheffield Utd, Southampton,
#   Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wolves

# Alias -> canonical. Keys are lowercased for case-insensitive lookup.
_TEAM_ALIASES: dict[str, str] = {
    # OddsPortal / football-data.co.uk / ESPN / FBref variants
    "afc bournemouth": "Bournemouth",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "cardiff city": "Cardiff",
    "huddersfield town": "Huddersfield",
    "hull city": "Hull",
    "ipswich town": "Ipswich",
    "leicester city": "Leicester",
    "leeds united": "Leeds",
    "man city": "Manchester City",
    "man united": "Manchester Utd",
    "man utd": "Manchester Utd",
    "manchester united": "Manchester Utd",
    "newcastle united": "Newcastle",
    "newcastle utd": "Newcastle",
    "norwich city": "Norwich",
    "nott'm forest": "Nottingham",
    "nott'ham forest": "Nottingham",
    "nottingham forest": "Nottingham",
    "sheffield united": "Sheffield Utd",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "west bromwich albion": "West Brom",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to pipeline canonical form.

    Collapses whitespace then does case-insensitive alias lookup.
    Returns the canonical name, or the cleaned name unchanged if no alias matches.

    >>> normalize_team_name("Wolves")
    'Wolves'
    >>> normalize_team_name("  manchester   united ")
    'Manchester Utd'
    >>> normalize_team_name("Wolverhampton Wanderers")
    'Wolves'
    >>> normalize_team_name("Arsenal")
    'Arsenal'
    """
    cleaned = " ".join(name.split())
    canonical = _TEAM_ALIASES.get(cleaned.lower())
    if canonical is not None:
        return canonical
    return cleaned
