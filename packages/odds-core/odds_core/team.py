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
#
# Pipeline canonical MLB names (from The Odds API / events table):
#   Arizona Diamondbacks, Athletics, Atlanta Braves, Baltimore Orioles,
#   Boston Red Sox, Chicago Cubs, Chicago White Sox, Cincinnati Reds,
#   Cleveland Guardians, Colorado Rockies, Detroit Tigers, Houston Astros,
#   Kansas City Royals, Los Angeles Angels, Los Angeles Dodgers, Miami Marlins,
#   Milwaukee Brewers, Minnesota Twins, New York Mets, New York Yankees,
#   Philadelphia Phillies, Pittsburgh Pirates, San Diego Padres,
#   San Francisco Giants, Seattle Mariners, St.Louis Cardinals, Tampa Bay Rays,
#   Texas Rangers, Toronto Blue Jays, Washington Nationals.

# Alias -> canonical. Keys are lowercased for case-insensitive lookup.
# Cross-sport aliases share one map; lowercased keys do not collide between
# the EPL and MLB canonical sets above.
_TEAM_ALIASES: dict[str, str] = {
    # ---- EPL: OddsPortal / football-data.co.uk / ESPN / FBref / Betfair variants
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
    "nottm forest": "Nottingham",
    "nottingham forest": "Nottingham",
    "sheff utd": "Sheffield Utd",
    "sheffield united": "Sheffield Utd",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "west bromwich albion": "West Brom",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
    # ---- MLB: Betfair short forms (and a few common variants)
    "arizona d'backs": "Arizona Diamondbacks",
    "la angels": "Los Angeles Angels",
    "la dodgers": "Los Angeles Dodgers",
    "ny mets": "New York Mets",
    "ny yankees": "New York Yankees",
    "oakland athletics": "Athletics",
    "st louis cardinals": "St.Louis Cardinals",
    "st. louis cardinals": "St.Louis Cardinals",
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
