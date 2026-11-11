"""Centralized team name normalization.

Maps source-specific team names to pipeline canonical names (short forms
stored in the events table). Whitespace cleanup applied before alias lookup.
Passthrough for names that are already canonical.
"""

from __future__ import annotations

# Pipeline canonical EPL names (from The Odds API / events table):
#   Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Cardiff,
#   Chelsea, Coventry, Crystal Palace, Everton, Fulham, Huddersfield, Hull, Ipswich,
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
#
# Pipeline canonical names of the other football leagues are The Odds API's spellings
# ("Atlético Madrid", "Borussia Monchengladbach"), except that a club with an EPL canonical
# name keeps it in the Championship. Aliases cover the football-data.co.uk and OddsPortal
# spellings of every club in those leagues' 2024-26 seasons, and the Odds API's own variants
# ("Girona FC" in its second division). A club promoted since (no Odds API history) takes the
# OddsPortal spelling ("Bolton", "Schalke"). ESPN's and Betfair's spellings of every club in the
# 2026-27 seasons are aliases too.
#
# Pipeline canonical NBA names are The Odds API's full names ("Boston Celtics"); the
# 30 are the values of odds_lambda.polymarket_matching.NBA_ABBREV_MAP. OddsPortal and
# Betfair already use them, so only the "LA" spellings of the two Los Angeles
# teams need an alias. A bare nickname ("Wolves") is not an alias here: this map is
# shared across sports and "Wolves" is the canonical EPL name.

# Alias -> canonical. Keys are lowercased for case-insensitive lookup.
# Cross-sport aliases share one map; lowercased keys do not collide between
# the canonical sets above.
_TEAM_ALIASES: dict[str, str] = {
    # ---- EPL: OddsPortal / football-data.co.uk / ESPN / FBref / Betfair variants
    "afc bournemouth": "Bournemouth",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "cardiff city": "Cardiff",
    "coventry city": "Coventry",
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
    # ---- Championship: clubs without an EPL canonical name above
    "birmingham": "Birmingham City",
    "blackburn": "Blackburn Rovers",
    "charlton": "Charlton Athletic",
    "derby": "Derby County",
    "oxford": "Oxford United",
    "oxford utd": "Oxford United",
    "plymouth": "Plymouth Argyle",
    "preston": "Preston North End",
    "qpr": "Queens Park Rangers",
    "sheffield weds": "Sheffield Wednesday",
    "sheffield wed": "Sheffield Wednesday",
    "wrexham": "Wrexham AFC",
    "bolton wanderers": "Bolton",
    "lincoln city": "Lincoln",
    # ---- La Liga
    "alaves": "Alavés",
    "ath bilbao": "Athletic Bilbao",
    "ath madrid": "Atlético Madrid",
    "atl. madrid": "Atlético Madrid",
    "betis": "Real Betis",
    "celta": "Celta Vigo",
    "elche": "Elche CF",
    "espanol": "Espanyol",
    "girona fc": "Girona",
    "leganes": "Leganés",
    "osasuna": "CA Osasuna",
    "sociedad": "Real Sociedad",
    "valladolid": "Real Valladolid",
    "real valladolid cf": "Real Valladolid",
    "vallecano": "Rayo Vallecano",
    "athletic club": "Athletic Bilbao",
    "dep. a coruna": "Deportivo La Coruña",
    "deportivo": "Deportivo La Coruña",
    "malaga": "Málaga",
    # ---- Serie A
    "atalanta": "Atalanta BC",
    "inter": "Inter Milan",
    "milan": "AC Milan",
    "roma": "AS Roma",
    "verona": "Hellas Verona",
    "internazionale": "Inter Milan",
    # ---- Bundesliga
    "bochum": "VfL Bochum",
    "dortmund": "Borussia Dortmund",
    "ein frankfurt": "Eintracht Frankfurt",
    "fc koln": "1. FC Köln",
    "freiburg": "SC Freiburg",
    "hamburg": "Hamburger SV",
    "heidenheim": "1. FC Heidenheim",
    "hoffenheim": "TSG Hoffenheim",
    "leverkusen": "Bayer Leverkusen",
    "m'gladbach": "Borussia Monchengladbach",
    "b. monchengladbach": "Borussia Monchengladbach",
    "mainz": "FSV Mainz 05",
    "st pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "stuttgart": "VfB Stuttgart",
    "wolfsburg": "VfL Wolfsburg",
    "1. fc union berlin": "Union Berlin",
    "borussia mönchengladbach": "Borussia Monchengladbach",
    "fc augsburg": "Augsburg",
    "fc cologne": "1. FC Köln",
    "hamburg sv": "Hamburger SV",
    "mgladbach": "Borussia Monchengladbach",
    "sc paderborn 07": "Paderborn",
    "schalke 04": "Schalke",
    "sv elversberg": "Elversberg",
    # ---- Ligue 1
    "lens": "RC Lens",
    "monaco": "AS Monaco",
    "paris sg": "Paris Saint Germain",
    "psg": "Paris Saint Germain",
    "reims": "Stade de Reims",
    "st etienne": "Saint Etienne",
    "aj auxerre": "Auxerre",
    "estac troyes": "Troyes",
    "le havre ac": "Le Havre",
    "paris saint-germain": "Paris Saint Germain",
    "paris st-g": "Paris Saint Germain",
    "stade rennais": "Rennes",
    # ---- MLB: Betfair short forms (and a few common variants)
    "arizona d'backs": "Arizona Diamondbacks",
    "la angels": "Los Angeles Angels",
    "la dodgers": "Los Angeles Dodgers",
    "ny mets": "New York Mets",
    "ny yankees": "New York Yankees",
    "oakland athletics": "Athletics",
    "st louis cardinals": "St.Louis Cardinals",
    "st. louis cardinals": "St.Louis Cardinals",
    # ---- NBA: "LA" spellings of the Los Angeles teams
    "la clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
    "l.a. clippers": "Los Angeles Clippers",
    "l.a. lakers": "Los Angeles Lakers",
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
