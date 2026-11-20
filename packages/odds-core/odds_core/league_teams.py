"""Each league's canonical team names: the only names an event of the league may hold.

A league's list holds the canonical name (``odds_core.team.normalize_team_name``) of every
team the league's stored events name, across every season stored. A name a source lists
normalises to one of them or the listing is refused (``league_team``), so one team cannot be
stored under two spellings. A promoted club, or a season further back, needs its name added
here, and any new source spelling an alias in ``odds_core.team``, before its events are stored.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from odds_core.team import normalize_team_name

# NBA ticker abbreviation (Polymarket's, PBPStats') → canonical team name
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

# Keyed by every ``odds_core.sports.SportKey``.
LEAGUE_TEAMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "soccer_epl": frozenset(
            {
                "Arsenal",
                "Aston Villa",
                "Bournemouth",
                "Brentford",
                "Brighton",
                "Burnley",
                "Cardiff",
                "Chelsea",
                "Coventry",
                "Crystal Palace",
                "Everton",
                "Fulham",
                "Huddersfield",
                "Hull",
                "Ipswich",
                "Leeds",
                "Leicester",
                "Liverpool",
                "Luton",
                "Manchester City",
                "Manchester Utd",
                "Middlesbrough",
                "Newcastle",
                "Norwich",
                "Nottingham",
                "Sheffield Utd",
                "Southampton",
                "Stoke",
                "Sunderland",
                "Swansea",
                "Tottenham",
                "Watford",
                "West Brom",
                "West Ham",
                "Wolves",
            }
        ),
        "soccer_efl_champ": frozenset(
            {
                "Birmingham City",
                "Blackburn Rovers",
                "Bolton",
                "Bristol City",
                "Burnley",
                "Cardiff",
                "Charlton Athletic",
                "Coventry",
                "Derby County",
                "Hull",
                "Ipswich",
                "Leeds",
                "Leicester",
                "Lincoln",
                "Luton",
                "Middlesbrough",
                "Millwall",
                "Norwich",
                "Oxford United",
                "Plymouth Argyle",
                "Portsmouth",
                "Preston North End",
                "Queens Park Rangers",
                "Sheffield Utd",
                "Sheffield Wednesday",
                "Southampton",
                "Stoke",
                "Sunderland",
                "Swansea",
                "Watford",
                "West Brom",
                "West Ham",
                "Wolves",
                "Wrexham AFC",
            }
        ),
        "soccer_spain_la_liga": frozenset(
            {
                "Alavés",
                "Athletic Bilbao",
                "Atlético Madrid",
                "Barcelona",
                "CA Osasuna",
                "Celta Vigo",
                "Deportivo La Coruña",
                "Elche CF",
                "Espanyol",
                "Getafe",
                "Girona",
                "Las Palmas",
                "Leganés",
                "Levante",
                "Mallorca",
                "Málaga",
                "Oviedo",
                "Racing Santander",
                "Rayo Vallecano",
                "Real Betis",
                "Real Madrid",
                "Real Sociedad",
                "Real Valladolid",
                "Sevilla",
                "Valencia",
                "Villarreal",
            }
        ),
        "soccer_italy_serie_a": frozenset(
            {
                "AC Milan",
                "AS Roma",
                "Atalanta BC",
                "Bologna",
                "Cagliari",
                "Como",
                "Cremonese",
                "Empoli",
                "Fiorentina",
                "Frosinone",
                "Genoa",
                "Hellas Verona",
                "Inter Milan",
                "Juventus",
                "Lazio",
                "Lecce",
                "Monza",
                "Napoli",
                "Parma",
                "Pisa",
                "Sassuolo",
                "Torino",
                "Udinese",
                "Venezia",
            }
        ),
        "soccer_germany_bundesliga": frozenset(
            {
                "1. FC Heidenheim",
                "1. FC Köln",
                "Augsburg",
                "Bayer Leverkusen",
                "Bayern Munich",
                "Borussia Dortmund",
                "Borussia Monchengladbach",
                "Eintracht Frankfurt",
                "Elversberg",
                "FC St. Pauli",
                "FSV Mainz 05",
                "Hamburger SV",
                "Holstein Kiel",
                "Paderborn",
                "RB Leipzig",
                "SC Freiburg",
                "Schalke",
                "TSG Hoffenheim",
                "Union Berlin",
                "VfB Stuttgart",
                "VfL Bochum",
                "VfL Wolfsburg",
                "Werder Bremen",
            }
        ),
        "soccer_france_ligue_one": frozenset(
            {
                "Angers",
                "AS Monaco",
                "Auxerre",
                "Brest",
                "Le Havre",
                "Le Mans",
                "Lille",
                "Lorient",
                "Lyon",
                "Marseille",
                "Metz",
                "Montpellier",
                "Nantes",
                "Nice",
                "Paris FC",
                "Paris Saint Germain",
                "RC Lens",
                "Rennes",
                "Saint Etienne",
                "Stade de Reims",
                "Strasbourg",
                "Toulouse",
                "Troyes",
            }
        ),
        "baseball_mlb": frozenset(
            {
                "American League",
                "Arizona Diamondbacks",
                "Athletics",
                "Atlanta Braves",
                "Baltimore Orioles",
                "Boston Red Sox",
                "Chicago Cubs",
                "Chicago White Sox",
                "Cincinnati Reds",
                "Cleveland Guardians",
                "Colorado Rockies",
                "Detroit Tigers",
                "Houston Astros",
                "Kansas City Royals",
                "Los Angeles Angels",
                "Los Angeles Dodgers",
                "Miami Marlins",
                "Milwaukee Brewers",
                "Minnesota Twins",
                "National League",
                "New York Mets",
                "New York Yankees",
                "Philadelphia Phillies",
                "Pittsburgh Pirates",
                "San Diego Padres",
                "San Francisco Giants",
                "Seattle Mariners",
                "St.Louis Cardinals",
                "Tampa Bay Rays",
                "Texas Rangers",
                "Toronto Blue Jays",
                "Washington Nationals",
            }
        ),
        "basketball_nba": frozenset(NBA_ABBREV_MAP.values()),
    }
)


class UnknownTeam(ValueError):
    """A team name that normalises to none of the league's canonical teams."""

    def __init__(self, sport_key: str, name: str) -> None:
        self.sport_key = sport_key
        self.name = name
        if sport_key not in LEAGUE_TEAMS:
            reason = "a league with no canonical team list"
        else:
            reason = f"normalises to {normalize_team_name(name)!r}, not a canonical team"
        super().__init__(f"{sport_key}: team {name!r} refused, {reason}")


def league_team(sport_key: str, name: str) -> str:
    """The canonical name of ``name`` in the league ``sport_key``.

    Raises:
        UnknownTeam: when ``name`` normalises to none of the league's ``LEAGUE_TEAMS``.
    """
    canonical = normalize_team_name(name)
    if canonical not in LEAGUE_TEAMS.get(sport_key, frozenset()):
        raise UnknownTeam(sport_key, name)
    return canonical
