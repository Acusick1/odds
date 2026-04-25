"""Betfair Exchange constants and per-sport config."""

from __future__ import annotations

from dataclasses import dataclass

from odds_core.sports import SportKey

# Betfair event-type IDs (stable; see developer.betfair.com).
EVENT_TYPE_SOCCER = "1"
EVENT_TYPE_BASEBALL = "7511"

# Premier League competition ID — community-stable across seasons.
EPL_COMPETITION_ID = "10932509"

# MLB competition ID — Betfair calls it "Major League Baseball".
# Pinning the discovered ID here as the primary path; the name hint is
# kept as a fallback in case Betfair re-issues the competition_id.
MLB_COMPETITION_ID = "11196870"
MLB_COMPETITION_NAME_HINTS: tuple[str, ...] = ("Major League Baseball", "MLB")


# Betfair runner-name aliases → pipeline canonical team names.
# Soccer entries supplement odds_core.team alias map; baseball is exhaustive.
BETFAIR_TEAM_ALIASES: dict[str, str] = {
    # EPL — Betfair renders many short forms not in odds_core.team
    "spurs": "Tottenham",
    "wolves": "Wolves",
    "man utd": "Manchester Utd",
    "man city": "Manchester City",
    "newcastle": "Newcastle",
    "leeds": "Leeds",
    "leicester": "Leicester",
    "brighton": "Brighton",
    "bournemouth": "Bournemouth",
    "burnley": "Burnley",
    "ipswich": "Ipswich",
    "nottm forest": "Nottingham",
    "nott'm forest": "Nottingham",
    "sheff utd": "Sheffield Utd",
    "west ham": "West Ham",
    "west brom": "West Brom",
    # MLB — Betfair short forms → canonical "events.home_team" names
    "arizona d'backs": "Arizona Diamondbacks",
    "arizona diamondbacks": "Arizona Diamondbacks",
    "atlanta braves": "Atlanta Braves",
    "baltimore orioles": "Baltimore Orioles",
    "boston red sox": "Boston Red Sox",
    "chicago cubs": "Chicago Cubs",
    "chicago white sox": "Chicago White Sox",
    "cincinnati reds": "Cincinnati Reds",
    "cleveland guardians": "Cleveland Guardians",
    "colorado rockies": "Colorado Rockies",
    "detroit tigers": "Detroit Tigers",
    "houston astros": "Houston Astros",
    "kansas city royals": "Kansas City Royals",
    "la angels": "Los Angeles Angels",
    "los angeles angels": "Los Angeles Angels",
    "la dodgers": "Los Angeles Dodgers",
    "los angeles dodgers": "Los Angeles Dodgers",
    "miami marlins": "Miami Marlins",
    "milwaukee brewers": "Milwaukee Brewers",
    "minnesota twins": "Minnesota Twins",
    "ny mets": "New York Mets",
    "new york mets": "New York Mets",
    "ny yankees": "New York Yankees",
    "new york yankees": "New York Yankees",
    "oakland athletics": "Athletics",
    "athletics": "Athletics",
    "philadelphia phillies": "Philadelphia Phillies",
    "pittsburgh pirates": "Pittsburgh Pirates",
    "san diego padres": "San Diego Padres",
    "san francisco giants": "San Francisco Giants",
    "seattle mariners": "Seattle Mariners",
    "st louis cardinals": "St.Louis Cardinals",
    "st. louis cardinals": "St.Louis Cardinals",
    "tampa bay rays": "Tampa Bay Rays",
    "texas rangers": "Texas Rangers",
    "toronto blue jays": "Toronto Blue Jays",
    "washington nationals": "Washington Nationals",
}


@dataclass(frozen=True)
class SportBetfairConfig:
    """Per-sport Betfair Exchange ingestion config."""

    sport_key: SportKey
    sport_title: str
    event_type_id: str
    market_type_code: str
    # Pipeline market_key written into raw_data.bookmakers[].markets[].key
    market_key: str
    # Optional: pin a competition_id (preferred). If None, client uses
    # competition_name_hints to discover via list_competitions.
    competition_id: str | None
    competition_name_hints: tuple[str, ...] = ()
    # If True, the market is a 3-way h2h with a "draw" outcome.
    has_draw: bool = False
    # Betfair event-name separator and ordering. Soccer: "Home v Away".
    # Baseball: "Away @ Home". The first token in the tuple is the literal
    # separator; the second is "home_first" or "away_first".
    name_separator: str = " v "
    name_order: str = "home_first"


SPORT_CONFIG: dict[SportKey, SportBetfairConfig] = {
    "soccer_epl": SportBetfairConfig(
        sport_key="soccer_epl",
        sport_title="EPL",
        event_type_id=EVENT_TYPE_SOCCER,
        market_type_code="MATCH_ODDS",
        market_key="1x2",
        competition_id=EPL_COMPETITION_ID,
        has_draw=True,
    ),
    "baseball_mlb": SportBetfairConfig(
        sport_key="baseball_mlb",
        sport_title="MLB",
        event_type_id=EVENT_TYPE_BASEBALL,
        market_type_code="MATCH_ODDS",
        market_key="h2h",
        competition_id=MLB_COMPETITION_ID,
        competition_name_hints=MLB_COMPETITION_NAME_HINTS,
        has_draw=False,
        name_separator=" @ ",
        name_order="away_first",
    ),
}
