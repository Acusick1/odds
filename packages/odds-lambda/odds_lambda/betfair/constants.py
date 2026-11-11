"""Betfair Exchange constants and per-sport config."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from odds_core.sports import SportKey
from odds_core.vocabulary import Market

# Betfair event-type IDs (stable; see developer.betfair.com).
EVENT_TYPE_SOCCER = "1"
EVENT_TYPE_BASEBALL = "7511"
EVENT_TYPE_BASKETBALL = "7522"

# Premier League competition ID — community-stable across seasons.
# Hints are a fallback in case Betfair re-issues the competition_id.
EPL_COMPETITION_ID = "10932509"
EPL_COMPETITION_NAME_HINTS: tuple[str, ...] = ("English Premier League", "Premier League")

# MLB competition ID — Betfair calls it "Major League Baseball".
# Pinning the discovered ID here as the primary path; the name hint is
# kept as a fallback in case Betfair re-issues the competition_id.
MLB_COMPETITION_ID = "11196870"
MLB_COMPETITION_NAME_HINTS: tuple[str, ...] = ("Major League Baseball", "MLB")

# NBA competition ID — Betfair calls it "NBA".
NBA_COMPETITION_ID = "10547864"
NBA_COMPETITION_NAME_HINTS: tuple[str, ...] = ("NBA",)


@dataclass(frozen=True)
class BetfairMarketSpec:
    """A Betfair market type the job collects and the pipeline market it is stored under."""

    market_type_code: str
    market: Market
    # The goal line of a totals market; None for match odds.
    line: Decimal | None = None

    @property
    def runner_count(self) -> int:
        return 3 if self.market is Market.ONE_X_TWO else 2


@dataclass(frozen=True)
class SportBetfairConfig:
    """Per-sport Betfair Exchange ingestion config."""

    sport_key: SportKey
    sport_title: str
    event_type_id: str
    # The match odds market names an event's teams and dates its kick-off.
    match_odds: BetfairMarketSpec
    # Optional: pin a competition_id (preferred). If None, client uses
    # competition_name_hints to discover via list_competitions.
    competition_id: str | None
    # How far ahead a scheduled fixture must be before a run is expected to price it.
    fixture_horizon: timedelta
    competition_name_hints: tuple[str, ...] = ()
    # Markets collected alongside the match odds, stored only for an event whose match odds are.
    other_markets: tuple[BetfairMarketSpec, ...] = ()
    # Betfair event-name separator and ordering. Soccer: "Home v Away".
    # Baseball and basketball: "Away @ Home". ``name_separator`` splits the event name and
    # ``name_order`` says which side comes first in the resulting parts.
    name_separator: str = " v "
    name_order: Literal["home_first", "away_first"] = "home_first"

    @property
    def markets(self) -> tuple[BetfairMarketSpec, ...]:
        return (self.match_odds, *self.other_markets)

    def market_spec(self, market_type_code: str) -> BetfairMarketSpec | None:
        return next((m for m in self.markets if m.market_type_code == market_type_code), None)


def _football_league(
    sport_key: SportKey, sport_title: str, competition_id: str, competition_name: str
) -> SportBetfairConfig:
    """A football league collected like the EPL: match odds and the 2.5 goals line."""
    return SportBetfairConfig(
        sport_key=sport_key,
        sport_title=sport_title,
        event_type_id=EVENT_TYPE_SOCCER,
        match_odds=BetfairMarketSpec("MATCH_ODDS", Market.ONE_X_TWO),
        competition_id=competition_id,
        fixture_horizon=timedelta(hours=72),
        competition_name_hints=(competition_name,),
        other_markets=(BetfairMarketSpec("OVER_UNDER_25", Market.TOTALS, Decimal("2.5")),),
    )


SPORT_CONFIG: dict[SportKey, SportBetfairConfig] = {
    "soccer_epl": SportBetfairConfig(
        sport_key="soccer_epl",
        sport_title="EPL",
        event_type_id=EVENT_TYPE_SOCCER,
        match_odds=BetfairMarketSpec("MATCH_ODDS", Market.ONE_X_TWO),
        competition_id=EPL_COMPETITION_ID,
        fixture_horizon=timedelta(hours=72),
        competition_name_hints=EPL_COMPETITION_NAME_HINTS,
        other_markets=(BetfairMarketSpec("OVER_UNDER_25", Market.TOTALS, Decimal("2.5")),),
    ),
    # Competition ids from listCompetitions (2026-11-10). The hints are Betfair's full names:
    # shorter ones also match other competitions ("Scottish Championship", the women's
    # "French Premiere Ligue").
    "soccer_efl_champ": _football_league(
        "soccer_efl_champ", "Championship", "7129730", "English Sky Bet Championship"
    ),
    "soccer_spain_la_liga": _football_league(
        "soccer_spain_la_liga", "La Liga", "117", "Spanish La Liga"
    ),
    "soccer_italy_serie_a": _football_league(
        "soccer_italy_serie_a", "Serie A", "81", "Italian Serie A"
    ),
    "soccer_germany_bundesliga": _football_league(
        "soccer_germany_bundesliga", "Bundesliga", "59", "German Bundesliga"
    ),
    "soccer_france_ligue_one": _football_league(
        "soccer_france_ligue_one", "Ligue 1", "55", "French Ligue 1"
    ),
    "baseball_mlb": SportBetfairConfig(
        sport_key="baseball_mlb",
        sport_title="MLB",
        event_type_id=EVENT_TYPE_BASEBALL,
        match_odds=BetfairMarketSpec("MATCH_ODDS", Market.H2H),
        competition_id=MLB_COMPETITION_ID,
        fixture_horizon=timedelta(hours=6),
        competition_name_hints=MLB_COMPETITION_NAME_HINTS,
        name_separator=" @ ",
        name_order="away_first",
    ),
    "basketball_nba": SportBetfairConfig(
        sport_key="basketball_nba",
        sport_title="NBA",
        event_type_id=EVENT_TYPE_BASKETBALL,
        match_odds=BetfairMarketSpec("MATCH_ODDS", Market.H2H),
        competition_id=NBA_COMPETITION_ID,
        fixture_horizon=timedelta(hours=6),
        competition_name_hints=NBA_COMPETITION_NAME_HINTS,
        name_separator=" @ ",
        name_order="away_first",
    ),
}
