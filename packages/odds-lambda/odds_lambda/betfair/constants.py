"""Betfair Exchange constants and per-sport config."""

from __future__ import annotations

from dataclasses import dataclass

from odds_core.sports import SportKey

# Betfair event-type IDs (stable; see developer.betfair.com).
EVENT_TYPE_SOCCER = "1"
EVENT_TYPE_BASEBALL = "7511"

# Premier League competition ID — community-stable across seasons.
# Hints are a fallback in case Betfair re-issues the competition_id.
EPL_COMPETITION_ID = "10932509"
EPL_COMPETITION_NAME_HINTS: tuple[str, ...] = ("English Premier League", "Premier League")

# MLB competition ID — Betfair calls it "Major League Baseball".
# Pinning the discovered ID here as the primary path; the name hint is
# kept as a fallback in case Betfair re-issues the competition_id.
MLB_COMPETITION_ID = "11196870"
MLB_COMPETITION_NAME_HINTS: tuple[str, ...] = ("Major League Baseball", "MLB")


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
        competition_name_hints=EPL_COMPETITION_NAME_HINTS,
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
