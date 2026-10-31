"""Per-sport Polymarket discovery config."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from re import Pattern, compile, escape
from typing import Any, Literal

from odds_core.sports import SportKey
from odds_core.team import normalize_team_name

from odds_lambda.polymarket_matching import NBA_ABBREV_MAP

Side = Literal["home", "draw", "away"]

_CLUB_SUFFIX = compile(r"\s+A?FC$")


def epl_team_name(team: Mapping[str, Any]) -> str:
    """The events-table name of a Polymarket EPL team: no trailing FC or AFC."""
    return normalize_team_name(_CLUB_SUFFIX.sub("", team["name"]))


def nba_team_name(team: Mapping[str, Any]) -> str | None:
    """The full name of a Polymarket NBA team, identified by its ticker abbreviation ("lal"); the
    nickname ("Lakers", "Spurs") is ambiguous across sports, so a team with an unknown
    abbreviation has no name and its match is dropped."""
    return NBA_ABBREV_MAP.get(str(team.get("abbreviation", "")).lower())


@dataclass(frozen=True)
class PolymarketSportConfig:
    sport_key: SportKey
    sport_title: str
    series_id: str
    tag_id: str | None
    ticker_prefix: str
    has_draw: bool
    market_key: str
    # The events-table name of a team entry of a Gamma event.
    team_name: Callable[[Mapping[str, Any]], str | None]

    @property
    def match_slug(self) -> Pattern[str]:
        return compile(
            rf"^{escape(self.ticker_prefix)}-(?P<first>[a-z0-9]+)-(?P<second>[a-z0-9]+)-\d{{4}}-\d{{2}}-\d{{2}}$"
        )


POLYMARKET_SPORTS: dict[SportKey, PolymarketSportConfig] = {
    "basketball_nba": PolymarketSportConfig(
        sport_key="basketball_nba",
        sport_title="NBA",
        series_id="10345",
        tag_id="100639",
        ticker_prefix="nba",
        has_draw=False,
        market_key="h2h",
        team_name=nba_team_name,
    ),
    "soccer_epl": PolymarketSportConfig(
        sport_key="soccer_epl",
        sport_title="EPL",
        series_id="10188",
        tag_id=None,
        ticker_prefix="epl",
        has_draw=True,
        market_key="1x2",
        team_name=epl_team_name,
    ),
}
