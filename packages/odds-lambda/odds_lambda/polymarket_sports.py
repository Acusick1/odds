"""Per-sport Polymarket discovery config."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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


def nba_team_name(team: Mapping[str, Any]) -> str:
    """The full name of a Polymarket NBA team, which Polymarket lists by nickname ("Lakers")
    with a ticker abbreviation ("lal") that identifies it."""
    abbreviation = str(team.get("abbreviation", "")).lower()
    return NBA_ABBREV_MAP.get(abbreviation) or normalize_team_name(team["name"])


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
    team_name: Callable[[Mapping[str, Any]], str]
    # US sports list the away team first in a ticker (nba-den-phx is Denver at Phoenix).
    away_first_in_slug: bool
    # Where the backfill starts by default: the first EPL market, and the 2026-27 NBA season opener.
    backfill_since: datetime

    @property
    def sides(self) -> tuple[Side, ...]:
        return ("home", "draw", "away") if self.has_draw else ("home", "away")

    @property
    def match_slug(self) -> Pattern[str]:
        first, second = ("away", "home") if self.away_first_in_slug else ("home", "away")
        return compile(
            rf"^{escape(self.ticker_prefix)}-(?P<{first}>[a-z0-9]+)-(?P<{second}>[a-z0-9]+)-\d{{4}}-\d{{2}}-\d{{2}}$"
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
        away_first_in_slug=True,
        backfill_since=datetime(2026, 10, 21, tzinfo=UTC),
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
        away_first_in_slug=False,
        backfill_since=datetime(2025, 8, 1, tzinfo=UTC),
    ),
}
