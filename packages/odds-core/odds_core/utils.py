"""Shared utility functions for odds pipeline."""

import copy
from typing import Any

from odds_core.team import normalize_team_name


def raw_data_has_market(raw_data: dict[str, Any] | None, market: str) -> bool:
    """Check if raw_data JSON contains data for the given market key.

    Iterates bookmakers -> markets -> key to find a match.
    """
    if not raw_data:
        return False
    for bookmaker in raw_data.get("bookmakers", []):
        for m in bookmaker.get("markets", []):
            if m.get("key") == market:
                return True
    return False


def is_three_way_sport(sport_key: str) -> bool:
    return sport_key.startswith("soccer_")


def relabel_three_way_markets(raw_event: dict[str, Any], sport_key: str) -> dict[str, Any]:
    """Return a copy of ``raw_event`` with soccer three-way markets in pipeline form.

    The Odds API keys the home/draw/away market ``h2h`` and names teams in full
    ("Newcastle United"). The pipeline stores that market as ``1x2`` with the
    canonical team names the events table uses. Two-way sports, and any market
    without a Draw outcome, are returned unchanged. ``raw_event`` is not mutated.
    """
    if not is_three_way_sport(sport_key):
        return raw_event
    result = copy.deepcopy(raw_event)
    for bookmaker in result.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            outcomes = market.get("outcomes", [])
            if market.get("key") == "h2h" and any(o.get("name") == "Draw" for o in outcomes):
                market["key"] = "1x2"
                for outcome in outcomes:
                    if outcome.get("name") != "Draw":
                        outcome["name"] = normalize_team_name(outcome["name"])
    return result
