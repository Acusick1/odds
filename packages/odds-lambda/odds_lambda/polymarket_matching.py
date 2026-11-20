"""NBA team names: each canonical team's standard abbreviation, and the 30-team check."""

from __future__ import annotations

from odds_core.league_teams import NBA_ABBREV_MAP, UnknownTeam, league_team

# Canonical team name → standard NBA abbreviation (used by PBPStats, nba_api)
CANONICAL_TO_ABBREV: dict[str, str] = {v: k.upper() for k, v in NBA_ABBREV_MAP.items()}


def canonical_nba_team(name: str) -> str | None:
    """The shared team normalizer's name for ``name`` when it is one of the 30 NBA teams, else
    ``None``: ESPN's NBA Cup placeholders and All-Star teams, or an unknown spelling."""
    try:
        return league_team("basketball_nba", name)
    except UnknownTeam:
        return None
