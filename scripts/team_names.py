"""Centralized EPL team name normalization.

Delegates to odds_core.team.normalize_team_name().
"""

from __future__ import annotations

from odds_core.team import normalize_team_name


def normalize_team(name: str) -> str:
    """Normalize a team name to the pipeline canonical form."""
    return normalize_team_name(name)
