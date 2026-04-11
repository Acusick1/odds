"""Centralized EPL team name normalization.

Delegates to odds_core.team.normalize_team_name(). Source-specific overrides
(e.g. FPL abbreviations) are handled here as a thin wrapper.
"""

from __future__ import annotations

from odds_core.team import normalize_team_name


def normalize_team(name: str, source: str | None = None) -> str:
    """Normalize a team name to the pipeline canonical form.

    Args:
        name: Raw team name from the data source.
        source: Optional source identifier (e.g. "fpl", "espn", "fbref", "fduk")
                for source-specific overrides. Reserved for future use.
    """
    return normalize_team_name(name)


# Re-export for backward compatibility
__all__ = ["normalize_team"]
