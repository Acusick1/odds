"""Centralized EPL team name normalization.

Delegates to odds_core.team.normalize_team_name(). Source-specific overrides
(e.g. FPL abbreviations) are handled here as a thin wrapper.
"""

from __future__ import annotations

from odds_core.team import normalize_team_name

# Source-specific aliases (only entries not covered by _TEAM_ALIASES).
_SOURCE_ALIASES: dict[str, dict[str, str]] = {
    "fpl": {
        "Man Utd": "Manchester Utd",
    },
}


def normalize_team(name: str, source: str | None = None) -> str:
    """Normalize a team name to the pipeline canonical form.

    Args:
        name: Raw team name from the data source.
        source: Optional source identifier (e.g. "fpl", "espn", "fbref", "fduk")
                for source-specific overrides.
    """
    if source and source in _SOURCE_ALIASES:
        canonical = _SOURCE_ALIASES[source].get(name)
        if canonical:
            return canonical
    return normalize_team_name(name)


# Re-export for backward compatibility
__all__ = ["normalize_team"]
