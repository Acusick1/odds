"""EPL lineup-delta feature extraction for CLV prediction.

Computes lineup stability metrics by comparing each team's current starting XI
to their previous match XI. Weights dropped players by cumulative starts over
a sliding 38-match window (point-in-time, no future data).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    import pandas as pd
    from odds_core.models import Event

__all__ = [
    "EplLineupFeatures",
    "extract_epl_lineup_features",
    "build_lineup_cache",
    "LineupCache",
]

logger = structlog.get_logger()

_STARTS_WINDOW = 38


@dataclass
class EplLineupFeatures:
    home_xi_changes: float | None = None
    away_xi_changes: float | None = None
    diff_xi_changes: float | None = None
    home_cumulative_starts_lost: float | None = None
    away_cumulative_starts_lost: float | None = None
    diff_cumulative_starts_lost: float | None = None

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                getattr(self, f.name) if getattr(self, f.name) is not None else np.nan
                for f in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class _TeamMatchXI:
    match_date: str
    player_ids: set[str]


LineupCache = dict[str, list[_TeamMatchXI]]


def build_lineup_cache(lineup_df: pd.DataFrame) -> LineupCache:
    """Build a per-team chronological cache of starting XIs from the lineup DataFrame.

    The DataFrame must already be filtered to starters and have columns:
    ``team``, ``datetime``, ``match_date``, ``player_id``.
    """
    cache: LineupCache = {}

    grouped = lineup_df.groupby(["team", "match_date", "datetime"])
    match_xis: list[dict[str, Any]] = []
    for (team, mdate, dt_val), grp in grouped:
        player_ids = set(grp["player_id"].astype(str))
        match_xis.append(
            {
                "team": str(team),
                "match_date": str(mdate),
                "datetime": dt_val,
                "player_ids": player_ids,
            }
        )

    match_xis.sort(key=lambda x: (x["team"], x["datetime"]))

    for match in match_xis:
        team: str = match["team"]
        cache.setdefault(team, []).append(
            _TeamMatchXI(match_date=match["match_date"], player_ids=match["player_ids"])
        )

    logger.info(
        "lineup_cache_built",
        teams=len(cache),
        total_entries=sum(len(v) for v in cache.values()),
    )
    return cache


def _compute_team_features(
    team_matches: list[_TeamMatchXI],
    current_date: str,
) -> tuple[float, float] | None:
    """Compute xi_changes and cumulative_starts_lost for a team on a given date.

    Returns None if no current or prior match is found.
    """
    current_xi: set[str] | None = None
    prior_idx: int | None = None

    for i, m in enumerate(team_matches):
        if m.match_date == current_date:
            current_xi = m.player_ids
            if i > 0:
                prior_idx = i - 1
            break

    if current_xi is None or prior_idx is None:
        return None

    prior_xi = team_matches[prior_idx].player_ids
    dropped_ids = prior_xi - current_xi
    xi_changes = float(len(dropped_ids))

    # Cumulative starts lost: for each dropped player, count starts in the
    # last _STARTS_WINDOW matches (point-in-time sliding window)
    window_start = max(0, prior_idx + 1 - _STARTS_WINDOW)
    window_matches = team_matches[window_start : prior_idx + 1]

    cumulative_starts_lost = 0.0
    for pid in dropped_ids:
        starts = sum(1 for m in window_matches if pid in m.player_ids)
        cumulative_starts_lost += starts

    return xi_changes, cumulative_starts_lost


def extract_epl_lineup_features(
    lineup_cache: LineupCache,
    event: Event,
) -> EplLineupFeatures:
    """Extract lineup-delta features for a single event.

    Uses the prebuilt lineup cache to find current and previous starting XIs
    for both teams. Returns all-None when data is unavailable.
    """
    match_date = str(event.commence_time.date())

    home_result = None
    away_result = None

    home_matches = lineup_cache.get(event.home_team)
    if home_matches:
        home_result = _compute_team_features(home_matches, match_date)

    away_matches = lineup_cache.get(event.away_team)
    if away_matches:
        away_result = _compute_team_features(away_matches, match_date)

    if home_result is None and away_result is None:
        return EplLineupFeatures()

    home_xi_changes = home_result[0] if home_result else None
    home_csl = home_result[1] if home_result else None
    away_xi_changes = away_result[0] if away_result else None
    away_csl = away_result[1] if away_result else None

    diff_xi_changes: float | None = None
    if home_xi_changes is not None and away_xi_changes is not None:
        diff_xi_changes = home_xi_changes - away_xi_changes

    diff_csl: float | None = None
    if home_csl is not None and away_csl is not None:
        diff_csl = home_csl - away_csl

    return EplLineupFeatures(
        home_xi_changes=home_xi_changes,
        away_xi_changes=away_xi_changes,
        diff_xi_changes=diff_xi_changes,
        home_cumulative_starts_lost=home_csl,
        away_cumulative_starts_lost=away_csl,
        diff_cumulative_starts_lost=diff_csl,
    )
