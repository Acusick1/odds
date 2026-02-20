"""Rest and schedule feature extraction for CLV prediction.

Provides a RestScheduleFeatures dataclass and extractor that converts NBA
game log data into ML-ready rest and schedule features. Back-to-back games
and rest asymmetry are well-established predictive signals in sports analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date

import numpy as np
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.models import Event

__all__ = [
    "RestScheduleFeatures",
    "extract_rest_features",
]


@dataclass
class RestScheduleFeatures:
    """Rest and schedule features for a single event.

    All fields optional (None -> np.nan). Days rest counts calendar days
    between game dates: back-to-back (consecutive days) = 1.
    """

    home_days_rest: float | None = None
    away_days_rest: float | None = None
    rest_advantage: float | None = None
    home_is_b2b: float | None = None
    away_is_b2b: float | None = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array. None -> np.nan."""
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


def _identify_teams(
    event_logs: list[NbaTeamGameLog],
) -> tuple[str | None, str | None]:
    """Identify home and away team abbreviations from matchup strings.

    NBA matchup format: 'BOS vs. NYK' (home) or 'NYK @ BOS' (away).
    """
    home_abbrev: str | None = None
    away_abbrev: str | None = None
    for log in event_logs:
        if " vs. " in log.matchup:
            home_abbrev = log.team_abbreviation
        elif " @ " in log.matchup:
            away_abbrev = log.team_abbreviation
    return home_abbrev, away_abbrev


def _find_previous_game(
    all_logs: list[NbaTeamGameLog],
    team_abbreviation: str,
    current_game_date: date,
) -> NbaTeamGameLog | None:
    """Find the most recent game for a team before current_game_date."""
    prior = [
        g
        for g in all_logs
        if g.team_abbreviation == team_abbreviation and g.game_date < current_game_date
    ]
    if not prior:
        return None
    return max(prior, key=lambda g: g.game_date)


def extract_rest_features(
    game_logs: list[NbaTeamGameLog],
    event: Event,
) -> RestScheduleFeatures:
    """Extract rest/schedule features from pre-loaded game logs.

    Takes game logs for both teams (event's own logs + prior games) and
    computes rest features. Schedule data is public knowledge, so there
    is no look-ahead bias concern.

    Args:
        game_logs: Event's game logs + prior games for each team.
        event: Sportsbook event (provides event_id for filtering).

    Returns:
        RestScheduleFeatures, or all-None if no game logs available.
    """
    if not game_logs:
        return RestScheduleFeatures()

    event_logs = [g for g in game_logs if g.event_id == event.id]
    if not event_logs:
        return RestScheduleFeatures()

    current_game_date = event_logs[0].game_date
    home_abbrev, away_abbrev = _identify_teams(event_logs)

    home_days_rest: float | None = None
    away_days_rest: float | None = None

    if home_abbrev:
        prev = _find_previous_game(game_logs, home_abbrev, current_game_date)
        if prev:
            home_days_rest = float((current_game_date - prev.game_date).days)

    if away_abbrev:
        prev = _find_previous_game(game_logs, away_abbrev, current_game_date)
        if prev:
            away_days_rest = float((current_game_date - prev.game_date).days)

    rest_advantage: float | None = None
    if home_days_rest is not None and away_days_rest is not None:
        rest_advantage = home_days_rest - away_days_rest

    home_is_b2b: float | None = None
    if home_days_rest is not None:
        home_is_b2b = 1.0 if home_days_rest == 1.0 else 0.0

    away_is_b2b: float | None = None
    if away_days_rest is not None:
        away_is_b2b = 1.0 if away_days_rest == 1.0 else 0.0

    return RestScheduleFeatures(
        home_days_rest=home_days_rest,
        away_days_rest=away_days_rest,
        rest_advantage=rest_advantage,
        home_is_b2b=home_is_b2b,
        away_is_b2b=away_is_b2b,
    )
