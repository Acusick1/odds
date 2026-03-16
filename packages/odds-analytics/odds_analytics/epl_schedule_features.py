"""EPL schedule and rest feature extraction from match dates.

Derives rest-day and congestion features from Event commence times and an
optional all-competition fixtures DataFrame (from ESPN data). When the
DataFrame is provided, rest days account for European and cup matches.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from odds_core.models import Event

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "EplScheduleFeatures",
    "extract_epl_schedule_features",
]

_EUROPEAN_COMPETITIONS = frozenset(
    {
        "Champions League",
        "Europa League",
        "Conference League",
    }
)


@dataclass
class EplScheduleFeatures:
    """Rest, schedule, and congestion features for a single EPL event.

    All fields optional (None -> np.nan). Rest days are fractional days
    between commence times: a Saturday 15:00 to Tuesday 19:45 = ~3.2.
    """

    home_rest_days: float | None = None
    away_rest_days: float | None = None
    rest_advantage: float | None = None
    is_midweek: float | None = None
    home_matches_last_14d: float | None = None
    away_matches_last_14d: float | None = None
    home_european_last_7d: float | None = None
    away_european_last_7d: float | None = None

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


def _find_previous_match(
    prior_events: list[Event],
    team: str,
) -> Event | None:
    """Find the most recent prior event involving team (home or away).

    prior_events must be chronologically ordered and strictly before the
    current event.
    """
    for event in reversed(prior_events):
        if event.home_team == team or event.away_team == team:
            return event
    return None


def _team_fixtures_before(
    fixtures_df: pd.DataFrame,
    team: str,
    before: datetime,
) -> pd.DataFrame:
    """Filter fixtures DataFrame to a team's matches strictly before a time."""
    mask = ((fixtures_df["team"] == team) | (fixtures_df["opponent"] == team)) & (
        fixtures_df["date"] < before
    )
    return fixtures_df.loc[mask]


def _last_fixture_date(team_df: pd.DataFrame) -> datetime | None:
    """Get the most recent fixture date from a pre-filtered DataFrame."""
    if team_df.empty:
        return None
    return team_df["date"].max()


def _best_rest_days(
    epl_prev: Event | None,
    fixture_date: datetime | None,
    event_time: datetime,
) -> float | None:
    """Pick the shorter rest period from EPL-only and all-competition sources.

    Takes the more recent of the two previous-match dates, giving the minimum
    (most accurate) rest days. Either source may be None.
    """
    candidates: list[datetime] = []
    if epl_prev is not None:
        candidates.append(epl_prev.commence_time)
    if fixture_date is not None:
        candidates.append(fixture_date)

    if not candidates:
        return None

    most_recent = max(candidates)
    return (event_time - most_recent).total_seconds() / 86400.0


def _matches_in_window(team_df: pd.DataFrame, after: datetime) -> int:
    """Count matches in a team's DataFrame on or after the given time."""
    return int((team_df["date"] >= after).sum())


def _european_in_window(team_df: pd.DataFrame, after: datetime) -> bool:
    """Check if any European match occurred on or after the given time."""
    recent = team_df.loc[team_df["date"] >= after]
    if recent.empty:
        return False
    return bool(recent["competition"].isin(_EUROPEAN_COMPETITIONS).any())


def extract_epl_schedule_features(
    prior_events: list[Event],
    event: Event,
    fixtures_df: pd.DataFrame | None = None,
) -> EplScheduleFeatures:
    """Extract rest/schedule/congestion features from prior events.

    When ``fixtures_df`` is provided, rest days account for all competitions
    (EPL + cups + European) and congestion features are computed. Without it,
    only EPL matches are considered (backward-compatible).

    Args:
        prior_events: Completed EPL events from the same season,
            ordered chronologically, all strictly before the current event.
        event: The current event to extract features for.
        fixtures_df: Optional all-competition fixtures DataFrame with columns
            ``date``, ``team``, ``opponent``, ``competition``. Dates must be
            timezone-aware datetimes.

    Returns:
        EplScheduleFeatures with rest, congestion, and midweek fields.
    """
    home_prev = _find_previous_match(prior_events, event.home_team)
    away_prev = _find_previous_match(prior_events, event.away_team)

    home_fixture_date: datetime | None = None
    away_fixture_date: datetime | None = None
    home_matches_14d: float | None = None
    away_matches_14d: float | None = None
    home_european_7d: float | None = None
    away_european_7d: float | None = None

    if fixtures_df is not None and not fixtures_df.empty:
        t = event.commence_time

        home_df = _team_fixtures_before(fixtures_df, event.home_team, t)
        away_df = _team_fixtures_before(fixtures_df, event.away_team, t)

        home_fixture_date = _last_fixture_date(home_df)
        away_fixture_date = _last_fixture_date(away_df)

        cutoff_14d = t - timedelta(days=14)
        home_matches_14d = float(_matches_in_window(home_df, cutoff_14d))
        away_matches_14d = float(_matches_in_window(away_df, cutoff_14d))

        cutoff_7d = t - timedelta(days=7)
        home_european_7d = 1.0 if _european_in_window(home_df, cutoff_7d) else 0.0
        away_european_7d = 1.0 if _european_in_window(away_df, cutoff_7d) else 0.0

    home_rest = _best_rest_days(home_prev, home_fixture_date, event.commence_time)
    away_rest = _best_rest_days(away_prev, away_fixture_date, event.commence_time)

    rest_advantage: float | None = None
    if home_rest is not None and away_rest is not None:
        rest_advantage = home_rest - away_rest

    # Midweek: Tuesday (1), Wednesday (2)
    weekday = event.commence_time.weekday()
    is_midweek = 1.0 if weekday in (1, 2) else 0.0

    return EplScheduleFeatures(
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        rest_advantage=rest_advantage,
        is_midweek=is_midweek,
        home_matches_last_14d=home_matches_14d,
        away_matches_last_14d=away_matches_14d,
        home_european_last_7d=home_european_7d,
        away_european_last_7d=away_european_7d,
    )
