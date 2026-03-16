"""EPL schedule and rest feature extraction from match dates.

Derives rest-day features directly from Event commence times within the same
EPL season. When an all-competition fixture cache is provided (from ESPN data),
rest days account for European and cup matches, not just EPL.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from odds_core.models import Event

if TYPE_CHECKING:
    from odds_analytics.fixture_cache import FixtureCache

__all__ = [
    "EplScheduleFeatures",
    "extract_epl_schedule_features",
]


@dataclass
class EplScheduleFeatures:
    """Rest and schedule features for a single EPL event.

    All fields optional (None -> np.nan). Rest days are fractional days
    between commence times: a Saturday 15:00 to Tuesday 19:45 = ~3.2.
    """

    home_rest_days: float | None = None
    away_rest_days: float | None = None
    rest_advantage: float | None = None
    is_midweek: float | None = None

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


def _rest_days_from_date(event_time: datetime, prev_time: datetime) -> float:
    """Compute fractional rest days between two datetimes."""
    return (event_time - prev_time).total_seconds() / 86400.0


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
    return _rest_days_from_date(event_time, most_recent)


def extract_epl_schedule_features(
    prior_events: list[Event],
    event: Event,
    fixture_cache: FixtureCache | None = None,
) -> EplScheduleFeatures:
    """Extract rest/schedule features from prior events.

    When ``fixture_cache`` is provided, rest days account for all competitions
    (EPL + cups + European). Without it, only EPL matches are considered
    (backward-compatible with the original behavior).

    Args:
        prior_events: Completed EPL events from the same season,
            ordered chronologically, all strictly before the current event.
        event: The current event to extract features for.
        fixture_cache: Optional all-competition fixture cache from ESPN data.

    Returns:
        EplScheduleFeatures, or all-None if no prior matches found.
    """
    home_prev = _find_previous_match(prior_events, event.home_team)
    away_prev = _find_previous_match(prior_events, event.away_team)

    # All-competition lookup (if cache available)
    home_fixture_date: datetime | None = None
    away_fixture_date: datetime | None = None
    if fixture_cache is not None:
        from odds_analytics.fixture_cache import get_last_fixture_date

        home_fixture_date = get_last_fixture_date(
            fixture_cache, event.home_team, event.commence_time
        )
        away_fixture_date = get_last_fixture_date(
            fixture_cache, event.away_team, event.commence_time
        )

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
    )
