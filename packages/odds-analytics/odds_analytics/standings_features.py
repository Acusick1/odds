"""EPL standings and form feature extraction for CLV prediction.

Derives league table context (position, points, goal difference, form) from
historical match results. Processes matches chronologically per season to
produce point-in-time standings with no look-ahead bias.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields
from datetime import datetime

import numpy as np
import structlog
from odds_core.models import Event, EventStatus
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "StandingsFeatures",
    "extract_standings_features",
    "build_league_table",
    "TeamRecord",
    "load_season_events_cache",
    "get_prior_events_from_cache",
]

logger = structlog.get_logger()

_FORM_WINDOW = 5
_POINTS_WIN = 3
_POINTS_DRAW = 1

_SUPPORTED_SPORT_KEYS = frozenset({"soccer_epl"})

# Type alias for the preloaded cache: season key -> list of completed events (chronological)
StandingsCache = dict[str, list[Event]]


@dataclass
class StandingsFeatures:
    """Point-in-time league standings features for a single EPL event.

    All fields optional (None -> np.nan). Computed from completed matches
    in the same season prior to the current match.
    """

    home_league_position: float | None = None
    away_league_position: float | None = None
    points_gap: float | None = None
    home_goal_difference: float | None = None
    away_goal_difference: float | None = None
    home_form_last_5: float | None = None
    away_form_last_5: float | None = None
    home_goals_scored_rate: float | None = None
    home_goals_conceded_rate: float | None = None
    away_goals_scored_rate: float | None = None
    away_goals_conceded_rate: float | None = None

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


@dataclass
class TeamRecord:
    """Mutable league table record for a single team."""

    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    form: list[float] = field(default_factory=list)
    form_window: int = _FORM_WINDOW

    @property
    def points(self) -> int:
        return self.won * _POINTS_WIN + self.drawn * _POINTS_DRAW

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def goals_scored_rate(self) -> float | None:
        if self.played == 0:
            return None
        return self.goals_for / self.played

    @property
    def goals_conceded_rate(self) -> float | None:
        if self.played == 0:
            return None
        return self.goals_against / self.played

    @property
    def form_last_n(self) -> float | None:
        if not self.form:
            return None
        window = self.form[-self.form_window :]
        return sum(window) / len(window)

    def record_result(self, goals_for: int, goals_against: int) -> None:
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        if goals_for > goals_against:
            self.won += 1
            self.form.append(3.0)
        elif goals_for == goals_against:
            self.drawn += 1
            self.form.append(1.0)
        else:
            self.lost += 1
            self.form.append(0.0)


def epl_season_key(dt: datetime) -> str:
    """Derive EPL season string from a datetime.

    EPL season runs Aug-May. A match in Jan 2025 -> '2024-25'.
    A match in Aug 2024 -> '2024-25'.
    """
    year = dt.year
    if dt.month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def _sort_table(records: dict[str, TeamRecord]) -> list[TeamRecord]:
    """Sort league table by points, then goal difference, then goals scored."""
    return sorted(
        records.values(),
        key=lambda r: (r.points, r.goal_difference, r.goals_for),
        reverse=True,
    )


def build_league_table(
    completed_events: list[Event],
    form_window: int = _FORM_WINDOW,
) -> dict[str, TeamRecord]:
    """Build league table from a list of completed events.

    Events must be pre-filtered to a single season and ordered chronologically.
    Only events with FINAL status and non-null scores are processed.
    """
    records: dict[str, TeamRecord] = {}
    for event in completed_events:
        if event.status != EventStatus.FINAL:
            continue
        if event.home_score is None or event.away_score is None:
            continue

        if event.home_team not in records:
            records[event.home_team] = TeamRecord(team=event.home_team, form_window=form_window)
        if event.away_team not in records:
            records[event.away_team] = TeamRecord(team=event.away_team, form_window=form_window)

        records[event.home_team].record_result(event.home_score, event.away_score)
        records[event.away_team].record_result(event.away_score, event.home_score)

    return records


async def load_season_events_cache(
    session: AsyncSession,
    sport_key: str,
) -> StandingsCache:
    """Bulk-load all completed events for a sport, grouped by season.

    Returns a dict mapping season key (e.g. '2024-25') to a chronologically
    ordered list of completed events. Intended to be called once before the
    per-event loop to avoid N+1 queries.
    """
    if sport_key not in _SUPPORTED_SPORT_KEYS:
        raise ValueError(
            f"Standings features only support {_SUPPORTED_SPORT_KEYS}, got '{sport_key}'"
        )

    result = await session.execute(
        select(Event)
        .where(
            Event.sport_key == sport_key,
            Event.status == EventStatus.FINAL,
            Event.home_score.is_not(None),
            Event.away_score.is_not(None),
        )
        .order_by(Event.commence_time)
    )
    all_events = result.scalars().all()

    cache: StandingsCache = defaultdict(list)
    for event in all_events:
        season = epl_season_key(event.commence_time)
        cache[season].append(event)

    logger.info(
        "standings_cache_loaded",
        sport_key=sport_key,
        seasons=len(cache),
        total_events=len(all_events),
    )
    return dict(cache)


def get_prior_events_from_cache(
    cache: StandingsCache,
    event: Event,
) -> list[Event]:
    """Extract prior season events for a given event from the preloaded cache.

    Returns completed events from the same season with commence_time strictly
    before the given event, in chronological order.
    """
    season = epl_season_key(event.commence_time)
    season_events = cache.get(season, [])
    return [e for e in season_events if e.commence_time < event.commence_time]


def extract_standings_features(
    prior_events: list[Event],
    event: Event,
    form_window: int = _FORM_WINDOW,
) -> StandingsFeatures:
    """Extract standings features from completed matches prior to event.

    Args:
        prior_events: Completed EPL events from the same season, ordered
            chronologically, all strictly before the current event.
        event: The current event to extract features for.
        form_window: Number of recent matches used for form calculation.

    Returns:
        StandingsFeatures with league context, or all-None if no prior
        matches exist in the season.
    """
    if not prior_events:
        return StandingsFeatures()

    table = build_league_table(prior_events, form_window=form_window)
    sorted_table = _sort_table(table)

    # Build position lookup
    positions: dict[str, int] = {}
    for i, record in enumerate(sorted_table, start=1):
        positions[record.team] = i

    home = table.get(event.home_team)
    away = table.get(event.away_team)

    if home is None and away is None:
        return StandingsFeatures()

    home_pos = float(positions[home.team]) if home is not None else None
    away_pos = float(positions[away.team]) if away is not None else None

    points_gap: float | None = None
    if home is not None and away is not None:
        points_gap = float(home.points - away.points)

    return StandingsFeatures(
        home_league_position=home_pos,
        away_league_position=away_pos,
        points_gap=points_gap,
        home_goal_difference=float(home.goal_difference) if home is not None else None,
        away_goal_difference=float(away.goal_difference) if away is not None else None,
        home_form_last_5=home.form_last_n if home is not None else None,
        away_form_last_5=away.form_last_n if away is not None else None,
        home_goals_scored_rate=home.goals_scored_rate if home is not None else None,
        home_goals_conceded_rate=home.goals_conceded_rate if home is not None else None,
        away_goals_scored_rate=away.goals_scored_rate if away is not None else None,
        away_goals_conceded_rate=away.goals_conceded_rate if away is not None else None,
    )
