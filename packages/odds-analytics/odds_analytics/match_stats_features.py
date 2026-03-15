"""Rolling match stats feature extraction for CLV prediction.

Computes rolling N-match averages of per-team match stats (shots, shots on
target, corners, fouls, cards, half-time goals) from historical FDUK data
stored in snapshot raw_data. Post-match stats from *prior* games are used as
pre-match features for the *current* game — no look-ahead bias.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import numpy as np
import structlog
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "MatchStatsFeatures",
    "extract_match_stats_features",
    "MatchStatsCache",
    "load_match_stats_cache",
    "get_prior_match_stats_from_cache",
]

logger = structlog.get_logger()

_DEFAULT_WINDOW = 5

# Stat keys stored in raw_data["match_stats"] by the ingestion script.
# Each tuple: (home_key, away_key, feature_name_suffix).
_STAT_KEYS: list[tuple[str, str, str]] = [
    ("home_shots", "away_shots", "shots"),
    ("home_shots_on_target", "away_shots_on_target", "shots_on_target"),
    ("home_corners", "away_corners", "corners"),
    ("home_fouls", "away_fouls", "fouls"),
    ("home_yellow_cards", "away_yellow_cards", "yellow_cards"),
    ("home_red_cards", "away_red_cards", "red_cards"),
    ("home_ht_goals", "away_ht_goals", "ht_goals"),
]


@dataclass
class MatchStatsFeatures:
    """Rolling match stats features for a single EPL event.

    All fields optional (None -> np.nan). Each field is a rolling average
    over the configured window of prior matches for the respective team.
    """

    home_avg_shots: float | None = None
    away_avg_shots: float | None = None
    home_avg_shots_on_target: float | None = None
    away_avg_shots_on_target: float | None = None
    home_avg_corners: float | None = None
    away_avg_corners: float | None = None
    home_avg_fouls: float | None = None
    away_avg_fouls: float | None = None
    home_avg_yellow_cards: float | None = None
    away_avg_yellow_cards: float | None = None
    home_avg_red_cards: float | None = None
    away_avg_red_cards: float | None = None
    home_avg_ht_goals: float | None = None
    away_avg_ht_goals: float | None = None

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
class _TeamMatchEntry:
    """A single match's stats for one team, with timestamp for ordering."""

    commence_time: datetime
    stats: dict[str, int]


# Maps team_name -> chronologically ordered list of match stat entries
MatchStatsCache = dict[str, list[_TeamMatchEntry]]


def _extract_team_stats_from_raw_data(
    raw_data: dict[str, Any],
    home_team: str,
    away_team: str,
) -> dict[str, dict[str, int]]:
    """Extract per-team stat dicts from snapshot raw_data.

    Returns a dict mapping team name -> {stat_suffix: value}.
    """
    ms = raw_data.get("match_stats")
    if not ms:
        return {}

    result: dict[str, dict[str, int]] = {}

    for home_key, away_key, suffix in _STAT_KEYS:
        home_val = ms.get(home_key)
        away_val = ms.get(away_key)

        if home_val is not None:
            result.setdefault(home_team, {})[suffix] = int(home_val)
        if away_val is not None:
            result.setdefault(away_team, {})[suffix] = int(away_val)

    return result


def _rolling_average(
    entries: list[_TeamMatchEntry],
    stat_key: str,
    window: int,
) -> float | None:
    """Compute rolling average of a stat over the last `window` entries."""
    values = [e.stats[stat_key] for e in entries if stat_key in e.stats]
    if not values:
        return None
    window_values = values[-window:]
    return sum(window_values) / len(window_values)


async def load_match_stats_cache(
    session: AsyncSession,
    sport_key: str,
) -> MatchStatsCache:
    """Bulk-load all match stats from FDUK snapshots, grouped by team.

    Loads closing-tier snapshots (1h before game) for all completed events,
    extracts match_stats from raw_data, and builds a chronological per-team
    cache. Called once before the per-event loop to avoid N+1 queries.
    """
    # Load all completed events for the sport
    event_result = await session.execute(
        select(Event)
        .where(
            Event.sport_key == sport_key,
            Event.status == EventStatus.FINAL,
            Event.home_score.is_not(None),
            Event.away_score.is_not(None),
        )
        .order_by(Event.commence_time)
    )
    all_events = list(event_result.scalars().all())

    if not all_events:
        return {}

    event_ids = [e.id for e in all_events]
    event_map = {e.id: e for e in all_events}

    # Load all FDUK snapshots for these events
    snapshot_result = await session.execute(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.event_id.in_(event_ids),
            OddsSnapshot.api_request_id == "football_data_uk",
        )
        .order_by(OddsSnapshot.snapshot_time)
    )
    all_snapshots = list(snapshot_result.scalars().all())

    # Pick one snapshot per event (prefer closing = latest by time)
    best_snapshot: dict[str, OddsSnapshot] = {}
    for snap in all_snapshots:
        existing = best_snapshot.get(snap.event_id)
        if existing is None or snap.snapshot_time > existing.snapshot_time:
            best_snapshot[snap.event_id] = snap

    cache: MatchStatsCache = defaultdict(list)

    for event_id, snapshot in best_snapshot.items():
        event = event_map.get(event_id)
        if event is None:
            continue

        raw = snapshot.raw_data
        if not raw or "match_stats" not in raw:
            continue

        team_stats = _extract_team_stats_from_raw_data(raw, event.home_team, event.away_team)
        for team, stats in team_stats.items():
            if stats:
                cache[team].append(_TeamMatchEntry(commence_time=event.commence_time, stats=stats))

    # Sort each team's entries chronologically
    for entries in cache.values():
        entries.sort(key=lambda e: e.commence_time)

    logger.info(
        "match_stats_cache_loaded",
        sport_key=sport_key,
        teams=len(cache),
        total_entries=sum(len(v) for v in cache.values()),
    )
    return dict(cache)


def get_prior_match_stats_from_cache(
    cache: MatchStatsCache,
    event: Event,
) -> dict[str, list[dict[str, int]]]:
    """Get prior match stats for both teams, filtered to before current event.

    Returns dict mapping team name to list of stat dicts (chronological,
    all strictly before the current event's commence_time).
    """
    result: dict[str, list[dict[str, int]]] = {}
    for team in (event.home_team, event.away_team):
        entries = cache.get(team)
        if not entries:
            continue
        prior = [e.stats for e in entries if e.commence_time < event.commence_time]
        if prior:
            result[team] = prior
    return result


def extract_match_stats_features(
    prior_team_stats: dict[str, list[dict[str, int]]],
    event: Event,
    window: int = _DEFAULT_WINDOW,
) -> MatchStatsFeatures:
    """Extract rolling match stats features from prior games.

    Args:
        prior_team_stats: Maps team name -> chronological list of per-match
            stat dicts. Must only contain matches *before* the current event.
        event: The current event to extract features for.
        window: Rolling window size for averaging.

    Returns:
        MatchStatsFeatures with rolling averages, or all-None if no prior
        stats exist for either team.
    """
    home_history = prior_team_stats.get(event.home_team, [])
    away_history = prior_team_stats.get(event.away_team, [])

    if not home_history and not away_history:
        return MatchStatsFeatures()

    # Wrap in _TeamMatchEntry for _rolling_average (dummy timestamp, not used)
    dummy_time = event.commence_time
    home_entries = [_TeamMatchEntry(commence_time=dummy_time, stats=s) for s in home_history]
    away_entries = [_TeamMatchEntry(commence_time=dummy_time, stats=s) for s in away_history]

    kwargs: dict[str, Any] = {}
    for _, _, suffix in _STAT_KEYS:
        kwargs[f"home_avg_{suffix}"] = _rolling_average(home_entries, suffix, window)
        kwargs[f"away_avg_{suffix}"] = _rolling_average(away_entries, suffix, window)

    return MatchStatsFeatures(**kwargs)
