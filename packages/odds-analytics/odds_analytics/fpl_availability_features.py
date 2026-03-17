"""FPL availability feature extraction for CLV prediction.

Computes expected squad disruption from FPL chance_of_playing data, weighted
by each player's cumulative ESPN starts over a sliding 38-match window.
Features are available 24-48h before kickoff (pre-decision tier).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    import pandas as pd
    from odds_core.models import Event

__all__ = [
    "FplAvailabilityFeatures",
    "FplAvailabilityCache",
    "load_fpl_availability_cache",
    "extract_fpl_availability_features",
]

logger = structlog.get_logger()

_STARTS_WINDOW = 38
_SNAPSHOT_LOOKBACK_HOURS = 48

_FPL_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "fpl_availability"
_LINEUPS_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "espn_lineups"


@dataclass
class FplAvailabilityFeatures:
    home_expected_disruption: float | None = None
    away_expected_disruption: float | None = None
    diff_expected_disruption: float | None = None
    home_expected_disruption_unweighted: float | None = None
    away_expected_disruption_unweighted: float | None = None
    diff_expected_disruption_unweighted: float | None = None
    home_n_flagged_players: float | None = None
    away_n_flagged_players: float | None = None
    diff_n_flagged_players: float | None = None

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
class FplAvailabilityCache:
    """Precomputed cache for FPL availability feature extraction.

    Attributes:
        fpl_df: FPL availability snapshots DataFrame.
        player_map: Mapping from (team, fpl_player_code) to ESPN player_id.
        starts_lookup: Mapping from (team, match_date) to {player_id: start_count}.
        snapshot_times: Sorted list of unique FPL snapshot datetimes.
    """

    fpl_df: pd.DataFrame
    player_map: dict[tuple[str, int], str]
    starts_lookup: dict[tuple[str, Any], dict[str, int]]
    snapshot_times: list[Any]


def _build_fpl_to_espn_player_map(
    fpl_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """Build fuzzy name mapping from (team, fpl_player_code) -> espn_player_id.

    Matches within the same team using SequenceMatcher on player names.
    ~70% accuracy; unmatched players get zero weight (not NaN).
    """

    espn_players: dict[str, list[tuple[str, str]]] = {}
    for _, row in (
        lineup_df[["team", "player_id", "player_name"]]
        .drop_duplicates(subset=["team", "player_id"])
        .iterrows()
    ):
        espn_players.setdefault(str(row["team"]), []).append(
            (str(row["player_id"]), str(row["player_name"]))
        )

    fpl_players: dict[str, list[tuple[int, str]]] = {}
    for _, row in (
        fpl_df[["team", "player_code", "player_name"]]
        .drop_duplicates(subset=["team", "player_code"])
        .iterrows()
    ):
        fpl_players.setdefault(str(row["team"]), []).append(
            (int(row["player_code"]), str(row["player_name"]))
        )

    mapping: dict[tuple[str, int], str] = {}
    matched = 0
    unmatched = 0

    for team in fpl_players:
        if team not in espn_players:
            unmatched += len(fpl_players[team])
            continue

        espn_list = espn_players[team]

        for fpl_code, fpl_name in fpl_players[team]:
            best_score = 0.0
            best_espn_id = ""
            fpl_lower = fpl_name.lower()

            for espn_id, espn_name in espn_list:
                espn_lower = espn_name.lower()
                espn_parts = espn_lower.split()
                score_full = SequenceMatcher(None, fpl_lower, espn_lower).ratio()
                score_last = (
                    SequenceMatcher(None, fpl_lower, espn_parts[-1]).ratio() if espn_parts else 0.0
                )
                score = max(score_full, score_last)
                if score > best_score:
                    best_score = score
                    best_espn_id = espn_id

            if best_score >= 0.6 and best_espn_id:
                mapping[(team, fpl_code)] = best_espn_id
                matched += 1
            else:
                unmatched += 1

    logger.info(
        "fpl_to_espn_player_map_built",
        matched=matched,
        unmatched=unmatched,
    )
    return mapping


def _precompute_cumulative_starts(
    lineup_df: pd.DataFrame,
    window: int = _STARTS_WINDOW,
) -> dict[tuple[str, Any], dict[str, int]]:
    """Precompute cumulative starts for every (team, match_date) pair.

    Returns a lookup keyed by (team, match_date) -> {player_id: start_count}.
    For each entry, counts starts in the `window` matches strictly before that date.
    """
    lookup: dict[tuple[str, Any], dict[str, int]] = {}

    for team, team_df in lineup_df.groupby("team"):
        team_df = team_df.sort_values("datetime")
        unique_dates = team_df["match_date"].unique()

        date_players: list[tuple[Any, list[str]]] = []
        for mdate in unique_dates:
            pids = team_df[team_df["match_date"] == mdate]["player_id"].astype(str).tolist()
            date_players.append((mdate, pids))

        for i, (mdate, _) in enumerate(date_players):
            start_idx = max(0, i - window)
            starts: dict[str, int] = {}
            for j in range(start_idx, i):
                for pid in date_players[j][1]:
                    starts[pid] = starts.get(pid, 0) + 1
            lookup[(str(team), mdate)] = starts

    return lookup


def load_fpl_availability_cache(
    fpl_data_dir: Path | None = None,
    lineups_data_dir: Path | None = None,
) -> FplAvailabilityCache | None:
    """Load FPL availability and ESPN lineup CSVs and build the feature cache.

    Returns None if no FPL CSV files are found (e.g. in Lambda or before first run).
    """
    import pandas as pd

    fpl_dir = fpl_data_dir if fpl_data_dir is not None else _FPL_DATA_DIR
    lineups_dir = lineups_data_dir if lineups_data_dir is not None else _LINEUPS_DATA_DIR

    fpl_csv_files = sorted(fpl_dir.glob("fpl_availability_*.csv"))
    if not fpl_csv_files:
        logger.warning("fpl_availability_csvs_not_found", data_dir=str(fpl_dir))
        return None

    lineup_csv_files = sorted(lineups_dir.glob("lineups_*.csv"))
    if not lineup_csv_files:
        logger.warning("espn_lineups_not_found_for_fpl", data_dir=str(lineups_dir))
        return None

    fpl_frames = [pd.read_csv(f) for f in fpl_csv_files]
    fpl_df = pd.concat(fpl_frames, ignore_index=True)
    fpl_df["snapshot_time"] = pd.to_datetime(fpl_df["snapshot_time"], utc=True)

    lineup_frames = [pd.read_csv(f) for f in lineup_csv_files]
    lineup_df = pd.concat(lineup_frames, ignore_index=True)
    lineup_df["datetime"] = pd.to_datetime(lineup_df["date"], utc=True)
    lineup_df["match_date"] = lineup_df["datetime"].dt.date
    lineup_df = lineup_df[lineup_df["starter"].astype(str).str.lower() == "true"].copy()

    player_map = _build_fpl_to_espn_player_map(fpl_df, lineup_df)
    starts_lookup = _precompute_cumulative_starts(lineup_df)
    snapshot_times = sorted(fpl_df["snapshot_time"].unique())

    logger.info(
        "fpl_availability_cache_built",
        fpl_rows=len(fpl_df),
        lineup_rows=len(lineup_df),
        fpl_csv_files=len(fpl_csv_files),
        lineup_csv_files=len(lineup_csv_files),
        snapshot_times=len(snapshot_times),
    )

    return FplAvailabilityCache(
        fpl_df=fpl_df,
        player_map=player_map,
        starts_lookup=starts_lookup,
        snapshot_times=snapshot_times,
    )


def _compute_team_disruption(
    team: str,
    snap_data: pd.DataFrame,
    match_date: dt.date,
    player_map: dict[tuple[str, int], str],
    starts_lookup: dict[tuple[str, Any], dict[str, int]],
) -> tuple[float, float, float]:
    """Compute disruption metrics for one team from one FPL snapshot.

    Returns (expected_disruption_weighted, expected_disruption_unweighted, n_flagged).
    """
    team_players = snap_data[snap_data["team"] == team]
    cum_starts = starts_lookup.get((team, match_date), {})

    disruption_weighted = 0.0
    disruption_unweighted = 0.0
    n_flagged = 0

    for _, player in team_players.iterrows():
        chance = float(player["chance_of_playing"])
        if chance >= 100:
            continue

        severity = (100.0 - chance) / 100.0
        n_flagged += 1
        disruption_unweighted += severity

        fpl_code = int(player["player_code"])
        espn_id = player_map.get((team, fpl_code))
        weight = 0.0
        if espn_id:
            weight = float(cum_starts.get(espn_id, 0))
        disruption_weighted += severity * weight

    return disruption_weighted, disruption_unweighted, float(n_flagged)


def extract_fpl_availability_features(
    cache: FplAvailabilityCache | None,
    event: Event,
) -> FplAvailabilityFeatures:
    """Extract FPL availability features for a single event.

    Finds the latest FPL snapshot within [commence_time - 48h, commence_time)
    and computes expected squad disruption for both teams. Returns all-None when
    data is unavailable, cache is None, or no snapshot is found in the window.
    """
    if cache is None:
        return FplAvailabilityFeatures()

    commence = event.commence_time
    cutoff_early = commence - dt.timedelta(hours=_SNAPSHOT_LOOKBACK_HOURS)

    valid_snaps = [t for t in cache.snapshot_times if cutoff_early <= t < commence]
    if not valid_snaps:
        return FplAvailabilityFeatures()

    snap_time = max(valid_snaps)
    snap_data = cache.fpl_df[cache.fpl_df["snapshot_time"] == snap_time]

    match_date = commence.date()

    home_weighted, home_unweighted, home_n = _compute_team_disruption(
        event.home_team, snap_data, match_date, cache.player_map, cache.starts_lookup
    )
    away_weighted, away_unweighted, away_n = _compute_team_disruption(
        event.away_team, snap_data, match_date, cache.player_map, cache.starts_lookup
    )

    return FplAvailabilityFeatures(
        home_expected_disruption=home_weighted,
        away_expected_disruption=away_weighted,
        diff_expected_disruption=home_weighted - away_weighted,
        home_expected_disruption_unweighted=home_unweighted,
        away_expected_disruption_unweighted=away_unweighted,
        diff_expected_disruption_unweighted=home_unweighted - away_unweighted,
        home_n_flagged_players=home_n,
        away_n_flagged_players=away_n,
        diff_n_flagged_players=home_n - away_n,
    )
