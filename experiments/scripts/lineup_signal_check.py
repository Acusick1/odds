"""Phase A: Lineup-delta signal check for CLV prediction.

Loads ESPN lineup CSVs, matches them to pipeline events by team name + date,
computes lineup-delta metrics, and correlates each against the observed
devigged bet365 CLV target.

Four feature categories:
  - Tendency (bias-free): rolling averages of xi_changes and
    cumulative_starts_lost over prior N matches (not including the current
    match). Available at any decision tier — no lineup announcement needed.
  - Match-specific: xi_jaccard, xi_changes, cumulative_starts_lost, gk_changed
    (require the current match's starting XI, available ~75min pre-KO).
  - Rolling (legacy): rolling averages that include the current match value
    (look-ahead bias when used before lineup announcement).
  - FPL availability: expected-disruption features derived from FPL
    chance_of_playing data, weighted by cumulative ESPN starts. Available
    24-48h before kickoff (pre-decision tier).

Outputs: correlation matrix, scatter plots, per-feature univariate R²,
with separate tendency vs match-specific vs FPL availability signal assessment.
Results saved to experiments/results/lineup_signal_check/.

Usage:
    uv run python experiments/scripts/lineup_signal_check.py
    uv run python experiments/scripts/lineup_signal_check.py --config experiments/configs/xgboost_epl_combined_standings_tuning.yaml
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.sequence_loader import (
    calculate_devigged_bookmaker_target,
    extract_odds_from_snapshot,
)
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sqlmodel import select

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "lineup_signal_check"

LINEUP_DIR = PROJECT_ROOT / "data" / "espn_lineups"
FPL_DIR = PROJECT_ROOT / "data" / "fpl_availability"

DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "xgboost_epl_combined_standings_tuning.yaml"

# Sliding window for cumulative starts (1 full season of matches)
STARTS_WINDOW = 38

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_lineups() -> pd.DataFrame:
    """Load all ESPN lineup CSVs into a single DataFrame."""
    csv_files = sorted(LINEUP_DIR.glob("lineups_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No lineup CSVs found in {LINEUP_DIR}")

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    # Parse date to datetime, extract date-only for matching
    df["datetime"] = pd.to_datetime(df["date"], utc=True)
    df["match_date"] = df["datetime"].dt.date

    # Filter to starters only
    df = df[df["starter"].astype(str).str.lower() == "true"].copy()

    print(f"Loaded {len(df)} starter rows from {len(csv_files)} lineup CSVs")
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_lineup_metrics(
    lineup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-team per-match lineup-delta metrics.

    For each team in each match, compare the starting XI to the previous
    match's starting XI. Weights dropped players by their cumulative starts
    over a sliding window (STARTS_WINDOW matches) — point-in-time, no future data.
    """
    match_xis: list[dict[str, Any]] = []

    for (team, mdate, ht, at, dt_val), grp in lineup_df.groupby(
        ["team", "match_date", "home_team", "away_team", "datetime"]
    ):
        player_ids: set[str] = set()
        for _, r in grp.iterrows():
            player_ids.add(str(r["player_id"]))
        match_xis.append(
            {
                "team": team,
                "match_date": mdate,
                "home_team": ht,
                "away_team": at,
                "datetime": dt_val,
                "player_ids": player_ids,
            }
        )

    match_xis.sort(key=lambda x: (x["team"], x["datetime"]))

    rows: list[dict[str, Any]] = []
    # team -> previous match's player_ids
    prev_xi: dict[str, set[str]] = {}
    # team -> sliding window of recent starting XIs (deque of sets)
    start_history: dict[str, deque[set[str]]] = {}

    for match in match_xis:
        team: str = match["team"]
        current_ids: set[str] = match["player_ids"]

        prev = prev_xi.get(team)
        history = start_history.get(team)

        if prev is not None and history is not None:
            xi_jaccard = _jaccard(current_ids, prev)
            dropped_ids = prev - current_ids
            xi_changes = len(dropped_ids)

            # Count cumulative starts for each dropped player over the window
            cumulative_starts_lost = 0.0
            for pid in dropped_ids:
                starts = sum(1 for xi in history if pid in xi)
                cumulative_starts_lost += starts

            rows.append(
                {
                    "team": team,
                    "match_date": match["match_date"],
                    "datetime": match["datetime"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "xi_jaccard": xi_jaccard,
                    "xi_changes": float(xi_changes),
                    "cumulative_starts_lost": cumulative_starts_lost,
                }
            )

        # Update state for next iteration
        prev_xi[team] = current_ids
        if team not in start_history:
            start_history[team] = deque(maxlen=STARTS_WINDOW)
        start_history[team].append(current_ids)

    metrics_df = pd.DataFrame(rows)

    # Tendency features: lagged rolling averages (exclude current match)
    metrics_df = metrics_df.sort_values(["team", "datetime"]).copy()
    for window in (3, 5):
        for col in ("xi_changes", "cumulative_starts_lost"):
            metrics_df[f"{col}_tendency_{window}"] = metrics_df.groupby("team")[col].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )

    print(f"Computed lineup metrics: {len(metrics_df)} team-match rows")
    return metrics_df


def build_gk_changed(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Compute GK changed flag per team per match."""
    gk_df = lineup_df[lineup_df["position"] == "G"].copy()
    gk_grouped = (
        gk_df.groupby(["team", "match_date", "home_team", "away_team", "datetime"])
        .agg(gk_ids=("player_id", lambda x: set(x.astype(str))))
        .reset_index()
        .sort_values(["team", "datetime"])
    )

    rows: list[dict[str, Any]] = []
    prev_gk: dict[str, set[str]] = {}

    for _, match in gk_grouped.iterrows():
        team: str = match["team"]
        current_gk: set[str] = match["gk_ids"]

        if team in prev_gk:
            gk_changed = 1.0 if current_gk != prev_gk[team] else 0.0
            rows.append(
                {
                    "team": team,
                    "match_date": match["match_date"],
                    "gk_changed": gk_changed,
                }
            )

        prev_gk[team] = current_gk

    return pd.DataFrame(rows)


def _compute_rolling_metrics(metrics_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add rolling average versions of metrics over last N matches."""
    metrics_df = metrics_df.sort_values(["team", "datetime"]).copy()
    metric_cols = ["xi_jaccard", "xi_changes", "cumulative_starts_lost"]

    for col in metric_cols:
        metrics_df[f"{col}_roll{window}"] = metrics_df.groupby("team")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    return metrics_df


# ---------------------------------------------------------------------------
# FPL availability data
# ---------------------------------------------------------------------------


def load_fpl_availability() -> pd.DataFrame | None:
    """Load all FPL availability CSVs into a single DataFrame."""
    csv_files = sorted(FPL_DIR.glob("fpl_availability_*.csv"))
    if not csv_files:
        return None

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    print(f"Loaded {len(df)} FPL availability rows from {len(csv_files)} CSVs")
    return df


def _build_fpl_to_espn_player_map(
    fpl_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """Build fuzzy name mapping from (team, fpl_player_code) -> espn_player_id.

    Matches within the same team using SequenceMatcher on player names.
    """
    # Build ESPN player lookup: team -> list of (player_id, player_name)
    espn_players: dict[str, list[tuple[str, str]]] = {}
    for _, row in (
        lineup_df[["team", "player_id", "player_name"]]
        .drop_duplicates(subset=["team", "player_id"])
        .iterrows()
    ):
        espn_players.setdefault(row["team"], []).append(
            (str(row["player_id"]), str(row["player_name"]))
        )

    # Build FPL player lookup: team -> list of (code, web_name)
    fpl_players: dict[str, list[tuple[int, str]]] = {}
    for _, row in (
        fpl_df[["team", "player_code", "player_name"]]
        .drop_duplicates(subset=["team", "player_code"])
        .iterrows()
    ):
        fpl_players.setdefault(row["team"], []).append(
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
                # Try matching FPL web_name against last part of ESPN name
                espn_parts = espn_lower.split()
                # Compare against full name and last name
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

    print(f"FPL-to-ESPN player matching: {matched} matched, {unmatched} unmatched")
    return mapping


def _precompute_cumulative_starts(
    lineup_df: pd.DataFrame,
    window: int = STARTS_WINDOW,
) -> dict[tuple[str, Any], dict[str, int]]:
    """Precompute cumulative starts for every (team, match_date) pair.

    Returns a lookup keyed by (team, match_date) -> {player_id: start_count}.
    For each entry, counts starts in the `window` matches strictly before that date.
    """
    lookup: dict[tuple[str, Any], dict[str, int]] = {}

    for team, team_df in lineup_df.groupby("team"):
        team_df = team_df.sort_values("datetime")
        # Get unique match dates in order
        unique_dates = team_df["match_date"].unique()

        # Build per-match-date player sets
        date_players: list[tuple[Any, list[str]]] = []
        for mdate in unique_dates:
            pids = team_df[team_df["match_date"] == mdate]["player_id"].astype(str).tolist()
            date_players.append((mdate, pids))

        # For each match date, compute starts from the preceding window matches
        for i, (mdate, _) in enumerate(date_players):
            start_idx = max(0, i - window)
            starts: dict[str, int] = {}
            for j in range(start_idx, i):
                for pid in date_players[j][1]:
                    starts[pid] = starts.get(pid, 0) + 1
            lookup[(team, mdate)] = starts

    return lookup


def compute_fpl_disruption_features(
    fpl_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute expected-disruption features from FPL availability data.

    For each pipeline event, finds the latest FPL snapshot within 48h before
    commence_time, then computes:
      - expected_disruption: sum of (100 - chance) / 100, weighted by cumulative starts
      - expected_disruption_unweighted: unweighted severity sum
      - n_flagged_players: count with chance_of_playing < 100
    """
    player_map = _build_fpl_to_espn_player_map(fpl_df, lineup_df)

    # Precompute cumulative starts for all (team, date) pairs
    starts_lookup = _precompute_cumulative_starts(lineup_df)

    # Get unique snapshot times sorted
    snapshot_times = sorted(fpl_df["snapshot_time"].unique())

    rows: list[dict[str, Any]] = []
    matched = 0
    no_snapshot = 0

    for _, event in events_df.iterrows():
        commence = pd.Timestamp(event["commence_time"])
        if commence.tzinfo is None:
            commence = commence.tz_localize(UTC)

        event_id = event["event_id"]
        home = event["home_team"]
        away = event["away_team"]
        match_date = event["match_date"]

        # Find latest FPL snapshot within 48h before commence_time
        cutoff_early = commence - timedelta(hours=48)
        valid_snaps = [t for t in snapshot_times if cutoff_early <= t < commence]
        if not valid_snaps:
            no_snapshot += 1
            continue

        snap_time = max(valid_snaps)
        snap_data = fpl_df[fpl_df["snapshot_time"] == snap_time]

        row_dict: dict[str, Any] = {"event_id": event_id}

        for side, team in [("home", home), ("away", away)]:
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

                # Weight by cumulative starts
                fpl_code = int(player["player_code"])
                espn_id = player_map.get((team, fpl_code))
                weight = 0.0
                if espn_id:
                    weight = float(cum_starts.get(espn_id, 0))
                disruption_weighted += severity * weight

            row_dict[f"{side}_expected_disruption"] = disruption_weighted
            row_dict[f"{side}_expected_disruption_unweighted"] = disruption_unweighted
            row_dict[f"{side}_n_flagged_players"] = float(n_flagged)

        # Differentials
        for feat in ("expected_disruption", "expected_disruption_unweighted", "n_flagged_players"):
            h = row_dict[f"home_{feat}"]
            a = row_dict[f"away_{feat}"]
            row_dict[f"diff_{feat}"] = h - a

        rows.append(row_dict)
        matched += 1

    print(f"FPL disruption: {matched} events matched, {no_snapshot} with no snapshot")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Match to pipeline events
# ---------------------------------------------------------------------------


def match_metrics_to_events(
    metrics_df: pd.DataFrame,
    gk_df: pd.DataFrame,
    event_ids: np.ndarray,
    targets: np.ndarray,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join lineup metrics to pipeline events by team + date.

    Returns DataFrame with home/away metrics + differential + CLV target,
    one row per event.
    """
    # Build event lookup: (home_team, away_team, match_date) -> (event_id, target)
    event_lookup: dict[tuple[str, str, str], tuple[str, float]] = {}
    for i, eid in enumerate(event_ids):
        row = events_df[events_df["event_id"] == eid]
        if row.empty:
            continue
        r = row.iloc[0]
        key = (r["home_team"], r["away_team"], str(r["match_date"]))
        event_lookup[key] = (eid, targets[i])

    # Build metrics lookup: (team, match_date) -> metrics row
    metrics_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in metrics_df.iterrows():
        key = (row["team"], str(row["match_date"]))
        metrics_lookup[key] = row.to_dict()

    # Build GK lookup: (team, match_date) -> gk_changed
    gk_lookup: dict[tuple[str, str], float] = {}
    for _, row in gk_df.iterrows():
        key = (row["team"], str(row["match_date"]))
        gk_lookup[key] = row["gk_changed"]

    metric_cols = [
        "xi_jaccard",
        "xi_changes",
        "cumulative_starts_lost",
        "xi_jaccard_roll3",
        "xi_changes_roll3",
        "cumulative_starts_lost_roll3",
        "xi_changes_tendency_3",
        "cumulative_starts_lost_tendency_3",
        "xi_changes_tendency_5",
        "cumulative_starts_lost_tendency_5",
    ]

    joined_rows: list[dict[str, Any]] = []
    matched = 0
    unmatched = 0
    partial = 0

    for (home, away, date_str), (eid, target) in event_lookup.items():
        home_metrics = metrics_lookup.get((home, date_str))
        away_metrics = metrics_lookup.get((away, date_str))

        if home_metrics is None and away_metrics is None:
            unmatched += 1
            continue

        if home_metrics is None or away_metrics is None:
            partial += 1

        row_dict: dict[str, Any] = {
            "event_id": eid,
            "target": target,
            "home_team": home,
            "away_team": away,
            "match_date": date_str,
        }

        for col in metric_cols:
            row_dict[f"home_{col}"] = (
                home_metrics[col] if home_metrics and col in home_metrics else np.nan
            )
            row_dict[f"away_{col}"] = (
                away_metrics[col] if away_metrics and col in away_metrics else np.nan
            )

            # Differential: home - away
            h = row_dict[f"home_{col}"]
            a = row_dict[f"away_{col}"]
            if not np.isnan(h) and not np.isnan(a):
                row_dict[f"diff_{col}"] = h - a
            else:
                row_dict[f"diff_{col}"] = np.nan

        # GK changed
        home_gk = gk_lookup.get((home, date_str), np.nan)
        away_gk = gk_lookup.get((away, date_str), np.nan)
        row_dict["home_gk_changed"] = home_gk
        row_dict["away_gk_changed"] = away_gk
        if not np.isnan(home_gk) and not np.isnan(away_gk):
            row_dict["diff_gk_changed"] = home_gk - away_gk
        else:
            row_dict["diff_gk_changed"] = np.nan

        joined_rows.append(row_dict)
        matched += 1

    print(
        f"Event matching: {matched} matched, {unmatched} unmatched, {partial} partial (one side only)"
    )
    return pd.DataFrame(joined_rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

TENDENCY_FEATURE_COLS = [
    "home_xi_changes_tendency_3",
    "away_xi_changes_tendency_3",
    "diff_xi_changes_tendency_3",
    "home_cumulative_starts_lost_tendency_3",
    "away_cumulative_starts_lost_tendency_3",
    "diff_cumulative_starts_lost_tendency_3",
    "home_xi_changes_tendency_5",
    "away_xi_changes_tendency_5",
    "diff_xi_changes_tendency_5",
    "home_cumulative_starts_lost_tendency_5",
    "away_cumulative_starts_lost_tendency_5",
    "diff_cumulative_starts_lost_tendency_5",
]

MATCH_SPECIFIC_FEATURE_COLS = [
    "home_xi_jaccard",
    "away_xi_jaccard",
    "diff_xi_jaccard",
    "home_xi_changes",
    "away_xi_changes",
    "diff_xi_changes",
    "home_gk_changed",
    "away_gk_changed",
    "diff_gk_changed",
    "home_cumulative_starts_lost",
    "away_cumulative_starts_lost",
    "diff_cumulative_starts_lost",
]

LEGACY_ROLLING_FEATURE_COLS = [
    "home_xi_jaccard_roll3",
    "away_xi_jaccard_roll3",
    "diff_xi_jaccard_roll3",
    "home_xi_changes_roll3",
    "away_xi_changes_roll3",
    "diff_xi_changes_roll3",
    "home_cumulative_starts_lost_roll3",
    "away_cumulative_starts_lost_roll3",
    "diff_cumulative_starts_lost_roll3",
]

FPL_AVAILABILITY_FEATURE_COLS = [
    "home_expected_disruption",
    "away_expected_disruption",
    "diff_expected_disruption",
    "home_expected_disruption_unweighted",
    "away_expected_disruption_unweighted",
    "diff_expected_disruption_unweighted",
    "home_n_flagged_players",
    "away_n_flagged_players",
    "diff_n_flagged_players",
]

LINEUP_FEATURE_COLS = (
    TENDENCY_FEATURE_COLS
    + MATCH_SPECIFIC_FEATURE_COLS
    + LEGACY_ROLLING_FEATURE_COLS
    + FPL_AVAILABILITY_FEATURE_COLS
)


def compute_correlations(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute per-feature correlations with CLV target."""
    if feature_cols is None:
        feature_cols = LINEUP_FEATURE_COLS
    target = df["target"].values
    rows: list[dict[str, Any]] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(target))
        n_valid = valid.sum()

        if n_valid < 20 or np.std(x[valid]) < 1e-10:
            rows.append(
                {
                    "feature": col,
                    "n": n_valid,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "univariate_r2": np.nan,
                }
            )
            continue

        r, p_r = stats.pearsonr(x[valid], target[valid])
        rho, p_s = stats.spearmanr(x[valid], target[valid])
        r2 = r**2

        rows.append(
            {
                "feature": col,
                "n": n_valid,
                "pearson_r": r,
                "pearson_p": p_r,
                "spearman_rho": rho,
                "spearman_p": p_s,
                "univariate_r2": r2,
            }
        )

    corr_df = pd.DataFrame(rows).set_index("feature")
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    return corr_df.sort_values("abs_r", ascending=False)


def multivariate_r2(
    df: pd.DataFrame, feature_cols: list[str] | None = None, cv_folds: int = 5
) -> dict[str, Any]:
    """Cross-validated linear regression R² with lineup features as ceiling estimate."""
    if feature_cols is None:
        feature_cols = LINEUP_FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]
    subset = df[available + ["target"]].dropna()

    if len(subset) < 30:
        return {"n": len(subset), "cv_r2_mean": np.nan, "cv_r2_std": np.nan}

    X = subset[available].values
    y = subset["target"].values

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")

    return {
        "n": len(subset),
        "cv_r2_mean": scores.mean(),
        "cv_r2_std": scores.std(),
        "cv_folds": cv_folds,
        "n_features": len(available),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_correlation_bars(corr_df: pd.DataFrame) -> None:
    """Bar plot of Pearson r by feature, color-coded by category."""
    plot_df = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))

    tendency_set = set(TENDENCY_FEATURE_COLS)
    match_set = set(MATCH_SPECIFIC_FEATURE_COLS)
    fpl_set = set(FPL_AVAILABILITY_FEATURE_COLS)

    colors = []
    for feat in plot_df.index:
        if feat in tendency_set:
            colors.append("#3498db")  # blue = tendency (bias-free)
        elif feat in match_set:
            colors.append("#e67e22")  # orange = match-specific
        elif feat in fpl_set:
            colors.append("#9b59b6")  # purple = FPL availability
        else:
            colors.append("#95a5a6")  # grey = legacy rolling

    ax.barh(plot_df.index, plot_df["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.05, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-0.05, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title("Lineup Feature Correlations with CLV Target")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="Tendency (bias-free)"),
        Patch(facecolor="#e67e22", label="Match-specific (post-announcement)"),
        Patch(facecolor="#9b59b6", label="FPL availability (pre-decision)"),
        Patch(facecolor="#95a5a6", label="Legacy rolling (look-ahead)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_bars.png", bbox_inches="tight")
    plt.close()


def plot_scatter_top_features(df: pd.DataFrame, corr_df: pd.DataFrame, n: int = 6) -> None:
    """Scatter plots for top-N correlated features."""
    top = corr_df.dropna(subset=["pearson_r"]).head(n)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, (feat, row) in zip(axes.flat, top.iterrows(), strict=False):
        if feat not in df.columns:
            continue
        valid = df[[feat, "target"]].dropna()
        ax.scatter(valid[feat], valid["target"], alpha=0.3, s=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("CLV target")
        ax.set_title(f"r={row['pearson_r']:.4f}, R²={row['univariate_r2']:.4f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_top_features.png", bbox_inches="tight")
    plt.close()


def plot_feature_correlation_matrix(df: pd.DataFrame) -> None:
    """Correlation matrix among lineup features."""
    available = [c for c in LINEUP_FEATURE_COLS if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Inter-Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_matrix.png", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline data loading
# ---------------------------------------------------------------------------


async def load_pipeline_data(
    config_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    """Load CLV targets and event metadata from the pipeline."""
    config = MLTrainingConfig.from_yaml(config_path)
    features_config = config.training.features
    data_config = config.training.data

    start_dt = datetime.combine(data_config.start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(data_config.end_date, datetime.max.time(), tzinfo=UTC)

    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session=session,
            start_date=start_dt,
            end_date=end_dt,
            status=EventStatus.FINAL,
        )
        print(f"Pipeline events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    # Build events DataFrame for matching
    event_rows: list[dict[str, Any]] = []
    for e in events:
        event_rows.append(
            {
                "event_id": e.id,
                "home_team": e.home_team,
                "away_team": e.away_team,
                "commence_time": e.commence_time,
                "match_date": e.commence_time.date(),
            }
        )
    events_df = pd.DataFrame(event_rows)

    return result.X, result.y, result.feature_names, result.event_ids, events_df


# ---------------------------------------------------------------------------
# Closing-tier target variance analysis
# ---------------------------------------------------------------------------


async def analyze_closing_tier_target_variance(early_targets: np.ndarray) -> None:
    """Measure CLV target variance in the post-lineup window (0-45 min before KO).

    Compares target magnitude in closing-tier snapshots against the early window
    (3-12h) to determine whether enough line movement remains post-lineup for
    match-specific features to be exploitable.
    """
    async with async_session_maker() as session:
        # Get all closing-tier snapshots for final EPL events within 45 min of kickoff
        stmt = (
            select(OddsSnapshot, Event)
            .join(Event, OddsSnapshot.event_id == Event.id)
            .where(
                Event.sport_key == "soccer_epl",
                Event.status == EventStatus.FINAL,
                OddsSnapshot.fetch_tier == "closing",
            )
        )
        results = (await session.execute(stmt)).all()

    # Group by event, filter to <=45 min before kickoff
    # For each event: collect qualifying snapshots and find the latest (actual close)
    events_snapshots: dict[str, list[tuple[OddsSnapshot, Event]]] = {}
    for snapshot, event in results:
        time_to_ko = event.commence_time - snapshot.snapshot_time
        if timedelta(0) <= time_to_ko <= timedelta(minutes=45):
            events_snapshots.setdefault(event.id, []).append((snapshot, event))

    if not events_snapshots:
        print("\nTARGET VARIANCE: POST-LINEUP WINDOW (0-45 min)")
        print("=" * 50)
        print("No closing-tier snapshots found within 45 min of kickoff.")
        return

    # For each event, find the latest closing snapshot (actual close) and
    # compute target for each qualifying snapshot against that close
    closing_targets: list[float] = []

    for event_id, snap_event_pairs in events_snapshots.items():
        # Sort by snapshot_time descending — first is the actual close
        snap_event_pairs.sort(key=lambda x: x[0].snapshot_time, reverse=True)
        closing_snapshot, event = snap_event_pairs[0]
        closing_odds = extract_odds_from_snapshot(closing_snapshot, event_id, market="h2h")

        # Compute target for every snapshot in this event (including close vs itself = 0)
        for snapshot, _ in snap_event_pairs:
            if snapshot.id == closing_snapshot.id:
                continue  # skip close vs itself
            snapshot_odds = extract_odds_from_snapshot(snapshot, event_id, market="h2h")
            target = calculate_devigged_bookmaker_target(
                snapshot_odds,
                closing_odds,
                event.home_team,
                event.away_team,
                bookmaker_key="bet365",
            )
            if target is not None:
                closing_targets.append(target)

    # If no multi-snapshot events, use close-vs-close (target=0) for single-snapshot events
    # but also report how many events only had one closing snapshot
    single_snapshot_events = sum(1 for pairs in events_snapshots.values() if len(pairs) == 1)

    closing_arr = np.array(closing_targets) if closing_targets else np.array([])
    early_arr = early_targets

    print("\n" + "=" * 50)
    print("TARGET VARIANCE: POST-LINEUP WINDOW (0-45 min)")
    print("=" * 50)
    print(f"Events with closing snapshots <=45min: {len(events_snapshots)}")
    print(f"  (single-snapshot events, no intra-window target: {single_snapshot_events})")

    if len(closing_arr) == 0:
        print("No intra-window target pairs found (all events have only one closing snapshot).")
        return

    closing_mean = np.mean(closing_arr)
    closing_std = np.std(closing_arr)
    closing_abs_mean = np.mean(np.abs(closing_arr))

    early_abs_mean = np.mean(np.abs(early_arr))
    early_std = np.std(early_arr)

    print(
        f"Target mean: {closing_mean:.6f}, std: {closing_std:.6f}, mean |target|: {closing_abs_mean:.6f}"
    )

    print("\nComparison with early window (3-12h):")
    print(
        f"  Early:   mean |target| = {early_abs_mean:.6f}, std = {early_std:.6f}  ({len(early_arr)} samples)"
    )
    print(
        f"  Closing: mean |target| = {closing_abs_mean:.6f}, std = {closing_std:.6f}  ({len(closing_arr)} samples)"
    )

    ratio = closing_abs_mean / early_abs_mean if early_abs_mean > 0 else float("inf")
    print(f"  Ratio (closing/early): {ratio:.2f}")

    print("\nTarget magnitude distribution (closing tier):")
    for thresh in (0.005, 0.010, 0.020):
        frac = np.mean(np.abs(closing_arr) > thresh) * 100
        print(f"  |target| > {thresh:.3f}: {frac:.1f}%")

    verdict = "Insufficient" if ratio < 0.25 else "Sufficient"
    print(f"\nVERDICT: {verdict} target variance for late-window prediction")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Lineup-delta signal check for CLV prediction")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading ESPN lineups...")
    lineup_df = load_lineups()

    # Compute lineup metrics
    print("\n" + "=" * 60)
    print("Computing lineup-delta metrics...")
    metrics_df = compute_lineup_metrics(lineup_df)

    # Add rolling averages
    metrics_df = _compute_rolling_metrics(metrics_df, window=3)

    # Compute GK changed separately
    gk_df = build_gk_changed(lineup_df)
    print(f"GK changed rows: {len(gk_df)}")

    # Summary stats
    print("\nMetric summary statistics:")
    for col in ["xi_jaccard", "xi_changes", "cumulative_starts_lost"]:
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            print(
                f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                f"min={vals.min():.1f}, max={vals.max():.1f}"
            )

    n_with_changes = (metrics_df["xi_changes"] > 0).sum()
    print(
        f"\n  Rows with xi_changes > 0: {n_with_changes}/{len(metrics_df)} "
        f"({100 * n_with_changes / len(metrics_df):.1f}%)"
    )

    # Load pipeline data and analyze closing-tier variance in one event loop
    async def _load_and_analyze(
        config_path: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
        result = await load_pipeline_data(config_path)
        await analyze_closing_tier_target_variance(result[1])  # result[1] = y
        return result

    print("\n" + "=" * 60)
    print("Loading pipeline CLV targets...")
    X, y, feature_names, event_ids, events_df = asyncio.run(_load_and_analyze(args.config))
    print(f"Pipeline: {len(y)} samples, {len(set(event_ids))} events")

    # Match lineup metrics to pipeline events
    print("\n" + "=" * 60)
    print("Matching lineup metrics to pipeline events...")
    joined_df = match_metrics_to_events(metrics_df, gk_df, event_ids, y, events_df)

    if joined_df.empty:
        print("ERROR: No events matched. Check team name normalization.")
        return

    # FPL availability features
    fpl_df = load_fpl_availability()
    if fpl_df is not None:
        print("\n" + "=" * 60)
        print("Computing FPL expected-disruption features...")
        fpl_features = compute_fpl_disruption_features(fpl_df, lineup_df, events_df)
        if not fpl_features.empty:
            joined_df = joined_df.merge(fpl_features, on="event_id", how="left")
            n_with_fpl = joined_df["home_expected_disruption"].notna().sum()
            print(f"FPL features joined: {n_with_fpl}/{len(joined_df)} events")
    else:
        print("\n" + "=" * 60)
        print("No FPL availability data found in data/fpl_availability/. Skipping.")

    # Save joined data
    joined_df.to_csv(OUTPUT_DIR / "joined_data.csv", index=False)

    # Compute correlations — four sections
    display_cols = ["n", "pearson_r", "pearson_p", "spearman_rho", "univariate_r2"]
    threshold = 0.05

    # --- Tendency features (bias-free) ---
    print("\n" + "=" * 60)
    print("TENDENCY FEATURES (bias-free, available at any decision tier)")
    print("=" * 60)
    tendency_corr = compute_correlations(joined_df, TENDENCY_FEATURE_COLS)
    print(tendency_corr[display_cols].to_string())
    tendency_mv = multivariate_r2(joined_df, TENDENCY_FEATURE_COLS)
    print(
        f"\nMultivariate CV R²: {tendency_mv['cv_r2_mean']:.6f} ± {tendency_mv['cv_r2_std']:.6f} "
        f"(n={tendency_mv['n']}, features={tendency_mv.get('n_features', 'N/A')})"
    )

    # --- Match-specific features (post-announcement) ---
    print("\n" + "=" * 60)
    print("MATCH-SPECIFIC FEATURES (require announced lineup, ~75min pre-KO)")
    print("=" * 60)
    match_corr = compute_correlations(joined_df, MATCH_SPECIFIC_FEATURE_COLS)
    print(match_corr[display_cols].to_string())
    match_mv = multivariate_r2(joined_df, MATCH_SPECIFIC_FEATURE_COLS)
    print(
        f"\nMultivariate CV R²: {match_mv['cv_r2_mean']:.6f} ± {match_mv['cv_r2_std']:.6f} "
        f"(n={match_mv['n']}, features={match_mv.get('n_features', 'N/A')})"
    )

    # --- FPL availability features (pre-decision tier) ---
    fpl_available = [c for c in FPL_AVAILABILITY_FEATURE_COLS if c in joined_df.columns]
    if fpl_available:
        print("\n" + "=" * 60)
        print("FPL AVAILABILITY FEATURES (pre-decision tier, ~24-48h before KO)")
        print("=" * 60)
        fpl_corr = compute_correlations(joined_df, FPL_AVAILABILITY_FEATURE_COLS)
        print(fpl_corr[display_cols].to_string())
        fpl_mv = multivariate_r2(joined_df, FPL_AVAILABILITY_FEATURE_COLS)
        print(
            f"\nMultivariate CV R²: {fpl_mv['cv_r2_mean']:.6f} ± {fpl_mv['cv_r2_std']:.6f} "
            f"(n={fpl_mv['n']}, features={fpl_mv.get('n_features', 'N/A')})"
        )
    else:
        fpl_corr = None

    # --- All features ---
    print("\n" + "=" * 60)
    print("ALL FEATURES")
    print("=" * 60)
    corr_df = compute_correlations(joined_df)
    print(corr_df[display_cols].to_string())
    mv = multivariate_r2(joined_df)
    print(
        f"\nMultivariate CV R²: {mv['cv_r2_mean']:.6f} ± {mv['cv_r2_std']:.6f} "
        f"(n={mv['n']}, features={mv.get('n_features', 'N/A')})"
    )

    # Go/no-go decision — based on tendency features specifically
    print("\n" + "=" * 60)
    print("GO / NO-GO DECISION")
    print("=" * 60)

    tendency_max_r = tendency_corr["abs_r"].max()
    match_max_r = match_corr["abs_r"].max()

    if np.isnan(tendency_max_r):
        print("RESULT: No valid tendency correlations. Cannot assess bias-free signal.")
    elif tendency_max_r > threshold:
        top_tendency = tendency_corr["abs_r"].idxmax()
        print(
            f"GO (tendency): Max |r| = {tendency_max_r:.4f} "
            f"(feature: {top_tendency}) > threshold {threshold}"
        )
        print("Bias-free tendency features carry signal — safe for any decision tier.")
    else:
        print(f"NO-GO (tendency): Max |r| = {tendency_max_r:.4f} <= threshold {threshold}")
        print("Tendency features alone do not carry sufficient signal.")

    if not np.isnan(match_max_r) and match_max_r > threshold:
        top_match = match_corr["abs_r"].idxmax()
        print(
            f"\nNote: Match-specific features have signal (max |r| = {match_max_r:.4f}, "
            f"feature: {top_match}), but require post-lineup-announcement execution "
            f"(~75min pre-KO)."
        )

    if fpl_corr is not None:
        fpl_max_r = fpl_corr["abs_r"].max()
        if not np.isnan(fpl_max_r):
            if fpl_max_r > threshold:
                top_fpl = fpl_corr["abs_r"].idxmax()
                print(
                    f"\nGO (FPL availability): Max |r| = {fpl_max_r:.4f} "
                    f"(feature: {top_fpl}) > threshold {threshold}"
                )
                print("FPL expected-disruption carries signal — proceed to feature group.")
            else:
                print(
                    f"\nNO-GO (FPL availability): Max |r| = {fpl_max_r:.4f} <= threshold {threshold}"
                )
                print("FPL availability features show insufficient signal for CLV prediction.")

    # Save correlation results
    corr_df.to_csv(OUTPUT_DIR / "correlations.csv")

    # Plots
    print("\nGenerating plots...")
    plot_correlation_bars(corr_df)
    plot_scatter_top_features(joined_df, corr_df)
    plot_feature_correlation_matrix(joined_df)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
