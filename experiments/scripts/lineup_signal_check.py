"""Phase A: Lineup-delta signal check for CLV prediction.

Loads ESPN lineup CSVs and FBref player stats, matches them to pipeline events
by team name + date, computes lineup-delta metrics (Jaccard similarity,
XI changes, GK changed, weighted disruption, goals/minutes lost), and
correlates each against the observed devigged bet365 CLV target.

Outputs: correlation matrix, scatter plots, per-feature univariate R².
Results saved to experiments/results/lineup_signal_check/.

Usage:
    uv run python experiments/scripts/lineup_signal_check.py
    uv run python experiments/scripts/lineup_signal_check.py --config experiments/configs/xgboost_epl_combined_standings_tuning.yaml
"""

from __future__ import annotations

import asyncio
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from scipy import stats
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "lineup_signal_check"

LINEUP_DIR = PROJECT_ROOT / "data" / "espn_lineups"
FBREF_DIR = PROJECT_ROOT / "data" / "fbref_player_stats"

DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "xgboost_epl_combined_standings_tuning.yaml"

# FBref team name -> canonical (same as ingest script, kept here for self-containment)
FBREF_TEAM_NAME_MAP: dict[str, str] = {
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Newcastle Utd": "Newcastle",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'ham Forest": "Nottingham",
    "Nottingham Forest": "Nottingham",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth",
}

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _strip_accents(s: str) -> str:
    """Remove Unicode accents for fuzzy name matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


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


def load_fbref_stats() -> pd.DataFrame:
    """Load all FBref player stats CSVs into a single DataFrame."""
    csv_files = sorted(FBREF_DIR.glob("player_stats_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No FBref CSVs found in {FBREF_DIR}")

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    # Normalize team names
    df["team"] = df["team"].map(lambda t: FBREF_TEAM_NAME_MAP.get(t, t))

    # Create normalized player name for matching
    df["player_norm"] = df["player"].apply(lambda n: _strip_accents(str(n)).lower().strip())

    print(f"Loaded {len(df)} FBref player rows from {len(csv_files)} CSVs")
    print(f"  Teams: {sorted(df['team'].unique())}")
    return df


def _build_fbref_lookup(
    fbref_df: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Build lookup: (team, player_norm, season) -> stats dict."""
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in fbref_df.iterrows():
        key = (row["team"], row["player_norm"], row["season"])
        lookup[key] = {
            "minutes": row.get("minutes", 0) or 0,
            "goals": row.get("goals", 0) or 0,
            "games_starts": row.get("games_starts", 0) or 0,
            "nineties": row.get("nineties", 0) or 0,
        }
    return lookup


def _season_label_from_date(d: datetime) -> str:
    """Derive season label (e.g. '2024-25') from a datetime."""
    year = d.year
    month = d.month
    if month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


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
    fbref_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Compute per-team per-match lineup-delta metrics.

    For each team in each match, compare the starting XI to the previous
    match's starting XI. Returns a DataFrame with one row per team per match.
    """
    # Build per-match starting XIs: list of (team, match_date, ..., players)
    # We need player_id -> player_name mapping, so collect as list of tuples
    match_xis: list[dict[str, Any]] = []

    for (team, mdate, ht, at, dt_val), grp in lineup_df.groupby(
        ["team", "match_date", "home_team", "away_team", "datetime"]
    ):
        id_to_name: dict[str, str] = {}
        for _, r in grp.iterrows():
            id_to_name[str(r["player_id"])] = r["player_name"]
        match_xis.append(
            {
                "team": team,
                "match_date": mdate,
                "home_team": ht,
                "away_team": at,
                "datetime": dt_val,
                "id_to_name": id_to_name,
            }
        )

    # Sort chronologically per team
    match_xis.sort(key=lambda x: (x["team"], x["datetime"]))

    rows: list[dict[str, Any]] = []
    # team -> previous match's {player_id: player_name}
    prev_xi: dict[str, dict[str, str]] = {}

    for match in match_xis:
        team: str = match["team"]
        match_date = match["match_date"]
        dt: datetime = match["datetime"]
        home_team: str = match["home_team"]
        away_team: str = match["away_team"]
        current: dict[str, str] = match["id_to_name"]
        current_ids = set(current.keys())
        season = _season_label_from_date(dt)

        prev = prev_xi.get(team)

        if prev is not None:
            prev_ids = set(prev.keys())
            xi_jaccard = _jaccard(current_ids, prev_ids)
            xi_changes = len(prev_ids - current_ids)

            # Names of players dropped from previous XI
            dropped_names = [prev[pid] for pid in prev_ids - current_ids]

            # Match dropped players to FBref stats for weighted metrics
            weighted_change = 0.0
            goals_lost = 0.0
            minutes_lost = 0.0

            for pname in dropped_names:
                pname_norm = _strip_accents(pname).lower().strip()
                stats_row = fbref_lookup.get((team, pname_norm, season))
                if stats_row is None:
                    continue
                minutes_val = float(stats_row.get("minutes", 0) or 0)
                nineties_val = float(stats_row.get("nineties", 0) or 0)
                goals_val = float(stats_row.get("goals", 0) or 0)

                weighted_change += nineties_val
                goals_lost += goals_val
                minutes_lost += minutes_val

            row: dict[str, Any] = {
                "team": team,
                "match_date": match_date,
                "datetime": dt,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "xi_jaccard": xi_jaccard,
                "xi_changes": float(xi_changes),
                "weighted_xi_change": weighted_change,
                "goals_lost": goals_lost,
                "minutes_lost": minutes_lost,
            }
            rows.append(row)

        # Store current XI for next iteration
        prev_xi[team] = current

    metrics_df = pd.DataFrame(rows)
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
    metric_cols = ["xi_jaccard", "xi_changes", "weighted_xi_change", "goals_lost", "minutes_lost"]

    for col in metric_cols:
        metrics_df[f"{col}_roll{window}"] = metrics_df.groupby("team")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    return metrics_df


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
        "weighted_xi_change",
        "goals_lost",
        "minutes_lost",
        "xi_jaccard_roll3",
        "xi_changes_roll3",
        "weighted_xi_change_roll3",
        "goals_lost_roll3",
        "minutes_lost_roll3",
    ]

    joined_rows: list[dict[str, Any]] = []
    matched = 0
    unmatched = 0

    for (home, away, date_str), (eid, target) in event_lookup.items():
        home_metrics = metrics_lookup.get((home, date_str))
        away_metrics = metrics_lookup.get((away, date_str))

        if home_metrics is None and away_metrics is None:
            unmatched += 1
            continue

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
        row_dict["home_gk_changed"] = gk_lookup.get((home, date_str), np.nan)
        row_dict["away_gk_changed"] = gk_lookup.get((away, date_str), np.nan)

        joined_rows.append(row_dict)
        matched += 1

    print(f"Event matching: {matched} matched, {unmatched} unmatched")
    return pd.DataFrame(joined_rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

LINEUP_FEATURE_COLS = [
    "home_xi_jaccard",
    "away_xi_jaccard",
    "diff_xi_jaccard",
    "home_xi_changes",
    "away_xi_changes",
    "diff_xi_changes",
    "home_gk_changed",
    "away_gk_changed",
    "home_weighted_xi_change",
    "away_weighted_xi_change",
    "diff_weighted_xi_change",
    "home_goals_lost",
    "away_goals_lost",
    "diff_goals_lost",
    "home_minutes_lost",
    "away_minutes_lost",
    "diff_minutes_lost",
    "home_xi_jaccard_roll3",
    "away_xi_jaccard_roll3",
    "diff_xi_jaccard_roll3",
    "home_xi_changes_roll3",
    "away_xi_changes_roll3",
    "diff_xi_changes_roll3",
    "home_weighted_xi_change_roll3",
    "away_weighted_xi_change_roll3",
    "diff_weighted_xi_change_roll3",
    "home_goals_lost_roll3",
    "away_goals_lost_roll3",
    "diff_goals_lost_roll3",
    "home_minutes_lost_roll3",
    "away_minutes_lost_roll3",
    "diff_minutes_lost_roll3",
]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature correlations with CLV target."""
    target = df["target"].values
    rows: list[dict[str, Any]] = []

    for col in LINEUP_FEATURE_COLS:
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


def multivariate_r2(df: pd.DataFrame) -> dict[str, float]:
    """Fit simple linear regression with all lineup features to get ceiling R²."""
    available = [c for c in LINEUP_FEATURE_COLS if c in df.columns]
    subset = df[available + ["target"]].dropna()

    if len(subset) < 30:
        return {"n": len(subset), "r2": np.nan}

    X = subset[available].values
    y = subset["target"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    return {"n": len(subset), "r2": r2, "n_features": len(available)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_correlation_bars(corr_df: pd.DataFrame) -> None:
    """Bar plot of Pearson r by feature."""
    plot_df = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in plot_df["pearson_r"]]
    ax.barh(plot_df.index, plot_df["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.05, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-0.05, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title("Lineup-Delta Feature Correlations with CLV Target")
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

    print("\nLoading FBref player stats...")
    fbref_df = load_fbref_stats()
    fbref_lookup = _build_fbref_lookup(fbref_df)

    print(f"\nFBref lookup size: {len(fbref_lookup)} (team, player, season) entries")

    # Compute lineup metrics
    print("\n" + "=" * 60)
    print("Computing lineup-delta metrics...")
    metrics_df = compute_lineup_metrics(lineup_df, fbref_lookup)

    # Add rolling averages
    metrics_df = _compute_rolling_metrics(metrics_df, window=3)

    # Compute GK changed separately
    gk_df = build_gk_changed(lineup_df)
    print(f"GK changed rows: {len(gk_df)}")

    # Summary stats
    print("\nMetric summary statistics:")
    for col in ["xi_jaccard", "xi_changes", "weighted_xi_change", "goals_lost", "minutes_lost"]:
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            print(
                f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                f"min={vals.min():.1f}, max={vals.max():.1f}"
            )

    # FBref match rate
    n_with_weight = (metrics_df["weighted_xi_change"] > 0).sum()
    n_with_changes = (metrics_df["xi_changes"] > 0).sum()
    print(
        f"\n  Rows with xi_changes > 0: {n_with_changes}/{len(metrics_df)} "
        f"({100 * n_with_changes / len(metrics_df):.1f}%)"
    )
    print(
        f"  Rows with FBref-weighted changes: {n_with_weight}/{n_with_changes} "
        f"({100 * n_with_weight / max(n_with_changes, 1):.1f}% of changed lineups)"
    )

    # Load pipeline data
    print("\n" + "=" * 60)
    print("Loading pipeline CLV targets...")
    X, y, feature_names, event_ids, events_df = asyncio.run(load_pipeline_data(args.config))
    print(f"Pipeline: {len(y)} samples, {len(set(event_ids))} events")

    # Match lineup metrics to pipeline events
    print("\n" + "=" * 60)
    print("Matching lineup metrics to pipeline events...")
    joined_df = match_metrics_to_events(metrics_df, gk_df, event_ids, y, events_df)

    if joined_df.empty:
        print("ERROR: No events matched. Check team name normalization.")
        return

    # Save joined data
    joined_df.to_csv(OUTPUT_DIR / "joined_data.csv", index=False)

    # Compute correlations
    print("\n" + "=" * 60)
    print("Computing correlations with CLV target...")
    corr_df = compute_correlations(joined_df)

    print("\nCorrelation results (sorted by |r|):")
    print(corr_df[["n", "pearson_r", "pearson_p", "spearman_rho", "univariate_r2"]].to_string())

    # Multivariate ceiling
    mv = multivariate_r2(joined_df)
    print(
        f"\nMultivariate linear regression ceiling: R²={mv['r2']:.6f} "
        f"(n={mv['n']}, features={mv.get('n_features', 'N/A')})"
    )

    # Go/no-go decision
    print("\n" + "=" * 60)
    max_abs_r = corr_df["abs_r"].max()
    threshold = 0.05
    if np.isnan(max_abs_r):
        print("RESULT: No valid correlations computed. Cannot make go/no-go decision.")
    elif max_abs_r > threshold:
        top_feat = corr_df["abs_r"].idxmax()
        print(f"GO: Max |r| = {max_abs_r:.4f} (feature: {top_feat}) > threshold {threshold}")
        print("Proceed to Phase B: build epl_lineup feature group.")
    else:
        print(f"NO-GO: Max |r| = {max_abs_r:.4f} <= threshold {threshold}")
        print("Lineup-delta features show insufficient signal for CLV prediction.")

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
