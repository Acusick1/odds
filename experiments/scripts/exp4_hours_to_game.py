"""Experiment 4: Hours-to-Game Effect.

Analyse how target variance, feature-target correlations, and model
performance vary with decision time (hours before game).

Uses Odds API data (~1K events, 15+ snapshots each) with time_range sampling
to get multiple samples per event at different hours before game.

Key questions:
  1. How does target (CLV delta) variance change with hours before game?
  2. At what hour does the sharp-retail divergence signal peak?
  3. Where does a simple XGBoost model achieve the best R²?
  4. Is there a sweet spot — strong signal but line hasn't moved yet?

Outputs saved to experiments/results/exp4_hours_to_game/
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.training.config import MLTrainingConfig, SamplingConfig
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp4_hours_to_game"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "xgboost_bet365_tuning_best.yaml"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120

# Hour bins for analysis
HOUR_BINS = [0, 3, 6, 9, 12, 18, 24, 36, 48]
HOUR_LABELS = ["0-3h", "3-6h", "6-9h", "9-12h", "12-18h", "18-24h", "24-36h", "36-48h"]


def make_xgboost() -> XGBRegressor:
    """Light XGBoost for per-bin evaluation (not heavily tuned)."""
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


async def load_data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load Odds API data with wide time_range sampling for multi-hour coverage."""
    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    features_config = config.training.features
    data_config = config.training.data

    # Override to Odds API data with Pinnacle as sharp book
    features_config.sharp_bookmakers = ["pinnacle"]
    features_config.retail_bookmakers = ["fanduel", "draftkings", "betmgm"]
    features_config.target_bookmaker = "pinnacle"

    # Wide time_range sampling: 1-48h, up to 10 samples per event
    features_config.sampling = SamplingConfig(
        strategy="time_range",
        min_hours=1.0,
        max_hours=48.0,
        max_samples_per_event=10,
    )

    # Odds API Pinnacle coverage starts Oct 2025 (Mar-Apr 2025 has <10%)
    from datetime import date

    start_dt = datetime.combine(date(2025, 10, 1), datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(data_config.end_date, datetime.max.time(), tzinfo=UTC)

    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session=session,
            start_date=start_dt,
            end_date=end_dt,
            status=EventStatus.FINAL,
            data_source="oddsapi",
        )
        print(f"Events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    return result.X, result.y, result.feature_names, result.event_ids


def analyse_target_by_hour(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute target statistics per hour bin."""
    rows = []
    for label in HOUR_LABELS:
        mask = df["hour_bin"] == label
        subset = df[mask]
        if len(subset) < 5:
            continue
        y = subset["target"].values
        rows.append(
            {
                "hour_bin": label,
                "n_samples": len(subset),
                "n_events": subset["event_id"].nunique(),
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "target_var": float(np.var(y)),
                "target_abs_mean": float(np.mean(np.abs(y))),
                "target_median": float(np.median(y)),
            }
        )
    return pd.DataFrame(rows)


def analyse_correlations_by_hour(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute feature-target Pearson correlations per hour bin."""
    rows = []
    for label in HOUR_LABELS:
        mask = df["hour_bin"] == label
        subset = df[mask]
        if len(subset) < 20:
            continue

        y = subset["target"].values
        for feat in feature_cols:
            x = subset[feat].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 10:
                continue
            r, p = stats.pearsonr(x[valid], y[valid])
            rows.append(
                {
                    "hour_bin": label,
                    "feature": feat,
                    "correlation": r,
                    "p_value": p,
                    "n_valid": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def train_per_bin(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Train XGBoost per hour bin using chronological 70/30 split within each bin."""
    rows = []
    for label in HOUR_LABELS:
        mask = df["hour_bin"] == label
        subset = df[mask].copy()
        if len(subset) < 30:
            continue

        X = subset[feature_cols].values
        y = subset["target"].values

        # Drop NaN columns for this bin
        valid_cols = ~np.any(np.isnan(X), axis=0)
        X_clean = X[:, valid_cols]
        if X_clean.shape[1] == 0:
            continue

        # Chronological split (data is already time-sorted via event ordering)
        split_idx = int(len(X_clean) * 0.7)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_test) < 5:
            continue

        model = make_xgboost()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Baseline: predict training mean
        baseline_pred = np.full_like(y_test, y_train.mean())

        rows.append(
            {
                "hour_bin": label,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_clean.shape[1],
                "test_r2": r2_score(y_test, y_pred),
                "test_mse": mean_squared_error(y_test, y_pred),
                "baseline_mse": mean_squared_error(y_test, baseline_pred),
                "target_std_test": float(np.std(y_test)),
            }
        )

    return pd.DataFrame(rows)


def plot_target_variance(target_stats: pd.DataFrame) -> None:
    """Plot target variance and absolute mean vs hours to game."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(target_stats["hour_bin"], target_stats["target_std"], color="steelblue", alpha=0.8)
    ax1.set_xlabel("Hours Before Game")
    ax1.set_ylabel("Target Std Dev")
    ax1.set_title("CLV Delta Standard Deviation by Decision Time")
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(target_stats["hour_bin"], target_stats["target_abs_mean"], color="coral", alpha=0.8)
    ax2.set_xlabel("Hours Before Game")
    ax2.set_ylabel("Mean |CLV Delta|")
    ax2.set_title("Mean Absolute CLV Delta by Decision Time")
    ax2.tick_params(axis="x", rotation=30)

    # Add sample counts
    for ax, col in [(ax1, "target_std"), (ax2, "target_abs_mean")]:
        for _, row in target_stats.iterrows():
            ax.annotate(
                f"n={row['n_samples']:.0f}",
                (row["hour_bin"], row[col]),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Experiment 4: Target Properties by Decision Time", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_variance.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved target_variance.png")


def plot_key_correlations(corr_df: pd.DataFrame) -> None:
    """Plot correlations of key features across hour bins."""
    key_features = [
        "tab_retail_sharp_diff",
        "tab_sharp_prob",
        "tab_consensus_prob",
        "tab_num_bookmakers",
    ]
    available = [f for f in key_features if f in corr_df["feature"].values]

    if not available:
        print("  No key features found in correlation data, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for feat in available:
        feat_data = corr_df[corr_df["feature"] == feat].copy()
        # Ensure hour_bin ordering
        feat_data = (
            feat_data.set_index("hour_bin").reindex(HOUR_LABELS).dropna(subset=["correlation"])
        )
        ax.plot(feat_data.index, feat_data["correlation"], marker="o", label=feat, linewidth=2)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Hours Before Game")
    ax.set_ylabel("Pearson Correlation with CLV Delta")
    ax.set_title("Feature-Target Correlation by Decision Time")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_correlations_by_hour.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved feature_correlations_by_hour.png")


def plot_model_performance(perf_df: pd.DataFrame) -> None:
    """Plot XGBoost R² and MSE by hour bin."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(perf_df["hour_bin"], perf_df["test_r2"], color="forestgreen", alpha=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="R²=0 (predict mean)")
    ax1.set_xlabel("Hours Before Game")
    ax1.set_ylabel("Test R²")
    ax1.set_title("XGBoost R² by Decision Time")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=30)

    x = np.arange(len(perf_df))
    width = 0.35
    ax2.bar(
        x - width / 2, perf_df["test_mse"], width, label="XGBoost", color="forestgreen", alpha=0.8
    )
    ax2.bar(
        x + width / 2,
        perf_df["baseline_mse"],
        width,
        label="Predict Mean",
        color="lightcoral",
        alpha=0.8,
    )
    ax2.set_xlabel("Hours Before Game")
    ax2.set_ylabel("Test MSE")
    ax2.set_title("XGBoost vs Baseline MSE by Decision Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(perf_df["hour_bin"], rotation=30, ha="right")
    ax2.legend()

    plt.suptitle("Experiment 4: Model Performance by Decision Time", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved model_performance.png")


def plot_sample_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of samples across hours."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = df["hours_until_event"].values
    ax.hist(hours, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Hours Until Game")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Samples by Hours Before Game")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved sample_distribution.png")


def evaluate_pooled_model(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Train one XGBoost on all data (chrono split), evaluate predictions per hour bin.

    This avoids the small-sample problem of per-bin training.
    """
    # Chronological 70/30 split across all data
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    # Handle NaN columns
    valid_cols = ~np.any(np.isnan(X_train), axis=0)
    X_train_clean = X_train[:, valid_cols]
    X_test_clean = X_test[:, valid_cols]

    model = make_xgboost()
    model.fit(X_train_clean, y_train)
    y_pred = model.predict(X_test_clean)

    overall_r2 = r2_score(y_test, y_pred)
    overall_mse = mean_squared_error(y_test, y_pred)
    print(f"  Pooled model — overall test R²={overall_r2:.4f}, MSE={overall_mse:.6f}")
    print(f"  Train: {len(X_train_clean)}, Test: {len(X_test_clean)}")

    # Evaluate per hour bin on test set
    test_df = test_df.copy()
    test_df["y_pred"] = y_pred
    test_df["y_true"] = y_test

    rows = []
    for label in HOUR_LABELS:
        mask = test_df["hour_bin"] == label
        subset = test_df[mask]
        if len(subset) < 5:
            continue
        y_true_bin = subset["y_true"].values
        y_pred_bin = subset["y_pred"].values
        baseline_pred = np.full_like(y_true_bin, y_train.mean())

        rows.append(
            {
                "hour_bin": label,
                "n_test": len(subset),
                "r2": r2_score(y_true_bin, y_pred_bin),
                "mse": mean_squared_error(y_true_bin, y_pred_bin),
                "baseline_mse": mean_squared_error(y_true_bin, baseline_pred),
                "mean_abs_pred": float(np.mean(np.abs(y_pred_bin))),
                "target_std": float(np.std(y_true_bin)),
            }
        )

    return pd.DataFrame(rows)


def plot_pooled_performance(pooled_df: pd.DataFrame) -> None:
    """Plot pooled model R² and MSE ratio by hour bin."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["forestgreen" if r > 0 else "lightcoral" for r in pooled_df["r2"]]
    ax1.bar(pooled_df["hour_bin"], pooled_df["r2"], color=colors, alpha=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Hours Before Game")
    ax1.set_ylabel("Test R²")
    ax1.set_title("Pooled XGBoost R² by Decision Time")
    ax1.tick_params(axis="x", rotation=30)
    for _, row in pooled_df.iterrows():
        ax1.annotate(
            f"n={row['n_test']:.0f}",
            (row["hour_bin"], row["r2"]),
            ha="center",
            va="bottom" if row["r2"] >= 0 else "top",
            fontsize=8,
        )

    # MSE improvement over baseline
    pooled_df = pooled_df.copy()
    pooled_df["mse_ratio"] = pooled_df["mse"] / pooled_df["baseline_mse"]
    colors2 = ["forestgreen" if r < 1 else "lightcoral" for r in pooled_df["mse_ratio"]]
    ax2.bar(pooled_df["hour_bin"], pooled_df["mse_ratio"], color=colors2, alpha=0.8)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Baseline")
    ax2.set_xlabel("Hours Before Game")
    ax2.set_ylabel("MSE / Baseline MSE")
    ax2.set_title("MSE Ratio (< 1.0 = model beats baseline)")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)

    plt.suptitle("Experiment 4: Pooled Model Performance by Decision Time", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pooled_model_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved pooled_model_performance.png")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading Odds API data with wide time-range sampling...")
    X, y, feature_names, event_ids = await load_data()

    n_events = len(set(event_ids))
    print(f"Loaded {len(X)} samples, {n_events} events, {len(feature_names)} features")
    print(f"Feature names: {feature_names}")

    # Build DataFrame for analysis
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["event_id"] = event_ids

    # hours_until_event is the last feature from XGBoostAdapter
    hours_col = "hours_until_event"
    if hours_col not in df.columns:
        # Fallback: check for similar name
        hour_candidates = [c for c in df.columns if "hour" in c.lower()]
        if hour_candidates:
            hours_col = hour_candidates[0]
            print(f"  Using '{hours_col}' as hours column")
        else:
            raise ValueError(f"No hours column found. Columns: {feature_names}")

    print(f"\nHours range: {df[hours_col].min():.1f} - {df[hours_col].max():.1f}")
    print(f"Hours mean: {df[hours_col].mean():.1f}, median: {df[hours_col].median():.1f}")

    # Assign hour bins
    df["hour_bin"] = pd.cut(
        df[hours_col],
        bins=HOUR_BINS,
        labels=HOUR_LABELS,
        include_lowest=True,
    )

    # --- Plot sample distribution ---
    print("\nPlotting sample distribution...")
    plot_sample_distribution(df)

    # --- Target variance by hour ---
    print("\nAnalysing target variance by hour...")
    target_stats = analyse_target_by_hour(df)
    print(target_stats.to_string(index=False))
    target_stats.to_csv(OUTPUT_DIR / "target_stats.csv", index=False)

    print("\nPlotting target variance...")
    plot_target_variance(target_stats)

    # --- Feature-target correlations by hour ---
    feature_cols = [c for c in feature_names if c != hours_col]
    print(f"\nAnalysing correlations for {len(feature_cols)} features across hour bins...")
    corr_df = analyse_correlations_by_hour(df, feature_cols)
    corr_df.to_csv(OUTPUT_DIR / "correlations_by_hour.csv", index=False)

    # Summary: strongest correlation per bin
    if not corr_df.empty:
        print("\nStrongest correlation per bin:")
        for label in HOUR_LABELS:
            bin_corrs = corr_df[corr_df["hour_bin"] == label]
            if bin_corrs.empty:
                continue
            best = bin_corrs.loc[bin_corrs["correlation"].abs().idxmax()]
            print(
                f"  {label}: {best['feature']} r={best['correlation']:+.4f} "
                f"(p={best['p_value']:.4f}, n={best['n_valid']:.0f})"
            )

        print("\nPlotting key feature correlations...")
        plot_key_correlations(corr_df)

    # --- Model performance by hour ---
    print(f"\nTraining XGBoost per hour bin ({len(feature_cols)} features)...")
    perf_df = train_per_bin(df, feature_cols)
    if not perf_df.empty:
        print(perf_df.to_string(index=False))
        perf_df.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)

        print("\nPlotting model performance...")
        plot_model_performance(perf_df)
    else:
        print("  Insufficient data for per-bin model training")

    # --- Pooled model: train on all data, evaluate per bin ---
    print("\nTraining pooled XGBoost on all data, evaluating per hour bin...")
    pooled_df = evaluate_pooled_model(df, feature_cols)
    if not pooled_df.empty:
        print(pooled_df.to_string(index=False))
        pooled_df.to_csv(OUTPUT_DIR / "pooled_model_by_hour.csv", index=False)

        print("\nPlotting pooled model performance...")
        plot_pooled_performance(pooled_df)

    # --- Summary ---
    print(f"\nAll outputs in {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
