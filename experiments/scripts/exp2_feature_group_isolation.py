"""Experiment 2: Feature Group Isolation.

Train simple models (Ridge regression, shallow XGBoost, and LSTM) on each
feature group to determine which groups carry predictive signal for the
devigged Pinnacle CLV delta target.

Feature groups tested (Ridge + XGBoost):
  1. Tabular only (tab_* features)
  2. Trajectory only (traj_* features)
  3. PM + cross-source only (pm_* + xsrc_* features)
  4. Sharp-retail divergence only (hand-picked subset)
  5. All features combined (baseline)

LSTM (sequence model):
  - Uses SequenceFeatures (per-timestep odds representation), a different
    feature space from the tabular groups above — sequence group not comparable
    to tabular groups directly, but answers: does temporal modeling help?

Uses group timeseries CV (event-level splits) matching exp1 methodology.

Outputs saved to experiments/results/exp2_feature_group_isolation/
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2_feature_group_isolation"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "xgboost_cross_source_v1.yaml"
LSTM_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "lstm_line_movement_tuning_best.yaml"
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120

# Sharp-retail divergence features (strongest theoretical prior from exp1).
# Note: exp1 found tab_retail_sharp_diff_home/away as top features; PR #135
# merged these into tab_retail_sharp_diff (home/away split was a structural duplicate).
SHARP_RETAIL_FEATURES = [
    "tab_retail_sharp_diff",
    "traj_sharp_retail_divergence_trend",
    "traj_max_prob_decrease",
]

# Feature group definitions by prefix
FEATURE_GROUPS: dict[str, list[str]] = {
    "tabular": ["tab_"],
    "trajectory": ["traj_"],
    "pm_cross_source": ["pm_", "xsrc_"],
}


def select_features_by_group(feature_names: list[str], group_name: str) -> list[int]:
    """Return column indices for features belonging to a group."""
    if group_name == "sharp_retail":
        return [i for i, n in enumerate(feature_names) if n in SHARP_RETAIL_FEATURES]
    if group_name == "all":
        return list(range(len(feature_names)))
    if group_name == "all_no_pm":
        return [
            i
            for i, n in enumerate(feature_names)
            if not n.startswith("pm_") and not n.startswith("xsrc_")
        ]

    prefixes = FEATURE_GROUPS.get(group_name, [])
    return [i for i, n in enumerate(feature_names) if any(n.startswith(p) for p in prefixes)]


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    model_factory: Callable[[], RegressorMixin],
    n_folds: int = 3,
) -> dict[str, float]:
    """Run group timeseries CV (walk-forward on event boundaries).

    Uses TimeSeriesSplit on unique event IDs — same logic as run_cv()
    with cv_method='group_timeseries'. Always trains on chronologically
    earlier events and validates on later events.
    """
    unique_events = list(dict.fromkeys(event_ids))
    event_splitter = TimeSeriesSplit(n_splits=n_folds)
    event_indices = np.arange(len(unique_events))

    fold_metrics: list[dict[str, float]] = []

    for ev_train_idx, ev_val_idx in event_splitter.split(event_indices):
        train_events = {unique_events[i] for i in ev_train_idx}
        val_events = {unique_events[i] for i in ev_val_idx}
        train_idx = np.where([eid in train_events for eid in event_ids])[0]
        val_idx = np.where([eid in val_events for eid in event_ids])[0]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        fold_metrics.append(
            {
                "r2": r2_score(y_val, y_pred),
                "mse": mean_squared_error(y_val, y_pred),
                "mae": mean_absolute_error(y_val, y_pred),
                "n_val": len(y_val),
            }
        )

    total_n = sum(f["n_val"] for f in fold_metrics)
    r2_vals = [f["r2"] for f in fold_metrics]
    mse_vals = [f["mse"] for f in fold_metrics]
    mae_vals = [f["mae"] for f in fold_metrics]

    return {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "mse_mean": float(np.mean(mse_vals)),
        "mse_std": float(np.std(mse_vals)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "fold_r2s": r2_vals,
        "fold_mses": mse_vals,
        "n_total": total_n,
    }


def make_ridge() -> Ridge:
    return Ridge(alpha=1.0)


def make_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


async def load_data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load training data using existing pipeline."""
    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
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
        print(f"Events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    return result.X, result.y, result.feature_names, result.event_ids


def run_group_experiments(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    event_ids: np.ndarray,
) -> pd.DataFrame:
    """Run Ridge and XGBoost on each feature group."""
    groups_to_test = [
        "tabular",
        "trajectory",
        "pm_cross_source",
        "sharp_retail",
        "all_no_pm",
        "all",
    ]

    models = {
        "ridge": make_ridge,
        "xgboost": make_xgboost,
    }

    results = []
    for group_name in groups_to_test:
        col_idx = select_features_by_group(feature_names, group_name)
        if not col_idx:
            print(f"  Skipping {group_name}: no features found")
            continue

        group_feature_names = [feature_names[i] for i in col_idx]
        X_group = X[:, col_idx]

        # Check for zero-variance columns after subsetting
        variances = np.var(X_group, axis=0)
        nonzero_mask = variances > 1e-10
        n_dropped = (~nonzero_mask).sum()
        if n_dropped > 0:
            X_group = X_group[:, nonzero_mask]
            group_feature_names = [
                n for n, m in zip(group_feature_names, nonzero_mask, strict=False) if m
            ]

        n_features = X_group.shape[1]
        print(f"\n  {group_name}: {n_features} features ({n_dropped} zero-variance dropped)")

        for model_name, model_factory in models.items():
            metrics = cross_validate(X_group, y, event_ids, model_factory, n_folds=3)
            results.append(
                {
                    "group": group_name,
                    "model": model_name,
                    "n_features": n_features,
                    **metrics,
                }
            )
            print(
                f"    {model_name:8s}: R²={metrics['r2_mean']:+.4f}±{metrics['r2_std']:.4f}  "
                f"MSE={metrics['mse_mean']:.6f}  MAE={metrics['mae_mean']:.4f}"
            )

    return pd.DataFrame(results)


def compute_baselines(y: np.ndarray, event_ids: np.ndarray) -> dict[str, float]:
    """Compute naive baselines using walk-forward event splits."""
    unique_events = list(dict.fromkeys(event_ids))
    event_splitter = TimeSeriesSplit(n_splits=3)
    event_indices = np.arange(len(unique_events))

    r2s, mses, maes = [], [], []
    for ev_train_idx, ev_val_idx in event_splitter.split(event_indices):
        train_events = {unique_events[i] for i in ev_train_idx}
        val_events = {unique_events[i] for i in ev_val_idx}
        train_idx = np.where([eid in train_events for eid in event_ids])[0]
        val_idx = np.where([eid in val_events for eid in event_ids])[0]

        y_train, y_val = y[train_idx], y[val_idx]
        pred = np.full_like(y_val, y_train.mean())
        r2s.append(r2_score(y_val, pred))
        mses.append(mean_squared_error(y_val, pred))
        maes.append(mean_absolute_error(y_val, pred))

    return {
        "baseline_r2": float(np.mean(r2s)),
        "baseline_mse": float(np.mean(mses)),
        "baseline_mae": float(np.mean(maes)),
    }


def plot_group_comparison(results_df: pd.DataFrame, baselines: dict) -> None:
    """Bar chart comparing R² and MSE across groups and models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    groups = results_df["group"].unique()
    models = results_df["model"].unique()
    x = np.arange(len(groups))
    width = 0.35

    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
        r2_vals = [
            model_data[model_data["group"] == g]["r2_mean"].values[0]
            if g in model_data["group"].values
            else 0
            for g in groups
        ]
        r2_errs = [
            model_data[model_data["group"] == g]["r2_std"].values[0]
            if g in model_data["group"].values
            else 0
            for g in groups
        ]
        ax1.bar(x + i * width, r2_vals, width, yerr=r2_errs, label=model, capsize=3)

    ax1.axhline(y=baselines["baseline_r2"], color="red", linestyle="--", label="predict mean")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Feature Group")
    ax1.set_ylabel("R² (higher is better)")
    ax1.set_title("R² by Feature Group")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(groups, rotation=30, ha="right")
    ax1.legend()

    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
        mse_vals = [
            model_data[model_data["group"] == g]["mse_mean"].values[0]
            if g in model_data["group"].values
            else 0
            for g in groups
        ]
        mse_errs = [
            model_data[model_data["group"] == g]["mse_std"].values[0]
            if g in model_data["group"].values
            else 0
            for g in groups
        ]
        ax2.bar(x + i * width, mse_vals, width, yerr=mse_errs, label=model, capsize=3)

    ax2.axhline(y=baselines["baseline_mse"], color="red", linestyle="--", label="predict mean")
    ax2.set_xlabel("Feature Group")
    ax2.set_ylabel("MSE (lower is better)")
    ax2.set_title("MSE by Feature Group")
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(groups, rotation=30, ha="right")
    ax2.legend()

    plt.suptitle("Experiment 2: Feature Group Isolation", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "group_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved group_comparison.png")


def plot_fold_detail(results_df: pd.DataFrame) -> None:
    """Per-fold R² for each group to show variance across splits."""
    xgb_results = results_df[results_df["model"] == "xgboost"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fold_data = []
    for _, row in xgb_results.iterrows():
        for fold_idx, r2 in enumerate(row["fold_r2s"]):
            fold_data.append({"group": row["group"], "fold": fold_idx, "r2": r2})

    fold_df = pd.DataFrame(fold_data)
    sns.boxplot(data=fold_df, x="group", y="r2", ax=ax)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="R²=0 (predict mean)")
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("R² per fold")
    ax.set_title("XGBoost R² Distribution Across Folds")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fold_detail.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fold_detail.png")


def make_lstm_config() -> MLTrainingConfig:
    """Build LSTM config aligned with the tabular experiment (same dates, target, CV)."""
    xgb_config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    lstm_config = MLTrainingConfig.from_yaml(str(LSTM_CONFIG_PATH))

    lstm_config.training.data.start_date = xgb_config.training.data.start_date
    lstm_config.training.data.end_date = xgb_config.training.data.end_date
    lstm_config.training.data.cv_method = "group_timeseries"
    lstm_config.training.data.n_folds = 3
    lstm_config.training.data.shuffle = False
    # Match target type so MSE is on the same scale as tabular experiments
    lstm_config.training.features.target_type = "devigged_pinnacle"

    return lstm_config


async def load_lstm_data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load sequence training data using the LSTM adapter (3D array)."""
    lstm_config = make_lstm_config()
    features_config = lstm_config.training.features
    data_config = lstm_config.training.data

    start_dt = datetime.combine(data_config.start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(data_config.end_date, datetime.max.time(), tzinfo=UTC)

    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session=session,
            start_date=start_dt,
            end_date=end_dt,
            status=EventStatus.FINAL,
        )
        print(f"  LSTM events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    return result.X, result.y, result.feature_names, result.event_ids


def run_lstm_experiment(
    lstm_config: MLTrainingConfig,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    event_ids: np.ndarray,
) -> dict:
    """Run group timeseries CV for LSTM and return metrics dict."""
    strategy = LSTMLineMovementStrategy(
        hidden_size=lstm_config.training.model.hidden_size,
        num_layers=lstm_config.training.model.num_layers,
        dropout=lstm_config.training.model.dropout,
    )
    _, cv_result = strategy.train_with_cv(
        config=lstm_config,
        X=X,
        y=y,
        feature_names=feature_names,
        event_ids=event_ids,
    )
    return {
        "group": "sequence",
        "model": "lstm",
        "n_features": X.shape[2] if X.ndim == 3 else X.shape[1],
        "r2_mean": cv_result.mean_val_r2,
        "r2_std": cv_result.std_val_r2,
        "mse_mean": cv_result.mean_val_mse,
        "mse_std": cv_result.std_val_mse,
        "mae_mean": cv_result.mean_val_mae,
        "mae_std": cv_result.std_val_mae,
        "fold_r2s": [f.val_r2 for f in cv_result.fold_results],
        "fold_mses": [f.val_mse for f in cv_result.fold_results],
        "n_total": len(y),
    }


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, feature_names, event_ids = await load_data()
    n_events = len(set(event_ids))
    print(f"Loaded {len(X)} samples, {n_events} events, {len(feature_names)} features")

    print("\nComputing baselines...")
    baselines = compute_baselines(y, event_ids)
    print(
        f"  Predict-mean baseline: R²={baselines['baseline_r2']:.4f}, "
        f"MSE={baselines['baseline_mse']:.6f}"
    )

    print("\nRunning feature group experiments...")
    results_df = run_group_experiments(X, y, feature_names, event_ids)

    print("\nLoading LSTM sequence data...")
    X_lstm, y_lstm, feature_names_lstm, event_ids_lstm = await load_lstm_data()
    n_lstm_events = len(set(event_ids_lstm))
    print(f"  Loaded {len(X_lstm)} samples, {n_lstm_events} events, shape {X_lstm.shape}")

    print("\nRunning LSTM experiment...")
    lstm_config = make_lstm_config()
    lstm_row = run_lstm_experiment(lstm_config, X_lstm, y_lstm, feature_names_lstm, event_ids_lstm)
    print(
        f"  lstm: R²={lstm_row['r2_mean']:+.4f}±{lstm_row['r2_std']:.4f}  "
        f"MSE={lstm_row['mse_mean']:.6f}  MAE={lstm_row['mae_mean']:.4f}"
    )
    results_df = pd.concat([results_df, pd.DataFrame([lstm_row])], ignore_index=True)

    print("\nGenerating plots...")
    plot_group_comparison(results_df, baselines)
    plot_fold_detail(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print("  Saved results.csv")

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
