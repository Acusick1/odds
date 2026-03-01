"""Experiment 6b: GTD Injury Signal at Closing Tier (bet365, Odds API).

Exp 6 Part 2 tested injury timing on OddsPortal data, but 93% of pregame-tier
events fell back to ~19h sharp-tier snapshots (OddsPortal has no closing-tier
coverage). The 0-3h window where GTD signal is strongest (r=0.28, Exp 4) was
untestable.

This experiment uses Odds API data (~890 events with bet365 closing-tier
snapshots, avg 0.2h before game) to run the same 2x2 comparison:
  {tabular, tabular+injuries} x {closing, sharp}

If the GTD hypothesis holds: injuries should help at closing tier (late GTD
designations visible) but not at sharp tier (replicating Exp 6's null result).

Per-feature correlation analysis validates whether the Exp 4 r=0.28
inj_impact_gtd_away signal transfers from Pinnacle to bet365 target.

Outputs saved to experiments/results/exp6b_injury_closing_tier/
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.training.config import MLTrainingConfig, SamplingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp6b_injury_closing_tier"
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "xgboost_bet365_baseline_tuning_best.yaml"
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120

# Tuned params from xgboost_bet365_baseline_tuning_best.yaml
TUNED_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.2801239726127749,
    "min_child_weight": 36,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "gamma": 0.0042691916317070255,
    "reg_alpha": 0.6494989853089118,
    "reg_lambda": 1.4670424407885743,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def make_xgboost() -> XGBRegressor:
    return XGBRegressor(**TUNED_PARAMS)


def cross_validate_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    min_train_events: int,
    val_step_events: int,
) -> dict[str, Any]:
    """Walk-forward CV with per-fold detail."""
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in make_walk_forward_splits(
        event_ids,
        min_train_events=min_train_events,
        val_step_events=val_step_events,
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_xgboost()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        fold_metrics.append(
            {
                "r2": r2_score(y_val, y_pred),
                "mse": mean_squared_error(y_val, y_pred),
                "mae": mean_absolute_error(y_val, y_pred),
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_train_events": len(set(event_ids[train_idx])),
                "n_val_events": len(set(event_ids[val_idx])),
            }
        )

    r2_vals = [f["r2"] for f in fold_metrics]
    mse_vals = [f["mse"] for f in fold_metrics]

    return {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "mse_mean": float(np.mean(mse_vals)),
        "mse_std": float(np.std(mse_vals)),
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
    }


async def load_data(
    feature_groups: list[str],
    decision_tier: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load Odds API training data with bet365 bookmaker config."""
    from odds_lambda.fetch_tier import FetchTier

    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    features_config = config.training.features
    data_config = config.training.data

    features_config.feature_groups = tuple(feature_groups)
    features_config.sampling = SamplingConfig(
        strategy="tier",
        decision_tier=FetchTier(decision_tier),
        min_hours=3.0,
        max_hours=12.0,
    )
    features_config.sharp_bookmakers = ["bet365"]
    features_config.retail_bookmakers = ["betway", "betfred", "bwin"]
    features_config.target_type = "devigged_bookmaker"
    features_config.target_bookmaker = "bet365"

    start_dt = datetime.combine(data_config.start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(data_config.end_date, datetime.max.time(), tzinfo=UTC)

    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session=session,
            start_date=start_dt,
            end_date=end_dt,
            status=EventStatus.FINAL,
            data_source="oddsapi",
        )
        print(f"  Events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    return result.X, result.y, result.feature_names, result.event_ids


def compute_correlations(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    min_valid: int = 20,
) -> pd.DataFrame:
    """Compute Pearson r between each feature and the target."""
    rows = []
    for i, feat in enumerate(feature_names):
        x = X[:, i]
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < min_valid:
            continue
        r, p = stats.pearsonr(x[valid], y[valid])
        rows.append(
            {
                "feature": feat,
                "correlation": r,
                "p_value": p,
                "n_valid": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)


def run_2x2_comparison(
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str], np.ndarray]],
) -> pd.DataFrame:
    """Run walk-forward CV for each cell of the 2x2 design."""
    rows = []

    for (fg_label, tier_label), (X, y, _feature_names, event_ids) in data_cache.items():
        n_events = len(set(event_ids))
        min_train = max(200, int(n_events * 0.4))
        val_step = max(50, int(n_events * 0.05))

        print(
            f"\n  {fg_label} x {tier_label}: {n_events} events, {len(X)} rows, "
            f"min_train={min_train}, val_step={val_step}"
        )

        if n_events < min_train + val_step:
            print(
                f"    SKIP: not enough events for CV (need {min_train + val_step}, have {n_events})"
            )
            rows.append(
                {
                    "feature_group": fg_label,
                    "decision_tier": tier_label,
                    "n_events": n_events,
                    "n_rows": len(X),
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "mse_mean": np.nan,
                    "mse_std": np.nan,
                    "n_folds": 0,
                    "note": "insufficient data for CV",
                }
            )
            continue

        result = cross_validate_walk_forward(
            X,
            y,
            event_ids,
            min_train_events=min_train,
            val_step_events=val_step,
        )

        rows.append(
            {
                "feature_group": fg_label,
                "decision_tier": tier_label,
                "n_events": n_events,
                "n_rows": len(X),
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "mse_mean": result["mse_mean"],
                "mse_std": result["mse_std"],
                "n_folds": result["n_folds"],
                "note": "",
            }
        )

        print(
            f"    R²={result['r2_mean']:+.4f} ± {result['r2_std']:.4f}  "
            f"MSE={result['mse_mean']:.6f}  folds={result['n_folds']}"
        )

    return pd.DataFrame(rows)


def plot_2x2(df: pd.DataFrame) -> None:
    """Bar chart of 2x2 injury timing comparison."""
    plot_df = df[df["n_folds"] > 0].copy()
    if plot_df.empty:
        print("  No valid results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{row['feature_group']}\n{row['decision_tier']}" for _, row in plot_df.iterrows()]
    x = np.arange(len(labels))
    colors = ["#2196F3" if "tabular\n" in label else "#FF9800" for label in labels]

    ax.bar(x, plot_df["r2_mean"], yerr=plot_df["r2_std"], capsize=5, color=colors, alpha=0.8)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, -0.01, f"n={row['n_events']}", ha="center", va="top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² (walk-forward CV)")
    ax.set_title("Exp 6b: Injury Signal x Decision Tier (bet365, Odds API)")
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "injury_closing_tier.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved injury_closing_tier.png")


def plot_correlation_comparison(
    corr_closing: pd.DataFrame,
    corr_sharp: pd.DataFrame,
) -> None:
    """Side-by-side correlation bars for closing vs sharp tier."""
    # Merge on feature name
    merged = pd.merge(
        corr_closing[["feature", "correlation"]],
        corr_sharp[["feature", "correlation"]],
        on="feature",
        suffixes=("_closing", "_sharp"),
        how="outer",
    ).fillna(0)

    # Sort by absolute closing correlation
    merged["abs_closing"] = merged["correlation_closing"].abs()
    merged = merged.sort_values("abs_closing", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(merged) * 0.35)))

    y_pos = np.arange(len(merged))
    bar_height = 0.35

    ax.barh(
        y_pos - bar_height / 2,
        merged["correlation_closing"],
        bar_height,
        label="Closing (0-3h)",
        color="#FF9800",
        alpha=0.8,
    )
    ax.barh(
        y_pos + bar_height / 2,
        merged["correlation_sharp"],
        bar_height,
        label="Sharp (12-24h)",
        color="#2196F3",
        alpha=0.8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["feature"], fontsize=8)
    ax.set_xlabel("Pearson r with bet365 CLV target")
    ax.set_title("Per-Feature Correlation: Closing vs Sharp Tier")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved correlation_comparison.png")


def write_findings(
    results_df: pd.DataFrame,
    corr_closing: pd.DataFrame,
    corr_sharp: pd.DataFrame,
) -> None:
    """Write FINDINGS.md with results and interpretation."""
    lines = [
        "# Experiment 6b: GTD Injury Signal at Closing Tier (bet365, Odds API)",
        "",
        "## Design",
        "",
        "2x2 comparison: {tabular, tabular+injuries} x {closing (0-3h), sharp (12-24h)}",
        "on Odds API data with bet365 as target bookmaker. Fixed hyperparams from",
        "xgboost_bet365_baseline_tuning_best.yaml. Walk-forward CV with expanding window.",
        "",
        "## Results",
        "",
        "| Feature Group | Tier | N Events | N Rows | R² Mean | R² Std | MSE Mean | Folds |",
        "|---------------|------|----------|--------|---------|--------|----------|-------|",
    ]

    for _, row in results_df.iterrows():
        r2_str = f"{row['r2_mean']:+.4f}" if not np.isnan(row["r2_mean"]) else "N/A"
        r2_std_str = f"{row['r2_std']:.4f}" if not np.isnan(row["r2_std"]) else "N/A"
        mse_str = f"{row['mse_mean']:.6f}" if not np.isnan(row["mse_mean"]) else "N/A"
        note = row.get("note", "")
        folds = row["n_folds"]
        lines.append(
            f"| {row['feature_group']} | {row['decision_tier']} | "
            f"{row['n_events']:,} | {row['n_rows']:,} | "
            f"{r2_str} | {r2_std_str} | {mse_str} | {folds} |"
        )
        if note:
            lines[-1] += f" {note}"

    # Correlation tables
    for label, corr_df in [("Closing (0-3h)", corr_closing), ("Sharp (12-24h)", corr_sharp)]:
        lines.extend(
            [
                "",
                f"## Per-Feature Correlations: {label}",
                "",
                "| Feature | Pearson r | p-value | N |",
                "|---------|-----------|---------|---|",
            ]
        )
        if corr_df.empty:
            lines.append("| (no features with sufficient data) | | | |")
        else:
            for _, row in corr_df.iterrows():
                sig = (
                    "***"
                    if row["p_value"] < 0.001
                    else "**"
                    if row["p_value"] < 0.01
                    else "*"
                    if row["p_value"] < 0.05
                    else ""
                )
                lines.append(
                    f"| {row['feature']} | {row['correlation']:+.4f} | "
                    f"{row['p_value']:.4f}{sig} | {row['n_valid']:.0f} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "_(to be filled after reviewing results)_",
            "",
        ]
    )

    (OUTPUT_DIR / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print("  Saved FINDINGS.md")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [
        ("tabular", "closing"),
        ("tabular+injuries", "closing"),
        ("tabular", "sharp"),
        ("tabular+injuries", "sharp"),
    ]
    feature_group_map = {
        "tabular": ["tabular"],
        "tabular+injuries": ["tabular", "injuries"],
    }

    # Load data for each cell
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {}

    for fg_label, tier in cells:
        print(f"\nLoading {fg_label} x {tier}...")
        data_cache[(fg_label, tier)] = await load_data(
            feature_groups=feature_group_map[fg_label],
            decision_tier=tier,
        )
        X, y, fnames, eids = data_cache[(fg_label, tier)]
        print(f"  Loaded {len(X)} rows, {len(set(eids))} events, {len(fnames)} features")

    # Run 2x2 comparison
    print("\n" + "=" * 60)
    print("Running 2x2 walk-forward CV")
    print("=" * 60)

    results_df = run_2x2_comparison(data_cache)
    results_df.to_csv(OUTPUT_DIR / "injury_closing_tier.csv", index=False)
    print("\n  Saved injury_closing_tier.csv")

    # Per-feature correlations (use tabular+injuries data for full feature set)
    print("\n" + "=" * 60)
    print("Computing per-feature correlations")
    print("=" * 60)

    X_close, y_close, fnames_close, _ = data_cache[("tabular+injuries", "closing")]
    corr_closing = compute_correlations(X_close, y_close, fnames_close)
    corr_closing.to_csv(OUTPUT_DIR / "correlations_closing.csv", index=False)
    print(f"\n  Closing tier — {len(corr_closing)} features:")
    if not corr_closing.empty:
        for _, row in corr_closing.head(10).iterrows():
            print(
                f"    {row['feature']:30s}  r={row['correlation']:+.4f}  "
                f"p={row['p_value']:.4f}  n={row['n_valid']:.0f}"
            )

    X_sharp, y_sharp, fnames_sharp, _ = data_cache[("tabular+injuries", "sharp")]
    corr_sharp = compute_correlations(X_sharp, y_sharp, fnames_sharp)
    corr_sharp.to_csv(OUTPUT_DIR / "correlations_sharp.csv", index=False)
    print(f"\n  Sharp tier — {len(corr_sharp)} features:")
    if not corr_sharp.empty:
        for _, row in corr_sharp.head(10).iterrows():
            print(
                f"    {row['feature']:30s}  r={row['correlation']:+.4f}  "
                f"p={row['p_value']:.4f}  n={row['n_valid']:.0f}"
            )

    # Plots and findings
    print("\n" + "=" * 60)
    print("Generating outputs")
    print("=" * 60)

    plot_2x2(results_df)
    if not corr_closing.empty and not corr_sharp.empty:
        plot_correlation_comparison(corr_closing, corr_sharp)
    write_findings(results_df, corr_closing, corr_sharp)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
