"""Experiment 6: Data Volume Learning Curve + Injury Timing Diagnostic.

Part 1 — Learning curve: train XGBoost with fixed tuned params on chronologically
increasing subsets (500 → ~5K events) to determine whether R² is still rising
or has plateaued. Informs whether collecting more OddsPortal data is worthwhile.

Part 2 — Injury timing diagnostic: compare tabular-only vs tabular+injuries at
sharp tier (12-24h) vs pregame tier (3-12h). OddsPortal has zero snapshots in
the 0-3h closing window where GTD injury signal is theoretically strongest, so
the "injuries add nothing" conclusion may be a timing artifact.

Uses OddsPortal data (~5K events, bet365 target) with the tuned baseline params
from xgboost_bet365_baseline_tuning_best.yaml.

Outputs saved to experiments/results/exp6_learning_curve/
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
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp6_learning_curve"
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

SUBSET_SIZES = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]


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
    feature_groups: list[str] | None = None,
    decision_tier: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load OddsPortal training data with optional overrides."""
    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    features_config = config.training.features
    data_config = config.training.data

    if feature_groups is not None:
        features_config.feature_groups = tuple(feature_groups)

    if decision_tier is not None:
        from odds_lambda.fetch_tier import FetchTier

        features_config.sampling = SamplingConfig(
            strategy="tier",
            decision_tier=FetchTier(decision_tier),
            min_hours=3.0,
            max_hours=12.0,
        )

    start_dt = datetime.combine(data_config.start_date, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(data_config.end_date, datetime.max.time(), tzinfo=UTC)

    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session=session,
            start_date=start_dt,
            end_date=end_dt,
            status=EventStatus.FINAL,
            data_source="oddsportal",
        )
        print(f"  Events in range: {len(events)}")

        result = await prepare_training_data(
            events=events,
            session=session,
            config=features_config,
        )

    return result.X, result.y, result.feature_names, result.event_ids


def subset_by_events(
    X: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    n_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take the first n_events chronologically."""
    unique_events = list(dict.fromkeys(event_ids))
    keep_events = set(unique_events[:n_events])
    mask = np.array([eid in keep_events for eid in event_ids])
    return X[mask], y[mask], event_ids[mask]


def log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """R² = a * ln(x) + b"""
    return a * np.log(x) + b


def run_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
) -> pd.DataFrame:
    """Part 1: Learning curve across subset sizes."""
    unique_events = list(dict.fromkeys(event_ids))
    n_total = len(unique_events)
    sizes = [s for s in SUBSET_SIZES if s < n_total] + [n_total]

    rows = []
    for n in sizes:
        X_sub, y_sub, eid_sub = subset_by_events(X, y, event_ids, n)
        n_actual_events = len(set(eid_sub))

        min_train = max(200, int(n_actual_events * 0.4))
        val_step = max(50, int(n_actual_events * 0.05))

        print(
            f"\n  N={n_actual_events} events, {len(X_sub)} rows, "
            f"min_train={min_train}, val_step={val_step}"
        )

        result = cross_validate_walk_forward(
            X_sub,
            y_sub,
            eid_sub,
            min_train_events=min_train,
            val_step_events=val_step,
        )

        # hours_until_event is the last feature column
        hours = X_sub[:, -1]

        rows.append(
            {
                "n_events": n_actual_events,
                "n_rows": len(X_sub),
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "mse_mean": result["mse_mean"],
                "mse_std": result["mse_std"],
                "n_folds": result["n_folds"],
                "hours_mean": float(np.mean(hours)),
                "hours_median": float(np.median(hours)),
                "hours_p25": float(np.percentile(hours, 25)),
                "hours_p75": float(np.percentile(hours, 75)),
                "fold_r2s": result["fold_metrics"],
            }
        )

        print(
            f"    R²={result['r2_mean']:+.4f} ± {result['r2_std']:.4f}  "
            f"MSE={result['mse_mean']:.6f}  folds={result['n_folds']}"
        )

    return pd.DataFrame(rows)


def run_injury_timing(
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str], np.ndarray]],
) -> pd.DataFrame:
    """Part 2: 2x2 comparison of feature group × decision tier."""
    rows = []

    for (fg_label, tier_label), (X, y, _feature_names, event_ids) in data_cache.items():
        n_events = len(set(event_ids))
        min_train = max(200, int(n_events * 0.4))
        val_step = max(50, int(n_events * 0.05))

        print(
            f"\n  {fg_label} × {tier_label}: {n_events} events, {len(X)} rows, "
            f"min_train={min_train}, val_step={val_step}"
        )

        # Need enough events for at least 1 fold
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


def plot_learning_curve(lc_df: pd.DataFrame) -> None:
    """Plot R² vs N with log-fit extrapolation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ns = lc_df["n_events"].values.astype(float)
    r2s = lc_df["r2_mean"].values
    r2_stds = lc_df["r2_std"].values

    # Main learning curve
    ax1.errorbar(ns, r2s, yerr=r2_stds, fmt="o-", capsize=4, label="Walk-forward CV R²")
    ax1.fill_between(ns, r2s - r2_stds, r2s + r2_stds, alpha=0.2)

    # Log fit
    try:
        popt, _ = curve_fit(log_func, ns, r2s, p0=[0.01, -0.05])
        x_extrap = np.linspace(ns.min(), ns.max() * 2, 200)
        ax1.plot(
            x_extrap,
            log_func(x_extrap, *popt),
            "--",
            color="red",
            label=f"log fit: {popt[0]:.4f}·ln(N) + {popt[1]:.4f}",
        )

        # Derivative at max N
        deriv_at_max = popt[0] / ns.max()
        ax1.axvline(ns.max(), color="gray", linestyle=":", alpha=0.5)
        ax1.set_title(f"Learning Curve (dR²/dN at max = {deriv_at_max:.2e})")
    except RuntimeError:
        ax1.set_title("Learning Curve (log fit failed)")
        popt = None

    ax1.set_xlabel("Number of Events")
    ax1.set_ylabel("R² (walk-forward CV)")
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Per-fold scatter
    fold_data = []
    for _, row in lc_df.iterrows():
        for fold_info in row["fold_r2s"]:
            fold_data.append(
                {
                    "n_events": row["n_events"],
                    "r2": fold_info["r2"],
                    "n_val_events": fold_info["n_val_events"],
                }
            )
    fold_df = pd.DataFrame(fold_data)

    ax2.scatter(fold_df["n_events"], fold_df["r2"], alpha=0.5, s=40)
    ax2.plot(ns, r2s, "r-", linewidth=2, label="Mean R²")
    ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Number of Events (total in subset)")
    ax2.set_ylabel("R² per fold")
    ax2.set_title("Per-Fold R² Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 6: Data Volume Learning Curve (bet365, tabular-only)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved learning_curve.png")

    return popt


def plot_injury_timing(it_df: pd.DataFrame) -> None:
    """Bar chart of 2×2 injury timing comparison."""
    plot_df = it_df[it_df["n_folds"] > 0].copy()
    if plot_df.empty:
        print("  No valid injury timing results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{row['feature_group']}\n{row['decision_tier']}" for _, row in plot_df.iterrows()]
    x = np.arange(len(labels))
    colors = ["#2196F3" if "tabular" in label else "#FF9800" for label in labels]

    ax.bar(x, plot_df["r2_mean"], yerr=plot_df["r2_std"], capsize=5, color=colors, alpha=0.8)

    # Annotate with N
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, -0.01, f"n={row['n_events']}", ha="center", va="top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² (walk-forward CV)")
    ax.set_title("Injury Timing Diagnostic: Feature Group × Decision Tier")
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "injury_timing.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved injury_timing.png")


def write_findings(
    lc_df: pd.DataFrame,
    it_df: pd.DataFrame,
    log_params: tuple[float, float] | None,
) -> None:
    """Write FINDINGS.md with results."""
    lines = [
        "# Experiment 6: Data Volume Learning Curve + Injury Timing Diagnostic",
        "",
        "## Part 1: Learning Curve",
        "",
        "Train XGBoost (tuned bet365 baseline params, tabular-only) on chronologically",
        "increasing subsets of OddsPortal data. Walk-forward CV with expanding window.",
        "",
        "### Results",
        "",
        "| N Events | N Rows | R² Mean | R² Std | MSE Mean | Folds | Hours Mean |",
        "|----------|--------|---------|--------|----------|-------|------------|",
    ]

    for _, row in lc_df.iterrows():
        lines.append(
            f"| {row['n_events']:,} | {row['n_rows']:,} | "
            f"{row['r2_mean']:+.4f} | {row['r2_std']:.4f} | "
            f"{row['mse_mean']:.6f} | {row['n_folds']} | "
            f"{row['hours_mean']:.1f}h |"
        )

    lines.append("")

    if log_params is not None:
        a, b = log_params
        n_max = lc_df["n_events"].max()
        deriv = a / n_max
        r2_at_10k = log_func(np.array([10000.0]), a, b)[0]
        r2_at_20k = log_func(np.array([20000.0]), a, b)[0]
        lines.extend(
            [
                "### Log-Fit Extrapolation",
                "",
                f"- **Model**: R² = {a:.4f} · ln(N) + ({b:.4f})",
                f"- **Marginal return at N={n_max:,}**: dR²/dN = {deriv:.2e}",
                f"- **Extrapolated R² at 10K events**: {r2_at_10k:.4f}",
                f"- **Extrapolated R² at 20K events**: {r2_at_20k:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "### Hours-to-Event Distribution",
            "",
            "All subsets sample from the sharp tier (~12-24h before game). OddsPortal",
            "snapshots average ~19h before game, with zero coverage in the 0-3h closing window.",
            "",
            "| N Events | Hours Mean | Hours Median | Hours P25 | Hours P75 |",
            "|----------|-----------|-------------|-----------|-----------|",
        ]
    )

    for _, row in lc_df.iterrows():
        lines.append(
            f"| {row['n_events']:,} | {row['hours_mean']:.1f} | "
            f"{row['hours_median']:.1f} | {row['hours_p25']:.1f} | "
            f"{row['hours_p75']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Part 2: Injury Timing Diagnostic",
            "",
            "Compare tabular-only vs tabular+injuries at sharp (12-24h) vs pregame (3-12h) tier.",
            'Tests whether the "injuries add nothing" conclusion is a timing artifact of',
            "OddsPortal's ~19h average decision time.",
            "",
            "### Results",
            "",
            "| Feature Group | Tier | N Events | N Rows | R² Mean | R² Std | MSE Mean | Note |",
            "|---------------|------|----------|--------|---------|--------|----------|------|",
        ]
    )

    for _, row in it_df.iterrows():
        r2_str = f"{row['r2_mean']:+.4f}" if not np.isnan(row["r2_mean"]) else "N/A"
        r2_std_str = f"{row['r2_std']:.4f}" if not np.isnan(row["r2_std"]) else "N/A"
        mse_str = f"{row['mse_mean']:.6f}" if not np.isnan(row["mse_mean"]) else "N/A"
        note = row.get("note", "")
        lines.append(
            f"| {row['feature_group']} | {row['decision_tier']} | "
            f"{row['n_events']:,} | {row['n_rows']:,} | "
            f"{r2_str} | {r2_std_str} | {mse_str} | {note} |"
        )

    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "**Important caveat**: OddsPortal has zero snapshots in the 0-3h closing window",
            "where GTD injury signal is theoretically strongest (r=0.28, p=0.0003 in Exp 4).",
            "The pregame tier (3-12h) partially probes closer-to-game timing, but the true",
            "test requires Odds API data with dense snapshot coverage at < 3h before game.",
            "",
        ]
    )

    (OUTPUT_DIR / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print("  Saved FINDINGS.md")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Learning Curve ──
    print("=" * 60)
    print("Part 1: Learning Curve")
    print("=" * 60)

    print("\nLoading OddsPortal data (tabular-only, sharp tier)...")
    X, y, feature_names, event_ids = await load_data(feature_groups=["tabular"])
    n_events = len(set(event_ids))
    print(f"  Loaded {len(X)} rows, {n_events} events, {len(feature_names)} features")
    print(f"  Features: {feature_names}")

    print("\nRunning learning curve...")
    lc_df = run_learning_curve(X, y, event_ids)

    # Save (drop fold_r2s list column for CSV)
    lc_csv = lc_df.drop(columns=["fold_r2s"])
    lc_csv.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    print("\n  Saved learning_curve.csv")

    # ── Part 2: Injury Timing Diagnostic ──
    print("\n" + "=" * 60)
    print("Part 2: Injury Timing Diagnostic")
    print("=" * 60)

    # Load 4 combinations: {tabular, tabular+injuries} × {sharp, pregame}
    data_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {}

    # tabular × sharp — already loaded
    data_cache[("tabular", "sharp")] = (X, y, feature_names, event_ids)

    print("\nLoading tabular × pregame...")
    data_cache[("tabular", "pregame")] = await load_data(
        feature_groups=["tabular"], decision_tier="pregame"
    )

    print("\nLoading tabular+injuries × sharp...")
    data_cache[("tabular+injuries", "sharp")] = await load_data(
        feature_groups=["tabular", "injuries"], decision_tier="sharp"
    )

    print("\nLoading tabular+injuries × pregame...")
    data_cache[("tabular+injuries", "pregame")] = await load_data(
        feature_groups=["tabular", "injuries"], decision_tier="pregame"
    )

    print("\nRunning injury timing comparison...")
    it_df = run_injury_timing(dict(data_cache))

    it_df.to_csv(OUTPUT_DIR / "injury_timing.csv", index=False)
    print("\n  Saved injury_timing.csv")

    # ── Plots & Findings ──
    print("\n" + "=" * 60)
    print("Generating outputs")
    print("=" * 60)

    log_params = plot_learning_curve(lc_df)
    plot_injury_timing(it_df)
    write_findings(lc_df, it_df, log_params)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
