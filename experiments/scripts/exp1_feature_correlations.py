"""Experiment 1: Feature-Target Correlation Analysis.

Computes raw Pearson/Spearman correlations between each feature and the
devigged Pinnacle CLV delta target. If nothing correlates individually,
no model will extract signal from combinations.

Outputs saved to experiments/results/exp1_feature_correlations/
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
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp1_feature_correlations"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "xgboost_cross_source_v1.yaml"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120


def feature_group(name: str) -> str:
    if name.startswith("tab_"):
        return "tabular"
    elif name.startswith("traj_"):
        return "trajectory"
    elif name.startswith("xsrc_"):
        return "cross_source"
    elif name.startswith("pm_"):
        return "polymarket"
    elif name == "hours_until_event":
        return "timing"
    else:
        return "other"


async def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
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

    df = pd.DataFrame(result.X, columns=result.feature_names)
    df["target"] = result.y
    df["event_id"] = result.event_ids

    return df, result.y, result.feature_names, result.event_ids


def compute_correlations(df: pd.DataFrame, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_names:
        x = df[feat].values
        if np.std(x) < 1e-10:
            rows.append(
                {
                    "feature": feat,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                }
            )
            continue
        r, p_r = stats.pearsonr(x, y)
        rho, p_s = stats.spearmanr(x, y)
        rows.append(
            {
                "feature": feat,
                "pearson_r": r,
                "pearson_p": p_r,
                "spearman_rho": rho,
                "spearman_p": p_s,
            }
        )

    corr_df = pd.DataFrame(rows).set_index("feature")
    corr_df["abs_pearson"] = corr_df["pearson_r"].abs()
    corr_df["abs_spearman"] = corr_df["spearman_rho"].abs()
    corr_df["avg_abs"] = (corr_df["abs_pearson"] + corr_df["abs_spearman"]) / 2
    corr_df["group"] = [feature_group(f) for f in corr_df.index]

    # Multiple testing correction
    valid = corr_df["pearson_p"].notna()
    p_vals = corr_df.loc[valid, "pearson_p"].values
    reject_bh, pvals_bh, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    reject_bonf, pvals_bonf, _, _ = multipletests(p_vals, alpha=0.05, method="bonferroni")
    corr_df.loc[valid, "p_bh"] = pvals_bh
    corr_df.loc[valid, "p_bonferroni"] = pvals_bonf
    corr_df.loc[valid, "sig_bh"] = reject_bh
    corr_df.loc[valid, "sig_bonferroni"] = reject_bonf

    return corr_df.sort_values("avg_abs", ascending=False)


def plot_correlations(corr_df: pd.DataFrame) -> None:
    plot_df = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(plot_df) * 0.3)))

    colors_p = ["#d32f2f" if p < 0.05 else "#90a4ae" for p in plot_df["pearson_p"]]
    ax1.barh(range(len(plot_df)), plot_df["pearson_r"], color=colors_p)
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df.index, fontsize=8)
    ax1.set_xlabel("Pearson r")
    ax1.set_title("Pearson Correlation with Target")
    ax1.axvline(x=0, color="black", linewidth=0.5)

    plot_df_s = corr_df.dropna(subset=["spearman_rho"]).sort_values("spearman_rho")
    colors_s = ["#d32f2f" if p < 0.05 else "#90a4ae" for p in plot_df_s["spearman_p"]]
    ax2.barh(range(len(plot_df_s)), plot_df_s["spearman_rho"], color=colors_s)
    ax2.set_yticks(range(len(plot_df_s)))
    ax2.set_yticklabels(plot_df_s.index, fontsize=8)
    ax2.set_xlabel("Spearman ρ")
    ax2.set_title("Spearman Correlation with Target")
    ax2.axvline(x=0, color="black", linewidth=0.5)

    fig.suptitle("Feature-Target Correlations (red = p < 0.05)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlations_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved correlations_bar.png")


def plot_scatter_top(df: pd.DataFrame, y: np.ndarray, corr_df: pd.DataFrame) -> None:
    top_features = corr_df.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat in zip(axes.flat, top_features, strict=False):
        x = df[feat].values
        ax.scatter(x, y, alpha=0.3, s=15)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2)
        r_val = corr_df.loc[feat, "pearson_r"]
        p_val = corr_df.loc[feat, "pearson_p"]
        ax.set_title(f"{feat}\nr={r_val:.3f}, p={p_val:.3f}", fontsize=9)
        ax.set_ylabel("Target (CLV delta)")

    plt.suptitle("Top 6 Correlated Features vs Target", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_top6.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_top6.png")


def plot_target_distribution(y: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(y, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--")
    ax1.set_xlabel("Devigged Pinnacle CLV Delta")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Target Distribution (n={len(y)})")

    stats.probplot(y, dist="norm", plot=ax2)
    ax2.set_title("QQ Plot (target vs normal)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved target_distribution.png")


def plot_intercorrelation(
    df: pd.DataFrame, feature_names: list[str], corr_df: pd.DataFrame
) -> None:
    top_n = min(20, len(corr_df))
    top_feats = corr_df.head(top_n).index.tolist()
    sub_corr = df[top_feats + ["target"]].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub_corr,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="RdBu_r",
        square=True,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Inter-correlation: Top {top_n} Features + Target")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "intercorrelation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved intercorrelation_heatmap.png")


def print_summary(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    event_ids: np.ndarray,
    corr_df: pd.DataFrame,
    collinear_pairs: list[tuple[str, str, float]],
) -> str:
    n_events = len(set(event_ids))
    n_testable = corr_df["pearson_p"].notna().sum()
    n_constant = len(corr_df) - n_testable
    n_sig_raw = (corr_df["pearson_p"] < 0.05).sum()
    n_sig_bh = int(corr_df["sig_bh"].sum()) if "sig_bh" in corr_df else 0
    n_sig_bonf = int(corr_df["sig_bonferroni"].sum()) if "sig_bonferroni" in corr_df else 0

    lines = []
    lines.append("=" * 65)
    lines.append("EXPERIMENT 1: FEATURE-TARGET CORRELATION ANALYSIS")
    lines.append("=" * 65)
    lines.append(f"\nDataset: {len(df)} samples, {n_events} events, {len(feature_names)} features")
    lines.append(f"  {n_testable} testable, {n_constant} constant (excluded from correction)")
    lines.append(f"Avg rows/event: {len(df) / n_events:.1f}")
    lines.append("Target: devigged Pinnacle CLV delta")
    lines.append(f"  mean={y.mean():.5f}, std={y.std():.5f}, range=[{y.min():.4f}, {y.max():.4f}]")
    lines.append(f"  skewness={stats.skew(y):.3f}, kurtosis={stats.kurtosis(y):.3f}")

    lines.append("\nCorrelation summary:")
    lines.append(
        f"  Max |Pearson r|:  {corr_df['abs_pearson'].max():.4f} ({corr_df['abs_pearson'].idxmax()})"
    )
    lines.append(
        f"  Max |Spearman ρ|: {corr_df['abs_spearman'].max():.4f} ({corr_df['abs_spearman'].idxmax()})"
    )
    lines.append(f"  Median |Pearson|: {corr_df['abs_pearson'].median():.4f}")

    lines.append(f"\nSignificance (p < 0.05, {n_testable} testable features):")
    lines.append(f"  Uncorrected: {n_sig_raw}/{n_testable}")
    lines.append(f"  BH (FDR):    {n_sig_bh}/{n_testable}")
    lines.append(f"  Bonferroni:  {n_sig_bonf}/{n_testable}")

    lines.append("\nTop 15 features (by avg |correlation|):")
    for i, (feat, row) in enumerate(corr_df.head(15).iterrows(), 1):
        sig = "*" if row.get("sig_bh", False) else " "
        lines.append(
            f"  {i:2d}. {feat:40s} r={row['pearson_r']:+.4f}  "
            f"ρ={row['spearman_rho']:+.4f}  {sig} [{row['group']}]"
        )
    lines.append("\n  * = BH-significant (FDR < 0.05)")

    # Group summary
    lines.append("\nBy feature group:")
    group_stats = (
        corr_df.groupby("group")
        .agg(
            n=("avg_abs", "count"),
            mean_abs_r=("abs_pearson", "mean"),
            max_abs_r=("abs_pearson", "max"),
            n_sig=("pearson_p", lambda x: (x < 0.05).sum()),
        )
        .sort_values("mean_abs_r", ascending=False)
    )
    for grp, row in group_stats.iterrows():
        lines.append(
            f"  {grp:15s}  n={int(row['n']):2d}  mean|r|={row['mean_abs_r']:.4f}  max|r|={row['max_abs_r']:.4f}  sig={int(row['n_sig'])}"
        )

    # Sparse features
    zero_frac = (df[feature_names] == 0).mean()
    sparse = zero_frac[zero_frac > 0.5].sort_values(ascending=False)
    if len(sparse) > 0:
        lines.append(f"\nSparse features (>50% zero): {len(sparse)}")
        for feat, frac in sparse.items():
            lines.append(f"  {feat:40s} {frac:5.1%} zero  [{feature_group(feat)}]")

    lines.append(f"\nHighly correlated feature pairs (|r| > 0.8): {len(collinear_pairs)}")
    for f1, f2, r in collinear_pairs[:10]:
        lines.append(f"  {r:+.3f}  {f1}  <->  {f2}")
    if len(collinear_pairs) > 10:
        lines.append(f"  ... and {len(collinear_pairs) - 10} more")

    # Hours-to-game analysis
    if "hours_until_event" in feature_names:
        hours = df["hours_until_event"].values
        bins = np.linspace(hours.min(), hours.max(), 4)
        labels = ["close", "mid", "far"]
        df["time_bin"] = pd.cut(hours, bins=bins, labels=labels)

        lines.append("\nTarget std by time bin:")
        for lbl in ["far", "mid", "close"]:
            mask = df["time_bin"] == lbl
            t = df.loc[mask, "target"]
            lines.append(f"  {lbl:6s}: std={t.std():.5f}, mean={t.mean():.5f}, n={len(t)}")

        lines.append("\nPearson r by time bin (top 8 features):")
        top8 = corr_df.head(8).index.tolist()
        header = f"  {'feature':40s} {'far':>8s} {'mid':>8s} {'close':>8s}"
        lines.append(header)
        for feat in top8:
            vals = []
            for lbl in ["far", "mid", "close"]:
                mask = df["time_bin"] == lbl
                x = df.loc[mask, feat].values
                yt = df.loc[mask, "target"].values
                if len(x) > 5 and np.std(x) > 1e-10:
                    r, _ = stats.pearsonr(x, yt)
                    vals.append(f"{r:+.4f}")
                else:
                    vals.append("   N/A")
            lines.append(f"  {feat:40s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    # Outlier robustness: winsorized correlations
    lines.append("\nOutlier robustness (target winsorized at 1st/99th percentile):")
    lo, hi = np.percentile(y, [1, 99])
    y_wins = np.clip(y, lo, hi)
    lines.append(
        f"  Target clipped to [{lo:.4f}, {hi:.4f}] ({(y < lo).sum() + (y > hi).sum()} values clipped)"
    )
    lines.append(f"  {'feature':40s} {'raw r':>8s} {'winsorized r':>13s} {'delta':>8s}")
    top10 = corr_df.head(10).index.tolist()
    for feat in top10:
        x = df[feat].values
        if np.std(x) < 1e-10:
            continue
        r_raw = corr_df.loc[feat, "pearson_r"]
        r_wins, _ = stats.pearsonr(x, y_wins)
        delta = r_wins - r_raw
        lines.append(f"  {feat:40s} {r_raw:+8.4f} {r_wins:+13.4f} {delta:+8.4f}")

    lines.append("\n" + "=" * 65)

    text = "\n".join(lines)
    print(text)
    return text


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, y, feature_names, event_ids = await load_data()
    n_events = len(set(event_ids))
    print(f"Loaded {len(df)} samples, {n_events} events, {len(feature_names)} features\n")

    print("Computing correlations...")
    corr_df = compute_correlations(df, y, feature_names)

    # Collinear pairs
    feat_corr = df[feature_names].corr()
    collinear_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = feat_corr.iloc[i, j]
            if abs(r) > 0.8:
                collinear_pairs.append((feature_names[i], feature_names[j], r))
    collinear_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("Generating plots...")
    plot_correlations(corr_df)
    plot_scatter_top(df, y, corr_df)
    plot_target_distribution(y)
    plot_intercorrelation(df, feature_names, corr_df)

    print()
    print_summary(df, y, feature_names, event_ids, corr_df, collinear_pairs)

    # Save artifacts
    corr_df.to_csv(OUTPUT_DIR / "correlations.csv")
    print("\nSaved correlations.csv")
    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
