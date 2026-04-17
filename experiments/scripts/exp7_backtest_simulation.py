"""Experiment 7: Walk-Forward Betting Simulation.

Does the XGBoost CLV prediction signal (~3.6% R² on bet365) translate to
+EV betting after vig? Walk-forward CV trains on expanding window, predicts
on val folds. For each val-fold event, simulates a bet using model prediction,
raw bet365 vigged American odds, and actual game outcome.

Prediction → bet mapping:
  delta > +threshold  → bet home (home underpriced, line will move toward home)
  delta < -threshold  → bet away (home overpriced, line will move toward away)
  |delta| <= threshold → no bet

Flat $100 per bet. Sweeps threshold ∈ [0.005, 0.01, 0.015, 0.02, 0.03, 0.05].
Permutation test (1,000 shuffles) for significance.

Uses OddsPortal data (~5K events, bet365 target) with tuned baseline params
from xgboost_bet365_baseline_tuning_best.yaml.

Outputs saved to experiments/results/exp7_backtest_sim/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.odds_math import american_to_decimal
from odds_core.snapshot_utils import extract_odds_from_snapshot
from sqlalchemy import select
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7_backtest_sim"
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "xgboost_bet365_baseline_tuning_best.yaml"
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120

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

THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
N_PERMUTATIONS = 1000
STAKE = 100.0


@dataclass
class BettingContext:
    home_american_odds: int
    away_american_odds: int
    home_wins: bool


def make_xgboost() -> XGBRegressor:
    return XGBRegressor(**TUNED_PARAMS)


async def load_data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load OddsPortal training data (tabular-only, sharp tier)."""
    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    features_config = config.training.features
    data_config = config.training.data

    features_config.feature_groups = ("tabular",)

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


async def load_betting_context(event_ids: np.ndarray) -> dict[str, BettingContext]:
    """Load bet365 vigged odds and game outcomes for all training events.

    For each event, picks the sharp-tier snapshot (hours_before >= 12, most recent),
    extracts bet365 h2h American odds, and determines game outcome from scores.
    """
    unique_ids = list(dict.fromkeys(event_ids))
    print(f"  Loading betting context for {len(unique_ids)} events...")

    ctx: dict[str, BettingContext] = {}

    async with async_session_maker() as session:
        # Batch-load events for scores and team names
        event_result = await session.execute(select(Event).where(Event.id.in_(unique_ids)))
        events_by_id: dict[str, Event] = {e.id: e for e in event_result.scalars().all()}

        # Batch-load sharp-tier snapshots
        snapshot_result = await session.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.event_id.in_(unique_ids))
            .where(OddsSnapshot.fetch_tier == "sharp")
            .order_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time.desc())
        )
        snapshots = list(snapshot_result.scalars().all())

        # Group by event, take most recent sharp snapshot per event
        snapshots_by_event: dict[str, OddsSnapshot] = {}
        for snap in snapshots:
            if snap.event_id not in snapshots_by_event:
                snapshots_by_event[snap.event_id] = snap

    skipped_no_snapshot = 0
    skipped_no_event = 0
    skipped_no_odds = 0
    skipped_no_score = 0

    for eid in unique_ids:
        event = events_by_id.get(eid)
        if event is None:
            skipped_no_event += 1
            continue

        if event.home_score is None or event.away_score is None:
            skipped_no_score += 1
            continue

        snapshot = snapshots_by_event.get(eid)
        if snapshot is None:
            skipped_no_snapshot += 1
            continue

        # Extract bet365 h2h odds from raw_data
        odds_list = extract_odds_from_snapshot(snapshot, eid, market="h2h")
        bet365_odds = [o for o in odds_list if o.bookmaker_key == "bet365"]

        home_odds_obj = None
        away_odds_obj = None
        for o in bet365_odds:
            if o.outcome_name == event.home_team:
                home_odds_obj = o
            elif o.outcome_name == event.away_team:
                away_odds_obj = o

        if home_odds_obj is None or away_odds_obj is None:
            skipped_no_odds += 1
            continue

        ctx[eid] = BettingContext(
            home_american_odds=home_odds_obj.price,
            away_american_odds=away_odds_obj.price,
            home_wins=event.home_score > event.away_score,
        )

    print(f"  Loaded: {len(ctx)} events with betting context")
    if skipped_no_event:
        print(f"  Skipped (no event): {skipped_no_event}")
    if skipped_no_score:
        print(f"  Skipped (no score): {skipped_no_score}")
    if skipped_no_snapshot:
        print(f"  Skipped (no sharp snapshot): {skipped_no_snapshot}")
    if skipped_no_odds:
        print(f"  Skipped (no bet365 h2h): {skipped_no_odds}")

    return ctx


def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    min_train_events: int,
    val_step_events: int,
) -> list[dict[str, Any]]:
    """Walk-forward CV: train on expanding window, collect all val predictions."""
    predictions: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        make_walk_forward_splits(
            event_ids,
            min_train_events=min_train_events,
            val_step_events=val_step_events,
        )
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_xgboost()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        for i in range(len(val_idx)):
            predictions.append(
                {
                    "event_id": event_ids[val_idx[i]],
                    "y_pred": float(y_pred[i]),
                    "y_actual": float(y_val[i]),
                    "fold": fold_idx,
                }
            )

        n_train_events = len(set(event_ids[train_idx]))
        n_val_events = len(set(event_ids[val_idx]))
        print(
            f"  Fold {fold_idx}: train={n_train_events} events, "
            f"val={n_val_events} events, "
            f"predictions={len(y_pred)}"
        )

    return predictions


def simulate_betting(
    predictions: list[dict[str, Any]],
    betting_ctx: dict[str, BettingContext],
    threshold: float,
    stake: float = STAKE,
) -> dict[str, Any]:
    """Simulate P&L for a single threshold."""
    bets: list[dict[str, Any]] = []

    for pred in predictions:
        eid = pred["event_id"]
        bc = betting_ctx.get(eid)
        if bc is None:
            continue

        y_pred = pred["y_pred"]

        if y_pred > threshold:
            bet_side = "home"
            american_odds = bc.home_american_odds
            won = bc.home_wins
        elif y_pred < -threshold:
            bet_side = "away"
            american_odds = bc.away_american_odds
            won = not bc.home_wins
        else:
            continue

        decimal_odds = american_to_decimal(american_odds)
        pnl = stake * (decimal_odds - 1) if won else -stake

        # CLV: actual delta, sign-adjusted for bet direction
        sign = 1.0 if bet_side == "home" else -1.0
        clv = pred["y_actual"] * sign

        bets.append(
            {
                "event_id": eid,
                "fold": pred["fold"],
                "y_pred": pred["y_pred"],
                "y_actual": pred["y_actual"],
                "bet_side": bet_side,
                "american_odds": american_odds,
                "decimal_odds": decimal_odds,
                "won": won,
                "pnl": pnl,
                "clv": clv,
            }
        )

    if not bets:
        return {
            "threshold": threshold,
            "n_bets": 0,
            "n_home": 0,
            "n_away": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_staked": 0.0,
            "roi": 0.0,
            "avg_clv": 0.0,
            "bets": [],
        }

    n_home = sum(1 for b in bets if b["bet_side"] == "home")
    n_away = sum(1 for b in bets if b["bet_side"] == "away")
    wins = sum(1 for b in bets if b["won"])
    total_pnl = sum(b["pnl"] for b in bets)
    total_staked = stake * len(bets)
    avg_clv = float(np.mean([b["clv"] for b in bets]))

    return {
        "threshold": threshold,
        "n_bets": len(bets),
        "n_home": n_home,
        "n_away": n_away,
        "win_rate": wins / len(bets),
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": total_pnl / total_staked,
        "avg_clv": avg_clv,
        "bets": bets,
    }


def run_permutation_baseline(
    predictions: list[dict[str, Any]],
    betting_ctx: dict[str, BettingContext],
    threshold: float,
    observed_roi: float,
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, float]:
    """Shuffle predictions and compute null ROI distribution."""
    rng = np.random.default_rng(42)
    perm_rois: list[float] = []

    y_preds = np.array([p["y_pred"] for p in predictions])

    for _ in range(n_permutations):
        shuffled_preds = [dict(p) for p in predictions]
        shuffled_y = rng.permutation(y_preds)
        for i, sp in enumerate(shuffled_preds):
            sp["y_pred"] = float(shuffled_y[i])

        result = simulate_betting(shuffled_preds, betting_ctx, threshold)
        if result["n_bets"] > 0:
            perm_rois.append(result["roi"])

    if not perm_rois:
        return {"mean_roi": 0.0, "std_roi": 0.0, "p_value": 1.0}

    perm_arr = np.array(perm_rois)
    p_value = float(np.mean(perm_arr >= observed_roi))

    return {
        "mean_roi": float(np.mean(perm_arr)),
        "std_roi": float(np.std(perm_arr)),
        "p_value": p_value,
    }


def always_home_baseline(
    predictions: list[dict[str, Any]],
    betting_ctx: dict[str, BettingContext],
    stake: float = STAKE,
) -> dict[str, Any]:
    """Bet home on every val event to establish house edge baseline."""
    bets: list[dict[str, Any]] = []

    for pred in predictions:
        eid = pred["event_id"]
        bc = betting_ctx.get(eid)
        if bc is None:
            continue

        decimal_odds = american_to_decimal(bc.home_american_odds)
        won = bc.home_wins
        pnl = stake * (decimal_odds - 1) if won else -stake

        bets.append({"event_id": eid, "won": won, "pnl": pnl})

    if not bets:
        return {"n_bets": 0, "win_rate": 0.0, "roi": 0.0, "total_pnl": 0.0}

    wins = sum(1 for b in bets if b["won"])
    total_pnl = sum(b["pnl"] for b in bets)
    total_staked = stake * len(bets)

    return {
        "n_bets": len(bets),
        "win_rate": wins / len(bets),
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": total_pnl / total_staked,
    }


def plot_threshold_sweep(sweep_df: pd.DataFrame) -> None:
    """Plot ROI and n_bets vs threshold (dual y-axis)."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_roi = "#2196F3"
    color_bets = "#FF9800"

    ax1.bar(
        sweep_df["threshold"].astype(str),
        sweep_df["roi"] * 100,
        color=color_roi,
        alpha=0.7,
        label="ROI (%)",
    )
    ax1.set_xlabel("Minimum Predicted Movement Threshold")
    ax1.set_ylabel("ROI (%)", color=color_roi)
    ax1.tick_params(axis="y", labelcolor=color_roi)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        sweep_df["threshold"].astype(str),
        sweep_df["n_bets"],
        "o-",
        color=color_bets,
        linewidth=2,
        markersize=8,
        label="N bets",
    )
    ax2.set_ylabel("Number of Bets", color=color_bets)
    ax2.tick_params(axis="y", labelcolor=color_bets)

    # Add p-value annotations
    for _i, row in sweep_df.iterrows():
        p_str = f"p={row['p_value']:.3f}" if row["p_value"] < 1.0 else "p=1.0"
        ax1.annotate(
            p_str,
            (str(row["threshold"]), row["roi"] * 100),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="gray",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Experiment 7: ROI by Prediction Threshold (bet365, walk-forward)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved threshold_sweep.png")


def plot_equity_curve(bets: list[dict[str, Any]], threshold: float) -> None:
    """Plot cumulative P&L over time for a given threshold's bets."""
    if not bets:
        print("  No bets for equity curve")
        return

    cumulative_pnl = np.cumsum([b["pnl"] for b in bets])

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=1.5, color="#2196F3")
    ax.fill_between(
        range(len(cumulative_pnl)),
        cumulative_pnl,
        alpha=0.15,
        color="#2196F3",
    )
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # Mark fold boundaries
    folds = sorted({b["fold"] for b in bets})
    fold_starts = []
    for f in folds:
        idx = next(i for i, b in enumerate(bets) if b["fold"] == f)
        fold_starts.append(idx)

    for idx in fold_starts[1:]:
        ax.axvline(x=idx, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Bet Number (chronological)")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Equity Curve (threshold={threshold}, n={len(bets)} bets)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved equity_curve.png")


def write_findings(
    sweep_df: pd.DataFrame,
    home_baseline: dict[str, Any],
    n_events: int,
    n_predictions: int,
    n_with_ctx: int,
) -> None:
    """Write FINDINGS.md with results."""
    best_row = sweep_df.loc[sweep_df["roi"].idxmax()]
    best_threshold = best_row["threshold"]
    best_roi = best_row["roi"]
    best_pvalue = best_row["p_value"]

    lines = [
        "# Experiment 7: Walk-Forward Betting Simulation",
        "",
        "## Setup",
        "",
        f"- **Date**: {datetime.now(UTC).strftime('%Y-%m-%d')}",
        f"- **Dataset**: OddsPortal, {n_events} events, bet365 target",
        "- **Model**: XGBoost with tuned baseline params (tabular-only)",
        "- **CV**: Walk-forward expanding window (same splits as Exp 6)",
        f"- **Predictions**: {n_predictions} val-fold predictions, "
        f"{n_with_ctx} with betting context",
        "- **Sizing**: Flat $100 per bet",
        "- **Significance**: 1,000 permutation shuffles per threshold",
        "",
        "## Key Results",
        "",
        "### Threshold Sweep",
        "",
        "| Threshold | N Bets | Home | Away | Win Rate | P&L | ROI | Avg CLV | p-value |",
        "|-----------|--------|------|------|----------|-----|-----|---------|---------|",
    ]

    for _, row in sweep_df.iterrows():
        lines.append(
            f"| {row['threshold']:.3f} | {int(row['n_bets'])} | {int(row['n_home'])} | "
            f"{int(row['n_away'])} | {row['win_rate']:.1%} | "
            f"${row['total_pnl']:+,.0f} | {row['roi']:+.2%} | "
            f"{row['avg_clv']:+.4f} | {row['p_value']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Baselines",
            "",
            f"- **Always home**: {home_baseline['n_bets']} bets, "
            f"win rate {home_baseline['win_rate']:.1%}, "
            f"ROI {home_baseline['roi']:+.2%}, "
            f"P&L ${home_baseline['total_pnl']:+,.0f}",
            f"- **Permutation mean ROI** (at best threshold {best_threshold}): "
            f"{sweep_df.loc[sweep_df['threshold'] == best_threshold, 'perm_mean_roi'].iloc[0]:+.2%}",
            "",
            "### Best Threshold",
            "",
            f"- **Threshold**: {best_threshold}",
            f"- **ROI**: {best_roi:+.2%}",
            f"- **p-value**: {best_pvalue:.3f}",
            "",
            "## Interpretation",
            "",
        ]
    )

    if best_pvalue < 0.05:
        lines.append(
            f"The model achieves statistically significant ROI of {best_roi:+.2%} "
            f"at threshold={best_threshold} (p={best_pvalue:.3f}). "
            "The CLV prediction signal translates to profitable betting after vig."
        )
    elif best_roi > 0:
        lines.append(
            f"The model achieves positive ROI of {best_roi:+.2%} at threshold={best_threshold}, "
            f"but this is NOT statistically significant (p={best_pvalue:.3f}). "
            "The result could be due to chance — the CLV signal may not overcome the vig."
        )
    else:
        lines.append(
            "The model does not achieve positive ROI at any threshold. "
            "The ~3.6% R² CLV signal is too weak to overcome the bookmaker's vig."
        )

    lines.extend(
        [
            "",
            f"The always-home baseline ROI of {home_baseline['roi']:+.2%} confirms the "
            "house edge the model must overcome.",
            "",
            "## Implications",
            "",
        ]
    )

    if best_pvalue < 0.05:
        lines.extend(
            [
                "- Signal is viable for flat-bet execution on bet365 at the optimal threshold",
                "- Next step: Kelly sizing (Exp 8) to optimize bankroll growth",
                "- Cross-venue execution (sportsbook vs Polymarket) may extract more value",
            ]
        )
    else:
        lines.extend(
            [
                "- Public sportsbook features alone are insufficient for profitable betting",
                "- Consider: non-public features (order flow, bettor identity), "
                "cross-venue execution, or accepting the edge is too thin",
                "- The signal may still have value for market-making or spread strategies "
                "where transaction costs are lower",
            ]
        )

    lines.append("")
    (OUTPUT_DIR / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print("  Saved FINDINGS.md")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    print("\nLoading OddsPortal training data (tabular-only, sharp tier)...")
    X, y, feature_names, event_ids = await load_data()
    n_events = len(set(event_ids))
    print(f"  Loaded {len(X)} rows, {n_events} events, {len(feature_names)} features")

    print("\nLoading betting context (bet365 odds + game outcomes)...")
    betting_ctx = await load_betting_context(event_ids)

    # ── Walk-Forward Predictions ──
    print("\n" + "=" * 60)
    print("Walk-forward predictions")
    print("=" * 60)

    config = MLTrainingConfig.from_yaml(str(CONFIG_PATH))
    min_train = config.training.data.min_train_events or 2000
    val_step = config.training.data.val_step_events or 200

    print(f"\n  min_train_events={min_train}, val_step_events={val_step}")
    predictions = walk_forward_predict(X, y, event_ids, min_train, val_step)
    print(f"\n  Total predictions: {len(predictions)}")

    # Count how many predictions have betting context
    n_with_ctx = sum(1 for p in predictions if p["event_id"] in betting_ctx)
    print(f"  With betting context: {n_with_ctx}")

    # ── Threshold Sweep ──
    print("\n" + "=" * 60)
    print("Threshold sweep")
    print("=" * 60)

    sweep_rows: list[dict[str, Any]] = []

    for threshold in THRESHOLDS:
        print(f"\n  Threshold {threshold}:")
        result = simulate_betting(predictions, betting_ctx, threshold)

        print(
            f"    {result['n_bets']} bets "
            f"(home={result['n_home']}, away={result['n_away']}), "
            f"win rate={result['win_rate']:.1%}, "
            f"ROI={result['roi']:+.2%}, "
            f"P&L=${result['total_pnl']:+,.0f}"
        )

        # Permutation test
        if result["n_bets"] > 0:
            print(f"    Running {N_PERMUTATIONS} permutations...")
            perm = run_permutation_baseline(predictions, betting_ctx, threshold, result["roi"])
            print(
                f"    Permutation: mean ROI={perm['mean_roi']:+.2%}, p-value={perm['p_value']:.3f}"
            )
        else:
            perm = {"mean_roi": 0.0, "std_roi": 0.0, "p_value": 1.0}

        sweep_rows.append(
            {
                "threshold": threshold,
                "n_bets": result["n_bets"],
                "n_home": result["n_home"],
                "n_away": result["n_away"],
                "win_rate": result["win_rate"],
                "total_pnl": result["total_pnl"],
                "total_staked": result["total_staked"],
                "roi": result["roi"],
                "avg_clv": result["avg_clv"],
                "perm_mean_roi": perm["mean_roi"],
                "perm_std_roi": perm["std_roi"],
                "p_value": perm["p_value"],
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    print("\n  Saved threshold_sweep.csv")

    # ── Save All Bets (at lowest threshold for full coverage) ──
    full_result = simulate_betting(predictions, betting_ctx, THRESHOLDS[0])
    if full_result["bets"]:
        bets_df = pd.DataFrame(full_result["bets"])
        bets_df.to_csv(OUTPUT_DIR / "all_bets.csv", index=False)
        print(f"  Saved all_bets.csv ({len(full_result['bets'])} bets)")

    # ── Always-Home Baseline ──
    print("\n" + "=" * 60)
    print("Always-home baseline")
    print("=" * 60)

    home_baseline = always_home_baseline(predictions, betting_ctx)
    print(
        f"\n  {home_baseline['n_bets']} bets, "
        f"win rate={home_baseline['win_rate']:.1%}, "
        f"ROI={home_baseline['roi']:+.2%}, "
        f"P&L=${home_baseline['total_pnl']:+,.0f}"
    )

    # ── Plots & Findings ──
    print("\n" + "=" * 60)
    print("Generating outputs")
    print("=" * 60)

    plot_threshold_sweep(sweep_df)

    # Equity curve for best ROI threshold
    best_threshold = sweep_df.loc[sweep_df["roi"].idxmax(), "threshold"]
    best_result = simulate_betting(predictions, betting_ctx, best_threshold)
    plot_equity_curve(best_result["bets"], best_threshold)

    write_findings(sweep_df, home_baseline, n_events, len(predictions), n_with_ctx)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
