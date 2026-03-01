"""Experiment 7b: Line Shopping Across Bookmakers.

Exp 7 showed the ~3.6% R² CLV signal is directionally correct but unprofitable
when executing at bet365's vigged odds alone (-2.4% to -11.8% ROI at thresholds
≤0.03). The signal exists; the execution cost is too high.

Each OddsPortal snapshot already contains odds from 4 UK bookmakers (bet365,
betway, betfred, bwin). Exp 7 filters to bet365 only. Line shopping — executing
at the best available price across all bookmakers — is the simplest execution
optimization and directly reduces effective vig.

Re-runs the Exp 7 simulation identically, but for each bet, selects the
bookmaker offering the best odds for the predicted side. Compares side-by-side:
bet365-only vs best-available.

Outputs saved to experiments/results/exp7b_line_shopping/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.sequence_loader import extract_odds_from_snapshot
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_analytics.utils import american_to_decimal, calculate_implied_probability
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import select
from xgboost import XGBRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7b_line_shopping"
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
ALL_BOOKMAKERS = ["bet365", "betway", "betfred", "bwin"]


@dataclass
class BookmakerOdds:
    bookmaker: str
    american_odds: int
    decimal_odds: float


@dataclass
class BettingContext:
    home_odds: list[BookmakerOdds] = field(default_factory=list)
    away_odds: list[BookmakerOdds] = field(default_factory=list)
    home_wins: bool = False


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
    """Load all bookmaker h2h odds and game outcomes for training events.

    For each event, picks the sharp-tier snapshot (most recent with hours_before >= 12),
    extracts h2h American odds for ALL available bookmakers, and determines game outcome.
    """
    unique_ids = list(dict.fromkeys(event_ids))
    print(f"  Loading betting context for {len(unique_ids)} events...")

    ctx: dict[str, BettingContext] = {}

    async with async_session_maker() as session:
        event_result = await session.execute(select(Event).where(Event.id.in_(unique_ids)))
        events_by_id: dict[str, Event] = {e.id: e for e in event_result.scalars().all()}

        snapshot_result = await session.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.event_id.in_(unique_ids))
            .where(OddsSnapshot.fetch_tier == "sharp")
            .order_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time.desc())
        )
        snapshots = list(snapshot_result.scalars().all())

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

        odds_list = extract_odds_from_snapshot(snapshot, eid, market="h2h")

        home_odds: list[BookmakerOdds] = []
        away_odds: list[BookmakerOdds] = []

        for o in odds_list:
            if o.bookmaker_key not in ALL_BOOKMAKERS:
                continue
            bm_odds = BookmakerOdds(
                bookmaker=o.bookmaker_key,
                american_odds=o.price,
                decimal_odds=american_to_decimal(o.price),
            )
            if o.outcome_name == event.home_team:
                home_odds.append(bm_odds)
            elif o.outcome_name == event.away_team:
                away_odds.append(bm_odds)

        # Require at least bet365 for baseline comparison
        has_bet365_home = any(o.bookmaker == "bet365" for o in home_odds)
        has_bet365_away = any(o.bookmaker == "bet365" for o in away_odds)
        if not has_bet365_home or not has_bet365_away:
            skipped_no_odds += 1
            continue

        ctx[eid] = BettingContext(
            home_odds=home_odds,
            away_odds=away_odds,
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

    # Bookmaker coverage stats
    n_bookmakers_per_event = []
    for bc in ctx.values():
        n_bookmakers_per_event.append(len(bc.home_odds))
    if n_bookmakers_per_event:
        print(
            f"  Avg bookmakers per event: {np.mean(n_bookmakers_per_event):.1f} "
            f"(min={min(n_bookmakers_per_event)}, max={max(n_bookmakers_per_event)})"
        )

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


def _pick_odds(odds_list: list[BookmakerOdds], line_shop: bool) -> BookmakerOdds | None:
    """Pick bookmaker odds: best available if line_shop, else bet365 only."""
    if not odds_list:
        return None
    if line_shop:
        return max(odds_list, key=lambda o: o.decimal_odds)
    bet365 = [o for o in odds_list if o.bookmaker == "bet365"]
    return bet365[0] if bet365 else None


def simulate_betting(
    predictions: list[dict[str, Any]],
    betting_ctx: dict[str, BettingContext],
    threshold: float,
    *,
    line_shop: bool = False,
    stake: float = STAKE,
) -> dict[str, Any]:
    """Simulate P&L for a single threshold.

    When line_shop=True, pick max(decimal_odds) across all bookmakers.
    When line_shop=False, use bet365 only (reproduces Exp 7).
    """
    bets: list[dict[str, Any]] = []

    for pred in predictions:
        eid = pred["event_id"]
        bc = betting_ctx.get(eid)
        if bc is None:
            continue

        y_pred = pred["y_pred"]

        if y_pred > threshold:
            bet_side = "home"
            selected = _pick_odds(bc.home_odds, line_shop)
            won = bc.home_wins
        elif y_pred < -threshold:
            bet_side = "away"
            selected = _pick_odds(bc.away_odds, line_shop)
            won = not bc.home_wins
        else:
            continue

        if selected is None:
            continue

        pnl = stake * (selected.decimal_odds - 1) if won else -stake

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
                "bookmaker": selected.bookmaker,
                "american_odds": selected.american_odds,
                "decimal_odds": selected.decimal_odds,
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
    *,
    line_shop: bool = False,
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

        result = simulate_betting(shuffled_preds, betting_ctx, threshold, line_shop=line_shop)
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


def compute_vig_stats(betting_ctx: dict[str, BettingContext]) -> dict[str, Any]:
    """Compute effective vig (overround) for bet365-only vs best-available."""
    bet365_overrounds: list[float] = []
    best_overrounds: list[float] = []

    for bc in betting_ctx.values():
        # bet365 overround
        bet365_home = [o for o in bc.home_odds if o.bookmaker == "bet365"]
        bet365_away = [o for o in bc.away_odds if o.bookmaker == "bet365"]
        if bet365_home and bet365_away:
            ip_home = calculate_implied_probability(bet365_home[0].american_odds)
            ip_away = calculate_implied_probability(bet365_away[0].american_odds)
            bet365_overrounds.append((ip_home + ip_away - 1) * 100)

        # Best-available overround (best home from any book + best away from any book)
        if bc.home_odds and bc.away_odds:
            best_home = max(bc.home_odds, key=lambda o: o.decimal_odds)
            best_away = max(bc.away_odds, key=lambda o: o.decimal_odds)
            ip_home = 1.0 / best_home.decimal_odds
            ip_away = 1.0 / best_away.decimal_odds
            best_overrounds.append((ip_home + ip_away - 1) * 100)

    return {
        "bet365_mean_vig": float(np.mean(bet365_overrounds)) if bet365_overrounds else 0.0,
        "bet365_median_vig": float(np.median(bet365_overrounds)) if bet365_overrounds else 0.0,
        "best_mean_vig": float(np.mean(best_overrounds)) if best_overrounds else 0.0,
        "best_median_vig": float(np.median(best_overrounds)) if best_overrounds else 0.0,
        "n_events": len(bet365_overrounds),
    }


def compute_bookmaker_frequency(bets: list[dict[str, Any]]) -> pd.DataFrame:
    """Count how often each bookmaker is selected when line shopping."""
    if not bets:
        return pd.DataFrame(columns=["bookmaker", "side", "count", "pct"])

    rows: list[dict[str, Any]] = []
    for side in ["home", "away"]:
        side_bets = [b for b in bets if b["bet_side"] == side]
        if not side_bets:
            continue
        for bm in ALL_BOOKMAKERS:
            count = sum(1 for b in side_bets if b["bookmaker"] == bm)
            rows.append(
                {
                    "bookmaker": bm,
                    "side": side,
                    "count": count,
                    "pct": count / len(side_bets) * 100 if side_bets else 0.0,
                }
            )
    # Also add totals
    for bm in ALL_BOOKMAKERS:
        count = sum(1 for b in bets if b["bookmaker"] == bm)
        rows.append(
            {
                "bookmaker": bm,
                "side": "total",
                "count": count,
                "pct": count / len(bets) * 100 if bets else 0.0,
            }
        )

    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep_df: pd.DataFrame) -> None:
    """Plot side-by-side ROI bars for bet365-only vs best-available."""
    fig, ax = plt.subplots(figsize=(14, 8))

    thresholds = sweep_df["threshold"].unique()
    x = np.arange(len(thresholds))
    width = 0.35

    bet365_df = sweep_df[sweep_df["mode"] == "bet365"]
    best_df = sweep_df[sweep_df["mode"] == "best_available"]

    bars1 = ax.bar(
        x - width / 2,
        bet365_df["roi"].values * 100,
        width,
        label="bet365 only",
        color="#2196F3",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        best_df["roi"].values * 100,
        width,
        label="Best available",
        color="#4CAF50",
        alpha=0.8,
    )

    # Annotate p-values
    for bars, df in [(bars1, bet365_df), (bars2, best_df)]:
        for bar, (_, row) in zip(bars, df.iterrows(), strict=True):
            p_str = f"p={row['p_value']:.3f}" if row["p_value"] < 1.0 else "p=1.0"
            ax.annotate(
                p_str,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
                color="gray",
            )

    ax.set_xlabel("Minimum Predicted Movement Threshold")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Exp 7b: ROI by Threshold — bet365 Only vs Best Available (Line Shopping)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.3f}" for t in thresholds])
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add n_bets as secondary annotation
    for i, _t in enumerate(thresholds):
        n_b365 = int(bet365_df.iloc[i]["n_bets"])
        n_best = int(best_df.iloc[i]["n_bets"])
        ax.annotate(
            f"n={n_b365}/{n_best}",
            (i, ax.get_ylim()[0]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="gray",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved threshold_sweep.png")


def plot_equity_curves(
    bets_b365: list[dict[str, Any]],
    bets_best: list[dict[str, Any]],
    threshold: float,
) -> None:
    """Plot cumulative P&L for bet365-only and best-available on same axes."""
    if not bets_b365 and not bets_best:
        print("  No bets for equity curve")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    for bets, label, color in [
        (bets_b365, "bet365 only", "#2196F3"),
        (bets_best, "Best available", "#4CAF50"),
    ]:
        if not bets:
            continue
        cumulative_pnl = np.cumsum([b["pnl"] for b in bets])
        ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=1.5, color=color, label=label)
        ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.1, color=color)

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # Mark fold boundaries from bet365 bets
    ref_bets = bets_b365 or bets_best
    if ref_bets:
        folds = sorted({b["fold"] for b in ref_bets})
        for f in folds[1:]:
            idx = next((i for i, b in enumerate(ref_bets) if b["fold"] == f), None)
            if idx is not None:
                ax.axvline(x=idx, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Bet Number (chronological)")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Equity Curves (threshold={threshold})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved equity_curve.png")


def write_findings(
    sweep_df: pd.DataFrame,
    vig_stats: dict[str, Any],
    freq_df: pd.DataFrame,
    n_events: int,
    n_predictions: int,
    n_with_ctx: int,
) -> None:
    """Write FINDINGS.md with side-by-side results."""
    bet365_df = sweep_df[sweep_df["mode"] == "bet365"].reset_index(drop=True)
    best_df = sweep_df[sweep_df["mode"] == "best_available"].reset_index(drop=True)

    # Best threshold for each mode
    b365_best = bet365_df.loc[bet365_df["roi"].idxmax()]
    best_best = best_df.loc[best_df["roi"].idxmax()]

    lines = [
        "# Experiment 7b: Line Shopping Across Bookmakers",
        "",
        "## Setup",
        "",
        f"- **Date**: {datetime.now(UTC).strftime('%Y-%m-%d')}",
        f"- **Dataset**: OddsPortal, {n_events} events, bet365 target (devigged CLV)",
        "- **Bookmakers**: bet365, betway, betfred, bwin (UK OddsPortal set)",
        "- **Model**: XGBoost with tuned baseline params (tabular-only, same as Exp 7)",
        "- **CV**: Walk-forward expanding window (same splits as Exp 7)",
        f"- **Predictions**: {n_predictions} val-fold predictions, "
        f"{n_with_ctx} with betting context",
        "- **Sizing**: Flat $100 per bet",
        "- **Significance**: 1,000 permutation shuffles per threshold per mode",
        "",
        "## Vig Reduction",
        "",
        f"- **bet365 mean overround**: {vig_stats['bet365_mean_vig']:.2f}% "
        f"(median {vig_stats['bet365_median_vig']:.2f}%)",
        f"- **Best-available mean overround**: {vig_stats['best_mean_vig']:.2f}% "
        f"(median {vig_stats['best_median_vig']:.2f}%)",
        f"- **Vig reduction**: "
        f"{vig_stats['bet365_mean_vig'] - vig_stats['best_mean_vig']:.2f} percentage points",
        f"- **Events**: {vig_stats['n_events']}",
        "",
        "## Key Results",
        "",
        "### Threshold Sweep — bet365 Only",
        "",
        "| Threshold | N Bets | Win Rate | P&L | ROI | Avg CLV | p-value |",
        "|-----------|--------|----------|-----|-----|---------|---------|",
    ]

    for _, row in bet365_df.iterrows():
        lines.append(
            f"| {row['threshold']:.3f} | {int(row['n_bets'])} | {row['win_rate']:.1%} | "
            f"${row['total_pnl']:+,.0f} | {row['roi']:+.2%} | "
            f"{row['avg_clv']:+.4f} | {row['p_value']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Threshold Sweep — Best Available (Line Shopping)",
            "",
            "| Threshold | N Bets | Win Rate | P&L | ROI | Avg CLV | p-value |",
            "|-----------|--------|----------|-----|-----|---------|---------|",
        ]
    )

    for _, row in best_df.iterrows():
        lines.append(
            f"| {row['threshold']:.3f} | {int(row['n_bets'])} | {row['win_rate']:.1%} | "
            f"${row['total_pnl']:+,.0f} | {row['roi']:+.2%} | "
            f"{row['avg_clv']:+.4f} | {row['p_value']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Side-by-Side Comparison",
            "",
            "| Threshold | bet365 ROI | Best ROI | ROI Improvement |",
            "|-----------|-----------|----------|-----------------|",
        ]
    )

    for i in range(len(bet365_df)):
        b365_roi = bet365_df.iloc[i]["roi"]
        best_roi = best_df.iloc[i]["roi"]
        improvement = best_roi - b365_roi
        lines.append(
            f"| {bet365_df.iloc[i]['threshold']:.3f} | "
            f"{b365_roi:+.2%} | {best_roi:+.2%} | "
            f"{improvement:+.2%} |"
        )

    lines.extend(
        [
            "",
            "### Bookmaker Selection Frequency (Best Available, lowest threshold)",
            "",
            "| Bookmaker | Home | Away | Total |",
            "|-----------|------|------|-------|",
        ]
    )

    total_freq = freq_df[freq_df["side"] == "total"]
    home_freq = freq_df[freq_df["side"] == "home"]
    away_freq = freq_df[freq_df["side"] == "away"]

    for bm in ALL_BOOKMAKERS:
        home_row = home_freq[home_freq["bookmaker"] == bm]
        away_row = away_freq[away_freq["bookmaker"] == bm]
        total_row = total_freq[total_freq["bookmaker"] == bm]
        h_pct = f"{home_row.iloc[0]['pct']:.1f}%" if len(home_row) > 0 else "—"
        a_pct = f"{away_row.iloc[0]['pct']:.1f}%" if len(away_row) > 0 else "—"
        t_pct = f"{total_row.iloc[0]['pct']:.1f}%" if len(total_row) > 0 else "—"
        lines.append(f"| {bm} | {h_pct} | {a_pct} | {t_pct} |")

    lines.extend(
        [
            "",
            "### Best Thresholds",
            "",
            f"- **bet365 only**: threshold={b365_best['threshold']}, "
            f"ROI={b365_best['roi']:+.2%}, p={b365_best['p_value']:.3f}",
            f"- **Best available**: threshold={best_best['threshold']}, "
            f"ROI={best_best['roi']:+.2%}, p={best_best['p_value']:.3f}",
            "",
            "## Interpretation",
            "",
        ]
    )

    # Determine improvement
    improvement_at_all = all(
        best_df.iloc[i]["roi"] >= bet365_df.iloc[i]["roi"] for i in range(len(bet365_df))
    )
    best_profitable = best_best["roi"] > 0
    best_significant = best_best["p_value"] < 0.05

    if best_significant:
        lines.append(
            f"Line shopping achieves statistically significant ROI of "
            f"{best_best['roi']:+.2%} at threshold={best_best['threshold']} "
            f"(p={best_best['p_value']:.3f}). The vig reduction from shopping "
            f"across {len(ALL_BOOKMAKERS)} bookmakers is sufficient to make the "
            f"~3.6% R² CLV signal profitable."
        )
    elif best_profitable:
        lines.append(
            f"Line shopping improves ROI at the best threshold from "
            f"{b365_best['roi']:+.2%} (bet365) to {best_best['roi']:+.2%} "
            f"(best available), but this is NOT statistically significant "
            f"(p={best_best['p_value']:.3f})."
        )
    else:
        lines.append(
            "Line shopping reduces execution costs but is insufficient to make "
            "the signal profitable. The ~3.6% R² CLV signal remains too weak to "
            f"overcome the vig even with {len(ALL_BOOKMAKERS)}-bookmaker shopping."
        )

    if improvement_at_all:
        lines.append(
            "\nLine shopping improves ROI at every threshold, confirming that "
            "best-price execution can only help (or tie) versus single-book execution."
        )

    vig_reduction = vig_stats["bet365_mean_vig"] - vig_stats["best_mean_vig"]
    lines.extend(
        [
            "",
            f"Effective vig drops from {vig_stats['bet365_mean_vig']:.2f}% (bet365) to "
            f"{vig_stats['best_mean_vig']:.2f}% (best-available), a "
            f"{vig_reduction:.2f}pp reduction.",
            "",
            "## Implications",
            "",
        ]
    )

    if best_significant:
        lines.extend(
            [
                "- Line shopping across UK bookmakers makes the CLV signal profitable",
                "- Next steps: Kelly sizing (position sizing), cross-venue (Polymarket) execution",
                "- The strategy requires accounts at multiple bookmakers — "
                "account limits/restrictions are the practical constraint",
            ]
        )
    elif best_profitable:
        lines.extend(
            [
                "- Line shopping is a meaningful improvement but insufficient alone",
                "- The effective vig reduction may become significant with a larger sample",
                "- Consider combining with cross-venue execution (Polymarket) for additional vig reduction",
            ]
        )
    else:
        lines.extend(
            [
                "- Even with 4-bookmaker line shopping, the signal is too weak for profitable flat betting",
                "- The vig floor (~2-3% even after shopping) exceeds the ~3.6% R² signal strength",
                "- Would need either stronger features or lower-cost venues (exchanges, Polymarket)",
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

    print("\nLoading betting context (all bookmaker odds + game outcomes)...")
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

    n_with_ctx = sum(1 for p in predictions if p["event_id"] in betting_ctx)
    print(f"  With betting context: {n_with_ctx}")

    # ── Vig Analysis ──
    print("\n" + "=" * 60)
    print("Vig analysis")
    print("=" * 60)

    vig_stats = compute_vig_stats(betting_ctx)
    print(f"\n  bet365 mean overround: {vig_stats['bet365_mean_vig']:.2f}%")
    print(f"  Best-available mean overround: {vig_stats['best_mean_vig']:.2f}%")
    print(f"  Vig reduction: {vig_stats['bet365_mean_vig'] - vig_stats['best_mean_vig']:.2f}pp")

    # ── Threshold Sweep (both modes) ──
    print("\n" + "=" * 60)
    print("Threshold sweep")
    print("=" * 60)

    sweep_rows: list[dict[str, Any]] = []

    for mode, line_shop in [("bet365", False), ("best_available", True)]:
        print(f"\n  --- Mode: {mode} ---")

        for threshold in THRESHOLDS:
            print(f"\n  Threshold {threshold}:")
            result = simulate_betting(predictions, betting_ctx, threshold, line_shop=line_shop)

            print(
                f"    {result['n_bets']} bets "
                f"(home={result['n_home']}, away={result['n_away']}), "
                f"win rate={result['win_rate']:.1%}, "
                f"ROI={result['roi']:+.2%}, "
                f"P&L=${result['total_pnl']:+,.0f}"
            )

            if result["n_bets"] > 0:
                print(f"    Running {N_PERMUTATIONS} permutations...")
                perm = run_permutation_baseline(
                    predictions,
                    betting_ctx,
                    threshold,
                    result["roi"],
                    line_shop=line_shop,
                )
                print(
                    f"    Permutation: mean ROI={perm['mean_roi']:+.2%}, "
                    f"p-value={perm['p_value']:.3f}"
                )
            else:
                perm = {"mean_roi": 0.0, "std_roi": 0.0, "p_value": 1.0}

            sweep_rows.append(
                {
                    "mode": mode,
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

    # ── Save All Bets (at lowest threshold, both modes) ──
    all_bets_rows: list[dict[str, Any]] = []
    for mode, line_shop in [("bet365", False), ("best_available", True)]:
        result = simulate_betting(predictions, betting_ctx, THRESHOLDS[0], line_shop=line_shop)
        for b in result["bets"]:
            b["mode"] = mode
            all_bets_rows.append(b)

    if all_bets_rows:
        bets_df = pd.DataFrame(all_bets_rows)
        bets_df.to_csv(OUTPUT_DIR / "all_bets.csv", index=False)
        print(f"  Saved all_bets.csv ({len(all_bets_rows)} bets)")

    # ── Bookmaker Frequency ──
    best_result_lowest = simulate_betting(predictions, betting_ctx, THRESHOLDS[0], line_shop=True)
    freq_df = compute_bookmaker_frequency(best_result_lowest["bets"])
    freq_df.to_csv(OUTPUT_DIR / "bookmaker_frequency.csv", index=False)
    print("  Saved bookmaker_frequency.csv")

    print("\n  Bookmaker selection frequency (best-available, lowest threshold):")
    total_freq = freq_df[freq_df["side"] == "total"]
    for _, row in total_freq.iterrows():
        print(f"    {row['bookmaker']}: {row['count']} ({row['pct']:.1f}%)")

    # ── Plots ──
    print("\n" + "=" * 60)
    print("Generating outputs")
    print("=" * 60)

    plot_threshold_sweep(sweep_df)

    # Equity curves at the best threshold (best ROI across both modes)
    best_threshold = sweep_df.loc[sweep_df["roi"].idxmax(), "threshold"]
    bets_b365 = simulate_betting(predictions, betting_ctx, best_threshold, line_shop=False)["bets"]
    bets_best = simulate_betting(predictions, betting_ctx, best_threshold, line_shop=True)["bets"]
    plot_equity_curves(bets_b365, bets_best, best_threshold)

    # ── Findings ──
    write_findings(sweep_df, vig_stats, freq_df, n_events, len(predictions), n_with_ctx)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
