"""Fold-level diagnostics for walk-forward CV.

Analyzes each fold for:
- Date range and data source composition
- Pinnacle coverage (events with/without sharp closing)
- Target distribution shifts
- Feature statistics and coverage
- Identifies potential regime changes

Usage:
    uv run python experiments/scripts/fold_diagnostics.py
    uv run python experiments/scripts/fold_diagnostics.py --protocol expanding:150
    uv run python experiments/scripts/fold_diagnostics.py --protocol sliding-760:150
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import prepare_training_data_from_config
from odds_core.database import async_session_maker
from odds_core.models import Event
from sqlalchemy import select

logger = structlog.get_logger()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "fold_diagnostics"


async def load_event_metadata(
    event_ids: list[str],
) -> dict[str, dict]:
    """Load event metadata (dates, source) from the database."""
    async with async_session_maker() as session:
        result = await session.execute(select(Event).where(Event.id.in_(event_ids)))
        events = result.scalars().all()

    metadata = {}
    for event in events:
        source = "unknown"
        if event.id.startswith("op_"):
            source = "oddsportal"
        elif event.id.startswith("fduk_"):
            source = "football_data_uk"
        metadata[event.id] = {
            "commence_time": event.commence_time,
            "source": source,
        }
    return metadata


async def load_all_epl_events() -> pd.DataFrame:
    """Load ALL EPL events (including those excluded from training)."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Event).where(
                Event.sport_key == "soccer_epl",
                Event.status == "final",
            )
        )
        events = result.scalars().all()

    rows = []
    for event in events:
        rows.append(
            {
                "event_id": event.id,
                "commence_time": event.commence_time,
                "source": "oddsportal"
                if event.id.startswith("op_")
                else "football_data_uk"
                if event.id.startswith("fduk_")
                else "other",
            }
        )

    return pd.DataFrame(rows).sort_values("commence_time").reset_index(drop=True)


async def main(protocol: str, config_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse protocol
    parts = protocol.split(":")
    val_step = int(parts[1])
    if parts[0] == "expanding":
        window_type = "expanding"
        max_train = None
        min_train = 700
    else:
        window_type = "sliding"
        max_train = int(parts[0].split("-")[1])
        min_train = max_train

    config_path = CONFIGS_DIR / config_name
    config = MLTrainingConfig.from_yaml(str(config_path))

    # Load training data
    logger.info("loading_data", config=config_name)
    async with async_session_maker() as session:
        data = await prepare_training_data_from_config(config, session)

    # Combine all data (same as tuner)
    X_all = np.concatenate(
        [a for a in [data.X_train, data.X_val, data.X_test] if a is not None and len(a) > 0]
    )
    y_all = np.concatenate(
        [a for a in [data.y_train, data.y_val, data.y_test] if a is not None and len(a) > 0]
    )
    event_ids_parts = []
    if data.event_ids_train is not None:
        event_ids_parts.append(data.event_ids_train)
    if data.event_ids_val is not None and len(data.event_ids_val) > 0:
        event_ids_parts.append(data.event_ids_val)
    event_ids_all = np.concatenate(event_ids_parts) if event_ids_parts else None

    assert event_ids_all is not None

    # Get unique event IDs in order
    unique_event_ids = list(dict.fromkeys(event_ids_all))

    # Load metadata
    logger.info("loading_event_metadata", n_events=len(unique_event_ids))
    event_meta = await load_event_metadata(unique_event_ids)

    # Load ALL EPL events to identify excluded events (missing Pinnacle)
    logger.info("loading_all_events")
    all_events_df = await load_all_epl_events()
    included_set = set(unique_event_ids)
    all_events_df["included"] = all_events_df["event_id"].isin(included_set)

    # Apply CV protocol
    config.training.data.window_type = window_type
    config.training.data.max_train_events = max_train
    config.training.data.min_train_events = min_train
    config.training.data.val_step_events = val_step

    # Generate walk-forward splits
    splits = make_walk_forward_splits(
        event_ids=event_ids_all,
        min_train_events=min_train,
        val_step_events=val_step,
        window_type=window_type,
        max_train_events=max_train,
    )

    feature_names = data.feature_names
    n_features = len(feature_names)

    print(f"\n{'=' * 100}")
    print(f"Fold Diagnostics: {protocol} / {config_name} ({n_features} features)")
    print(f"{'=' * 100}")

    all_fold_diagnostics = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_event_ids = list(dict.fromkeys(event_ids_all[val_idx]))
        train_event_ids = list(dict.fromkeys(event_ids_all[train_idx]))

        # Date ranges
        val_dates = [event_meta[eid]["commence_time"] for eid in val_event_ids if eid in event_meta]
        train_dates = [
            event_meta[eid]["commence_time"] for eid in train_event_ids if eid in event_meta
        ]

        val_start = min(val_dates) if val_dates else None
        val_end = max(val_dates) if val_dates else None
        train_start = min(train_dates) if train_dates else None
        train_end = max(train_dates) if train_dates else None

        # Data source breakdown
        val_sources = [event_meta[eid]["source"] for eid in val_event_ids if eid in event_meta]
        train_sources = [event_meta[eid]["source"] for eid in train_event_ids if eid in event_meta]

        val_op = sum(1 for s in val_sources if s == "oddsportal")
        val_fduk = sum(1 for s in val_sources if s == "football_data_uk")
        train_op = sum(1 for s in train_sources if s == "oddsportal")
        train_fduk = sum(1 for s in train_sources if s == "football_data_uk")

        # Target statistics
        y_val = y_all[val_idx]
        y_train = y_all[train_idx]

        # Feature statistics for validation set
        X_val = X_all[val_idx]
        X_train = X_all[train_idx]

        # NaN counts per feature in validation set
        val_nan_counts = np.isnan(X_val).sum(axis=0)
        train_nan_counts = np.isnan(X_train).sum(axis=0)

        # Feature means/stds for validation vs training
        with np.errstate(all="ignore"):
            val_means = np.nanmean(X_val, axis=0)
            train_means = np.nanmean(X_train, axis=0)
            train_stds = np.nanstd(X_train, axis=0)

        # Pinnacle exclusion analysis: how many events in this date range
        # were excluded from training (presumably missing Pinnacle closing)?
        if val_start and val_end:
            val_period_mask = (all_events_df["commence_time"] >= val_start) & (
                all_events_df["commence_time"] <= val_end
            )
            val_period_events = all_events_df[val_period_mask]
            val_total_events = len(val_period_events)
            val_included = int(val_period_events["included"].sum())
            val_excluded = val_total_events - val_included
        else:
            val_total_events = 0
            val_included = 0
            val_excluded = 0

        if train_start and train_end:
            train_period_mask = (all_events_df["commence_time"] >= train_start) & (
                all_events_df["commence_time"] <= train_end
            )
            train_period_events = all_events_df[train_period_mask]
            train_total_events_all = len(train_period_events)
            train_included = int(train_period_events["included"].sum())
            train_excluded = train_total_events_all - train_included
        else:
            train_total_events_all = 0
            train_included = 0
            train_excluded = 0

        fold_diag = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_date_start": str(val_start)[:10] if val_start else "?",
            "val_date_end": str(val_end)[:10] if val_end else "?",
            "train_date_start": str(train_start)[:10] if train_start else "?",
            "train_date_end": str(train_end)[:10] if train_end else "?",
            "val_op": val_op,
            "val_fduk": val_fduk,
            "train_op": train_op,
            "train_fduk": train_fduk,
            "val_target_mean": float(y_val.mean()),
            "val_target_std": float(y_val.std()),
            "train_target_mean": float(y_train.mean()),
            "train_target_std": float(y_train.std()),
            "val_total_events_in_period": int(val_total_events),
            "val_included": val_included,
            "val_excluded": val_excluded,
            "train_total_events_in_period": int(train_total_events_all),
            "train_included": train_included,
            "train_excluded": train_excluded,
        }

        # Feature-level NaN and distribution shift
        feature_diagnostics = []
        for i, fname in enumerate(feature_names):
            shift = abs(val_means[i] - train_means[i]) / (train_stds[i] + 1e-10)
            feature_diagnostics.append(
                {
                    "feature": fname,
                    "val_nan_pct": float(val_nan_counts[i] / len(val_idx) * 100),
                    "train_nan_pct": float(train_nan_counts[i] / len(train_idx) * 100),
                    "val_mean": float(val_means[i]),
                    "train_mean": float(train_means[i]),
                    "distribution_shift": float(shift),
                }
            )

        fold_diag["features"] = feature_diagnostics
        all_fold_diagnostics.append(fold_diag)

        # Print summary
        print(f"\n--- Fold {fold_idx} ---")
        print(
            f"  Train: {fold_diag['train_date_start']} → {fold_diag['train_date_end']}  "
            f"({len(train_idx)} samples, {train_op} OP + {train_fduk} FDUK)"
        )
        print(
            f"  Val:   {fold_diag['val_date_start']} → {fold_diag['val_date_end']}  "
            f"({len(val_idx)} samples, {val_op} OP + {val_fduk} FDUK)"
        )
        pct_excl_val = (val_excluded / val_total_events * 100) if val_total_events else 0
        pct_excl_train = (
            (train_excluded / train_total_events_all * 100) if train_total_events_all else 0
        )
        val_flag = " <<<" if pct_excl_val > 5 else ""
        print(
            f"  Events in period (val):   {val_included}/{val_total_events} included "
            f"({val_excluded} excluded, {pct_excl_val:.0f}%){val_flag}"
        )
        print(
            f"  Events in period (train): {train_included}/{train_total_events_all} included "
            f"({train_excluded} excluded, {pct_excl_train:.0f}%)"
        )
        print(
            f"  Target: val mean={y_val.mean():.6f} std={y_val.std():.4f} | "
            f"train mean={y_train.mean():.6f} std={y_train.std():.4f}"
        )

        # Show features with significant distribution shift
        shifted = [f for f in feature_diagnostics if f["distribution_shift"] > 0.3]
        if shifted:
            shifted.sort(key=lambda x: x["distribution_shift"], reverse=True)
            print("  Features with distribution shift > 0.3σ:")
            for f in shifted[:5]:
                print(
                    f"    {f['feature']:40s} shift={f['distribution_shift']:.2f}σ  "
                    f"(train={f['train_mean']:.4f}, val={f['val_mean']:.4f})"
                )

        # Show features with NaN differences
        nan_diff = [
            f for f in feature_diagnostics if abs(f["val_nan_pct"] - f["train_nan_pct"]) > 1.0
        ]
        if nan_diff:
            print("  Features with NaN rate change > 1%:")
            for f in nan_diff:
                print(
                    f"    {f['feature']:40s} train={f['train_nan_pct']:.1f}% → val={f['val_nan_pct']:.1f}%"
                )

    # Event exclusion timeline (events excluded = likely missing Pinnacle closing)
    print(f"\n{'=' * 100}")
    print("Event Exclusion Timeline (excluded events likely missing Pinnacle closing)")
    print(f"{'=' * 100}")
    all_events_df["month"] = all_events_df["commence_time"].dt.to_period("M")
    monthly = (
        all_events_df.groupby("month")
        .agg(
            total=("event_id", "count"),
            included=("included", "sum"),
        )
        .reset_index()
    )
    monthly["excluded"] = monthly["total"] - monthly["included"]
    monthly["pct_excluded"] = (monthly["excluded"] / monthly["total"] * 100).round(1)

    print(f"{'Month':>10} {'Total':>6} {'Included':>9} {'Excluded':>9} {'% Excluded':>11}")
    print("-" * 49)
    for _, row in monthly.iterrows():
        flag = " <<<" if row["pct_excluded"] > 5 else ""
        print(
            f"{str(row['month']):>10} {int(row['total']):>6} {int(row['included']):>9} "
            f"{int(row['excluded']):>9} {row['pct_excluded']:>10.1f}%{flag}"
        )

    # Save full results
    # Strip feature details for the summary JSON (keep it readable)
    summary = []
    for fd in all_fold_diagnostics:
        s = {k: v for k, v in fd.items() if k != "features"}
        summary.append(s)

    output_file = OUTPUT_DIR / f"diagnostics_{protocol.replace(':', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(all_fold_diagnostics, f, indent=2, default=str)

    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fold-level diagnostics")
    parser.add_argument(
        "--protocol",
        type=str,
        default="expanding:150",
        help="CV protocol (e.g., expanding:150, sliding-760:150)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xgboost_epl_combined_all_tuning.yaml",
        help="Config file name",
    )
    args = parser.parse_args()
    asyncio.run(main(args.protocol, args.config))
