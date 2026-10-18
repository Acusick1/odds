"""Walk-forward CV protocol comparison on a common validation range (#452).

Tunes each protocol cell with Optuna on its own walk-forward folds, refits the best
trial fold by fold to collect out-of-fold predictions, and scores every cell on the
same validation events: those every cell predicts, which start at the largest
``min_train_events``. Each cell is compared with the expanding cell by a paired
bootstrap over events, and with the per-fold baselines defined on #495.

Only tuning-range events (commence_time before 2024-08-01) are loaded; the run fails
if any row falls on or after that date.

Usage:
    uv run python experiments/scripts/cv_protocol_common_range.py
    uv run python experiments/scripts/cv_protocol_common_range.py --n-trials 5 --workers 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import prepare_training_data_from_config
from odds_analytics.training.tuner import OptunaTuner, apply_trial_params, create_objective
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from sqlalchemy import select, text

# Per-fold training logs would run to millions of lines over 100 trials per cell.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "cv_protocol_common_range"
BASE_CONFIG = CONFIGS_DIR / "xgboost_epl_cv_protocol_base.yaml"
CELL_CONFIGS: dict[str, Path] = {
    "expanding": CONFIGS_DIR / "xgboost_epl_cv_protocol_expanding.yaml",
    "sliding_380": CONFIGS_DIR / "xgboost_epl_cv_protocol_sliding_380.yaml",
    "sliding_760": CONFIGS_DIR / "xgboost_epl_cv_protocol_sliding_760.yaml",
}
REFERENCE_CELL = "expanding"
EVALUATION_START = datetime(2024, 8, 1, tzinfo=UTC)
COUNT_START = date(2015, 8, 1)
BASELINE_FEATURE = "tab_retail_sharp_diff"
CV_FIELDS = {"window_type", "min_train_events", "max_train_events", "val_step_events"}


@dataclass(frozen=True)
class CellData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    event_ids: np.ndarray


def season_of(ts: pd.Timestamp) -> str:
    start = ts.year if ts.month >= 8 else ts.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


async def load_rows(config: MLTrainingConfig) -> CellData:
    async with async_session_maker() as session:
        data = await prepare_training_data_from_config(config, session)
    if data.validation_data is not None or data.X_test is not None:
        raise ValueError("expected every row in the training arrays (no val or test split)")
    assert data.event_ids_train is not None
    return CellData(data.X_train, data.y_train, data.feature_names, data.event_ids_train)


async def load_commence_times(until: datetime | None) -> dict[str, datetime]:
    conditions = [
        Event.sport_key == "soccer_epl",
        Event.status == EventStatus.FINAL,
        Event.commence_time >= datetime.combine(COUNT_START, datetime.min.time(), UTC),
    ]
    if until is not None:
        conditions.append(Event.commence_time < until)
    async with async_session_maker() as session:
        result = await session.execute(select(Event.id, Event.commence_time).where(*conditions))
        return {row.id: row.commence_time for row in result}


SNAPSHOT_COVERAGE_SQL = text(
    """
    SELECT e.commence_time, s.raw_data->>'source' AS source, s.fetch_tier,
           s.raw_data::text LIKE '%"pinnacle"%'
               OR s.raw_data::text LIKE '%"betfair_exchange"%' AS has_sharp,
           s.raw_data::text LIKE '%"bet365"%' AS has_bet365
    FROM odds_snapshots s JOIN events e ON e.id = s.event_id
    WHERE e.sport_key = 'soccer_epl' AND e.status = 'FINAL'
      AND e.commence_time >= :start AND e.commence_time < :until
    """
)


async def snapshot_coverage() -> pd.DataFrame:
    """Snapshots per season, source and tier, with how many quote a sharp book and bet365."""
    start = datetime.combine(COUNT_START, datetime.min.time(), UTC)
    async with async_session_maker() as session:
        result = await session.execute(
            SNAPSHOT_COVERAGE_SQL, {"start": start, "until": EVALUATION_START}
        )
        df = pd.DataFrame(result.all(), columns=list(result.keys()))
    df["season"] = pd.to_datetime(df["commence_time"], utc=True).map(season_of)
    return (
        df.groupby(["season", "source", "fetch_tier"])
        .agg(
            snapshots=("has_sharp", "size"),
            with_sharp=("has_sharp", "sum"),
            with_bet365=("has_bet365", "sum"),
        )
        .reset_index()
    )


def check_cells_share_data(base: MLTrainingConfig, cells: dict[str, MLTrainingConfig]) -> None:
    """Every cell must differ from the base only in its walk-forward window."""
    base_data = base.training.data.model_dump(exclude=CV_FIELDS)
    for name, cfg in cells.items():
        if cfg.training.data.model_dump(exclude=CV_FIELDS) != base_data:
            raise ValueError(f"{name}: data section differs from the base beyond the CV window")
        if cfg.training.features != base.training.features:
            raise ValueError(f"{name}: features differ from the base")
        if cfg.training.model != base.training.model or cfg.tuning != base.tuning:
            raise ValueError(f"{name}: model or tuning differs from the base")


def ols_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(x_train), x_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return coef[0] + coef[1] * x_val


def run_cell(name: str, config_path: Path, data: CellData, n_trials: int) -> dict[str, Any]:
    """Tune one cell, then refit its best trial per fold for out-of-fold predictions."""
    config = MLTrainingConfig.from_yaml(config_path)
    assert config.tuning is not None
    tuner = OptunaTuner(
        study_name=f"cv_protocol_common_range_{name}",
        direction=config.tuning.direction,
        sampler=config.tuning.sampler,
        pruner=config.tuning.pruner,
        storage=None,
        tracking_config=None,
    )
    objective = create_objective(
        config=config,
        X_train=data.X,
        y_train=data.y,
        feature_names=data.feature_names,
        event_ids_train=data.event_ids,
    )
    t0 = time.monotonic()
    study = tuner.optimize(objective=objective, n_trials=n_trials)
    tune_seconds = time.monotonic() - t0

    best_config = config.model_copy(deep=True)
    apply_trial_params(best_config, study.best_params)
    wf = best_config.training.data
    assert wf.min_train_events is not None and wf.val_step_events is not None
    splits = make_walk_forward_splits(
        event_ids=data.event_ids,
        min_train_events=wf.min_train_events,
        val_step_events=wf.val_step_events,
        window_type=wf.window_type,
        max_train_events=wf.max_train_events,
    )

    baseline_col = data.feature_names.index(BASELINE_FEATURE)
    if np.isnan(data.X[:, baseline_col]).any():
        raise ValueError(f"{BASELINE_FEATURE} has NaN rows; the OLS baseline needs every row")

    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    fold_mses: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        strategy = XGBoostLineMovementStrategy()
        history = strategy.train_from_config(
            config=best_config,
            X_train=data.X[train_idx],
            y_train=data.y[train_idx],
            feature_names=data.feature_names,
            X_val=data.X[val_idx],
            y_val=data.y[val_idx],
        )
        assert strategy.model is not None
        pred = strategy.model.predict(data.X[val_idx])
        fold_mses.append(history["val_mse"])
        rows.append(
            pd.DataFrame(
                {
                    "cell": name,
                    "fold": fold_idx,
                    "row": val_idx,
                    "event_id": data.event_ids[val_idx],
                    "y": data.y[val_idx],
                    "pred_model": pred,
                    "pred_ols": ols_fit_predict(
                        data.X[train_idx, baseline_col],
                        data.y[train_idx],
                        data.X[val_idx, baseline_col],
                    ),
                    "pred_mean": float(np.mean(data.y[train_idx])),
                }
            )
        )
        train_events = list(dict.fromkeys(data.event_ids[train_idx]))
        val_events = list(dict.fromkeys(data.event_ids[val_idx]))
        folds.append(
            {
                "cell": name,
                "fold": fold_idx,
                "n_train_events": len(train_events),
                "n_val_events": len(val_events),
                "first_train_event": train_events[0],
                "last_train_event": train_events[-1],
                "first_val_event": val_events[0],
                "last_val_event": val_events[-1],
                "val_mse": history["val_mse"],
            }
        )

    # The refit must reproduce the tuned objective, or the predictions are not the best trial's.
    refit_mean = float(np.mean(fold_mses))
    if not np.isclose(refit_mean, study.best_value, rtol=1e-7, atol=0.0):
        raise RuntimeError(f"{name}: refit mean fold MSE {refit_mean} != {study.best_value}")

    trials = study.trials_dataframe()
    trials.insert(0, "cell", name)
    return {
        "cell": name,
        "config": str(config_path.relative_to(EXPERIMENTS_DIR.parent)),
        "window_type": wf.window_type,
        "min_train_events": wf.min_train_events,
        "max_train_events": wf.max_train_events,
        "val_step_events": wf.val_step_events,
        "n_folds": len(splits),
        "best_trial": study.best_trial.number,
        "best_cv_mean_fold_mse": study.best_value,
        "best_params": study.best_params,
        "tune_seconds": round(tune_seconds, 1),
        "oof": pd.concat(rows, ignore_index=True),
        "folds": pd.DataFrame(folds),
        "trials": trials,
    }


def event_sq_errors(oof: pd.DataFrame, pred_col: str, events: list[str]) -> pd.DataFrame:
    """Per-event sum of squared errors and row count, indexed by event."""
    sq = (oof[pred_col] - oof["y"]) ** 2
    grouped = sq.groupby(oof["event_id"]).agg(["sum", "size"])
    return grouped.loc[events]


def bootstrap_diff_ci(
    a: pd.DataFrame, b: pd.DataFrame, n_resamples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Pooled MSE(a) - MSE(b), resampling events, with a 95% percentile CI."""
    sse_a, sse_b, n = a["sum"].to_numpy(), b["sum"].to_numpy(), a["size"].to_numpy()
    point = sse_a.sum() / n.sum() - sse_b.sum() / n.sum()
    idx = rng.integers(0, len(n), size=(n_resamples, len(n)))
    rows = n[idx].sum(axis=1)
    diffs = sse_a[idx].sum(axis=1) / rows - sse_b[idx].sum(axis=1) / rows
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


async def count_events(base: MLTrainingConfig) -> tuple[pd.DataFrame, CellData]:
    """Events in the database and events yielding rows, by season, from COUNT_START."""
    wide = base.model_copy(deep=True)
    wide.training.data.start_date = COUNT_START
    rows = await load_rows(wide)
    commence = await load_commence_times(until=EVALUATION_START)
    db = pd.Series({eid: season_of(pd.Timestamp(ct)) for eid, ct in commence.items()})
    with_rows = pd.Series(list(dict.fromkeys(rows.event_ids))).map(
        lambda eid: season_of(pd.Timestamp(commence[eid]))
    )
    counts = pd.DataFrame(
        {"events_in_db": db.value_counts(), "events_with_rows": with_rows.value_counts()}
    )
    counts = counts.fillna(0).astype(int).sort_index()
    counts.index.name = "season"
    row_seasons = pd.Series(rows.event_ids).map(lambda eid: season_of(pd.Timestamp(commence[eid])))
    hours = pd.Series(rows.X[:, rows.feature_names.index("hours_until_event")])
    counts["min_hours_until_event"] = hours.groupby(row_seasons.to_numpy()).min()
    counts["max_hours_until_event"] = hours.groupby(row_seasons.to_numpy()).max()
    counts["rows"] = row_seasons.value_counts().reindex(counts.index, fill_value=0)
    return counts, rows


async def main(n_trials: int, workers: int, n_resamples: int, seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = MLTrainingConfig.from_yaml(BASE_CONFIG)
    cells = {name: MLTrainingConfig.from_yaml(path) for name, path in CELL_CONFIGS.items()}
    check_cells_share_data(base, cells)

    counts, wide_rows = await count_events(base)
    counts.to_csv(OUTPUT_DIR / "event_counts.csv")
    (await snapshot_coverage()).to_csv(OUTPUT_DIR / "snapshot_coverage.csv", index=False)
    print(counts.to_string())

    data = await load_rows(base)
    # No upper bound here, so an evaluation-range row reaches the check below.
    commence = await load_commence_times(until=None)
    if set(dict.fromkeys(wide_rows.event_ids)) != set(dict.fromkeys(data.event_ids)):
        raise ValueError("base start_date drops events that yield rows from COUNT_START")
    event_times = pd.to_datetime([commence[eid] for eid in data.event_ids], utc=True)
    if (event_times >= EVALUATION_START).any():
        raise ValueError("an evaluation-range event reached the tuning data")
    if not event_times.is_monotonic_increasing:
        raise ValueError("rows are not in chronological order")
    unique_events = list(dict.fromkeys(data.event_ids))
    print(f"\n{len(data.y)} rows, {len(unique_events)} events, features {data.feature_names}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            name: pool.submit(run_cell, name, path, data, n_trials)
            for name, path in CELL_CONFIGS.items()
        }
        results = {name: f.result() for name, f in futures.items()}

    # Common range: events every cell predicts. All starts are at or after the largest
    # min_train_events; the tail is cut where the cell with the shortest coverage ends.
    covered = [set(r["oof"]["event_id"]) for r in results.values()]
    common = [eid for eid in unique_events if all(eid in c for c in covered)]
    first_pos = unique_events.index(common[0])
    if first_pos != max(r["min_train_events"] for r in results.values()):
        raise RuntimeError("common range does not start at the largest min_train_events")
    if unique_events[first_pos : first_pos + len(common)] != common:
        raise RuntimeError("common range is not contiguous")

    rng = np.random.default_rng(seed)
    ref = event_sq_errors(results[REFERENCE_CELL]["oof"], "pred_model", common)
    table: list[dict[str, Any]] = []
    for name, r in results.items():
        oof = r["oof"][r["oof"]["event_id"].isin(common)]
        model = event_sq_errors(oof, "pred_model", common)
        ols = event_sq_errors(oof, "pred_ols", common)
        mean = event_sq_errors(oof, "pred_mean", common)
        n_rows = int(model["size"].sum())
        vs_ref = bootstrap_diff_ci(model, ref, n_resamples, rng)
        vs_ols = bootstrap_diff_ci(model, ols, n_resamples, rng)
        table.append(
            {
                "cell": name,
                "window_type": r["window_type"],
                "min_train_events": r["min_train_events"],
                "max_train_events": r["max_train_events"],
                "n_folds": r["n_folds"],
                "best_cv_mean_fold_mse": r["best_cv_mean_fold_mse"],
                "common_events": len(common),
                "common_rows": n_rows,
                "pooled_mse_model": model["sum"].sum() / n_rows,
                "pooled_mse_ols_baseline": ols["sum"].sum() / n_rows,
                "pooled_mse_mean_floor": mean["sum"].sum() / n_rows,
                "diff_vs_expanding": vs_ref[0],
                "diff_vs_expanding_ci_low": vs_ref[1],
                "diff_vs_expanding_ci_high": vs_ref[2],
                "diff_vs_ols": vs_ols[0],
                "diff_vs_ols_ci_low": vs_ols[1],
                "diff_vs_ols_ci_high": vs_ols[2],
            }
        )
    results_df = pd.DataFrame(table)

    best = results_df.loc[results_df["pooled_mse_model"].idxmin()]
    ci_excludes_zero = bool(
        best["cell"] != REFERENCE_CELL
        and (best["diff_vs_expanding_ci_high"] < 0 or best["diff_vs_expanding_ci_low"] > 0)
    )
    chosen = str(best["cell"]) if ci_excludes_zero else REFERENCE_CELL

    def at(eid: str) -> str:
        return pd.Timestamp(commence[eid]).isoformat()

    folds_df = pd.concat([r["folds"] for r in results.values()], ignore_index=True)
    for col in ("first_train_event", "last_train_event", "first_val_event", "last_val_event"):
        folds_df[col.replace("event", "commence")] = folds_df[col].map(at)

    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    folds_df.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    pd.concat([r["oof"] for r in results.values()], ignore_index=True).to_csv(
        OUTPUT_DIR / "oof_predictions.csv", index=False
    )
    pd.concat([r["trials"] for r in results.values()], ignore_index=True).to_csv(
        OUTPUT_DIR / "trials.csv", index=False
    )

    chosen_result = results[chosen]
    summary = {
        "base_config": str(BASE_CONFIG.relative_to(EXPERIMENTS_DIR.parent)),
        "evaluation_start": EVALUATION_START.isoformat(),
        "n_trials_per_cell": n_trials,
        "bootstrap_resamples": n_resamples,
        "seed": seed,
        "tuning_range": {
            "n_rows": len(data.y),
            "n_events": len(unique_events),
            "first_event": unique_events[0],
            "first_commence": at(unique_events[0]),
            "last_event": unique_events[-1],
            "last_commence": at(unique_events[-1]),
        },
        "common_range": {
            "n_events": len(common),
            "first_event": common[0],
            "first_commence": at(common[0]),
            "last_event": common[-1],
            "last_commence": at(common[-1]),
            "first_position": first_pos,
        },
        "lowest_pooled_mse_cell": str(best["cell"]),
        "lowest_ci_excludes_zero_vs_expanding": ci_excludes_zero,
        "chosen": chosen,
        "chosen_config": chosen_result["config"],
        "chosen_best_params": chosen_result["best_params"],
        "rebaseline_fold_layout": {
            "window_type": chosen_result["window_type"],
            "max_train_events": chosen_result["max_train_events"],
            "val_step_events": chosen_result["val_step_events"],
            "min_train_events": len(unique_events),
            "start_date": str(base.training.data.start_date),
            "first_val_event_after": unique_events[-1],
        },
        "cells": {
            name: {k: v for k, v in r.items() if k not in ("oof", "folds", "trials")}
            for name, r in results.items()
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + results_df.to_string(index=False))
    print(f"\nChosen: {chosen} (lowest pooled MSE: {best['cell']})")
    print(f"Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=len(CELL_CONFIGS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.workers, args.bootstrap, args.seed))
