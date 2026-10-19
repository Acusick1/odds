"""How the retail-EV loss-weight scale changes what a frozen min_child_weight means (#556).

Development data only, so nothing here is a keep-bar result. Rows come from the
fixed-scale price-only config (`retail_ev_price_only.load_data`), which ends before 2026-27.

Two outputs:

- ``drift.csv``: for several training sets, the mean loss weight per event and the
  number of events that the frozen ``min_child_weight`` stands for, under the fixed
  scale (`odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`) and under rescaling to mean one
  over that set's rows.
- ``folds.csv`` and ``summary.json``: the frozen residual seed-42 trial of the as-of-t
  run (`experiments/results/retail_ev_as_of_t_groups/summary.json`, variant
  ``price_only``) refitted on the price-only scoring folds with three weightings: the
  fixed scale, mean one over every development row, and mean one over each fold's own
  training rows. Every weighting is scored with the same loss weights, and the
  weighted metrics do not depend on their scale.

Usage:
    uv run python experiments/scripts/retail_ev_weight_scale.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from odds_analytics.retail_ev import DECISION_WINDOW, LOSS_WEIGHT_SCALE
from odds_analytics.retail_ev_metrics import mse_gain, weighted_mse
from odds_analytics.training.config import MLTrainingConfig, RetailEvArm
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.tuner import apply_trial_params
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from retail_ev_price_only import (
    METRIC_SCALE,
    REPO_DIR,
    RunData,
    _est,
    data_fingerprint,
    git_dirty,
    git_sha,
    load_data,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "retail_ev_weight_scale"
CONFIG_PATH = EXPERIMENTS_DIR / "configs" / "xgboost_epl_retail_ev_price_only_fixed_scale.yaml"
FROZEN_SUMMARY = EXPERIMENTS_DIR / "results" / "retail_ev_as_of_t_groups" / "summary.json"
FROZEN_VARIANT = "price_only"
FROZEN_RUN = "residual_s42"
SEED = 42

Weighting = Callable[[np.ndarray, np.ndarray], np.ndarray]


def frozen_params() -> dict[str, Any]:
    runs = json.loads(FROZEN_SUMMARY.read_text())["variants"][FROZEN_VARIANT]["runs"]
    return next(r["best_params"] for r in runs if r["run"] == FROZEN_RUN)


def weightings(unscaled: np.ndarray) -> dict[str, Weighting]:
    return {
        "fixed": lambda _: LOSS_WEIGHT_SCALE * unscaled,
        "mean_one_all_rows": lambda _: unscaled / unscaled.mean(),
        "mean_one_per_fold": lambda train_idx: unscaled / unscaled[train_idx].mean(),
    }


def event_weight_summary(
    event_ids: np.ndarray, weights: np.ndarray, min_child_weight: float
) -> dict[str, float]:
    _, inverse = np.unique(event_ids, return_inverse=True)
    per_event = np.bincount(inverse, weights=weights)
    return {
        "mean_event_weight": float(per_event.mean()),
        "events_per_min_child_weight": min_child_weight / float(per_event.mean()),
    }


def drift_table(
    data: RunData, splits: list[tuple[np.ndarray, np.ndarray]], min_child_weight: float
) -> pd.DataFrame:
    unscaled = data.loss_weights / LOSS_WEIGHT_SCALE
    seasons = data.rows["season"].to_numpy()
    sets: list[tuple[str, np.ndarray]] = [
        (f"season {s}", np.flatnonzero(seasons == s)) for s in sorted(set(seasons))
    ]
    sets.append(("all development rows", np.arange(len(data.y))))
    sets += [(f"scoring fold {k} training rows", tr) for k, (tr, _) in enumerate(splits)]
    records = []
    for label, idx in sets:
        events = len(set(data.event_ids[idx].tolist()))
        record: dict[str, Any] = {
            "training_set": label,
            "events": events,
            "rows": len(idx),
            "rows_per_event": len(idx) / events,
        }
        for name, w in (
            ("fixed", LOSS_WEIGHT_SCALE * unscaled[idx]),
            ("mean_one", unscaled[idx] / unscaled[idx].mean()),
        ):
            summary = event_weight_summary(data.event_ids[idx], w, min_child_weight)
            record |= {f"{name}_{k}": v for k, v in summary.items()}
        records.append(record)
    return pd.DataFrame(records)


def refit(
    config: MLTrainingConfig,
    data: RunData,
    splits: list[tuple[np.ndarray, np.ndarray]],
    weighting: Weighting,
) -> np.ndarray:
    pred = np.full(len(data.y), np.nan)
    for train_idx, val_idx in splits:
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            config=config,
            X_train=data.X[train_idx],
            y_train=data.y[train_idx],
            feature_names=data.feature_names,
            sample_weight_train=weighting(train_idx)[train_idx],
        )
        pred[val_idx] = strategy.predict(data.X[val_idx])
    return pred


async def main(n_boot: int) -> None:
    sha, dirty = git_sha(), git_dirty()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = MLTrainingConfig.from_yaml(CONFIG_PATH)
    params = frozen_params()
    apply_trial_params(config, params)
    config.training.model.random_state = SEED
    config.training.features.retail_ev_arm = RetailEvArm.RESIDUAL
    data = await load_data(config)

    wf = config.training.data
    assert wf.min_train_events is not None and wf.val_step_events is not None
    splits = make_walk_forward_splits(
        event_ids=data.event_ids,
        min_train_events=wf.min_train_events,
        val_step_events=wf.val_step_events,
        commence_times=data.kickoffs,
        gap=DECISION_WINDOW,
    )
    drift = drift_table(data, splits, params["min_child_weight"])
    drift.to_csv(OUTPUT_DIR / "drift.csv", index=False)

    baseline = data.rows["baseline"].to_numpy()
    scored = np.concatenate([v for _, v in splits])
    y, w, ev = data.y[scored], data.loss_weights[scored], data.event_ids[scored]
    unscaled = data.loss_weights / LOSS_WEIGHT_SCALE
    results: dict[str, Any] = {}
    fold_records = []
    for name, weighting in weightings(unscaled).items():
        pred = refit(config, data, splits, weighting)
        gain = mse_gain(y, pred[scored], baseline[scored], w, ev, n_boot)
        results[name] = _est("mse_gain_e6", gain, METRIC_SCALE)
        for k, (train_idx, val_idx) in enumerate(splits):
            summary = event_weight_summary(
                data.event_ids[train_idx],
                weighting(train_idx)[train_idx],
                params["min_child_weight"],
            )
            yv, wv = data.y[val_idx], data.loss_weights[val_idx]
            fold_records.append(
                {
                    "weighting": name,
                    "fold": k,
                    "n_train_events": len(set(data.event_ids[train_idx].tolist())),
                    "n_val_events": len(set(data.event_ids[val_idx].tolist())),
                    "n_val_rows": len(val_idx),
                    **summary,
                    "wmse_baseline": weighted_mse(yv, baseline[val_idx], wv) * METRIC_SCALE,
                    "wmse_model": weighted_mse(yv, pred[val_idx], wv) * METRIC_SCALE,
                }
            )
    pd.DataFrame(fold_records).to_csv(OUTPUT_DIR / "folds.csv", index=False)

    summary_out = {
        "git_sha": sha,
        "git_dirty": dirty,
        "config": str(CONFIG_PATH.relative_to(REPO_DIR)),
        "data_sha256": data_fingerprint(data),
        "frozen_params": params,
        "frozen_source": f"{FROZEN_SUMMARY.relative_to(REPO_DIR)} {FROZEN_VARIANT} {FROZEN_RUN}",
        "loss_weight_scale": LOSS_WEIGHT_SCALE,
        "n_boot": n_boot,
        "scored_events": len(set(ev.tolist())),
        "scored_rows": len(scored),
        "wmse_baseline_e6": weighted_mse(y, baseline[scored], w) * METRIC_SCALE,
        "residual_weighted_mean": float(np.average(data.y - baseline, weights=data.loss_weights)),
        "residual_weighted_rms": float(
            np.sqrt(np.average((data.y - baseline) ** 2, weights=data.loss_weights))
        ),
        "refits": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")

    pd.set_option("display.width", 250)
    print(drift.round(2).to_string(index=False))
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(main(args.n_boot))
