"""Retail-EV residual models tuned on 2025-26 folds that carry the as-of-t columns (#549).

Development data only, so nothing here is a keep-bar result. The #540 and #545 runs
tune on 2024-25, whose FDUK opening rows have none of the as-of-t columns, and freeze
the settings. Here each scoring fold is tuned afresh by nested walk-forward:

- Outer folds: 17-event blocks over the second half of 2025-26 (events 171 to 340 of
  the season), expanding from 2024-25 plus the first half, with the 7-day gap.
- Inner folds: each outer fold tunes only on its own training rows (every event that
  kicks off at least 7 days before the fold). The inner validation blocks are 17-event
  blocks of 2025-26 aligned to end at the last of those events, each with the 7-day gap,
  so the most recent block is always a tuning fold and 2024-25 is only ever trained on.

Only the residual arm runs, for price-only, `polymarket_at_t` alone and all four groups,
with two seeds. Every outer fold also refits the #545 hyperparameters (tuned on 2024-25
and frozen) on the same rows, so the effect of the tuning scheme is measured on the
same scored rows. The config ends before 2026-27, the candidate fresh season.

Usage:
    uv run python experiments/scripts/retail_ev_nested_tuning.py
    uv run python experiments/scripts/retail_ev_nested_tuning.py --n-trials 2 --n-boot 200
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from odds_analytics.retail_ev import DECISION_WINDOW
from odds_analytics.retail_ev_metrics import mse_gain, pick_ev_gain, weighted_mse
from odds_analytics.training.config import MLTrainingConfig, RetailEvFeatureGroupName
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.tuner import OptunaTuner, apply_trial_params, create_objective
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from retail_ev_as_of_t_groups import variant_config
from retail_ev_price_only import (
    CONFIG_PATH,
    METRIC_SCALE,
    REPO_DIR,
    TUNING_END,
    RunData,
    _est,
    _importances,
    _utc64,
    _wstd,
    data_fingerprint,
    git_dirty,
    git_sha,
    load_data,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "retail_ev_nested_tuning"
FROZEN_SUMMARY = EXPERIMENTS_DIR / "results" / "retail_ev_as_of_t_groups" / "summary.json"
REFERENCE = "price_only"
VARIANTS: dict[str, tuple[RetailEvFeatureGroupName, ...]] = {
    REFERENCE: (),
    "polymarket_at_t": ("polymarket_at_t",),
    "all_groups": ("polymarket_at_t", "dynamics", "schedule_as_of_t", "standings_as_of_t"),
}
SEEDS: tuple[int, ...] = (42, 7)
PRIOR_SEASON_EVENTS = 380
FIRST_HALF_EVENTS = 170
OUTER_STEP_EVENTS = 17
INNER_STEP_EVENTS = 17
FAMILY_ALPHA = 0.05
MAX_WORKERS = 20
CORNER_MIN_CHILD_WEIGHT = 1000
CORNER_MAX_ESTIMATORS = 150
CORNER_MAX_LEARNING_RATE = 0.015


class Tuning(StrEnum):
    NESTED = "nested"
    FROZEN = "frozen"


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    fold: int

    @property
    def label(self) -> str:
        return f"{self.variant}:s{self.seed}"


DATA: dict[str, RunData] = {}
CONFIGS: dict[str, MLTrainingConfig] = {}


def outer_splits(data: RunData) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = make_walk_forward_splits(
        event_ids=data.event_ids,
        min_train_events=PRIOR_SEASON_EVENTS + FIRST_HALF_EVENTS,
        val_step_events=OUTER_STEP_EVENTS,
        commence_times=data.kickoffs,
        gap=DECISION_WINDOW,
    )
    if splits[-1][1][-1] != len(data.y) - 1:
        raise ValueError("the outer folds leave a partial block unscored")
    return splits


def inner_min_train_events(n_tune_events: int) -> int:
    """Initial training events so that the 2025-26 inner blocks end at the last tuning event."""
    n_blocks = (n_tune_events - PRIOR_SEASON_EVENTS) // INNER_STEP_EVENTS
    return n_tune_events - n_blocks * INNER_STEP_EVENTS


def residual_target(data: RunData) -> np.ndarray:
    return (data.y - data.rows["baseline"].to_numpy()).astype(np.float32)


def inner_references(data: RunData, n_rows: int, min_train: int) -> dict[str, Any]:
    """Baseline mean fold weighted MSE on the inner folds, and what their validation rows hold."""
    splits = make_walk_forward_splits(
        event_ids=data.event_ids[:n_rows],
        min_train_events=min_train,
        val_step_events=INNER_STEP_EVENTS,
        commence_times=data.kickoffs[:n_rows],
        gap=DECISION_WINDOW,
    )
    baseline = data.rows["baseline"].to_numpy()
    live = (data.rows["event_type"] == "live_collected").to_numpy()
    val_rows = np.concatenate([v for _, v in splits])
    return {
        "n_inner_folds": len(splits),
        "inner_baseline_mean_fold_wmse": float(
            np.mean([weighted_mse(data.y[v], baseline[v], data.loss_weights[v]) for _, v in splits])
        ),
        "inner_val_events": len(set(data.event_ids[val_rows].tolist())),
        "inner_val_live_events": len(set(data.event_ids[val_rows[live[val_rows]]].tolist())),
        "inner_val_live_rows": int(live[val_rows].sum()),
        "last_inner_val_live_events": len(
            set(data.event_ids[splits[-1][1][live[splits[-1][1]]]].tolist())
        ),
    }


def tune_fold(
    config: MLTrainingConfig, data: RunData, n_rows: int, job: Job, n_trials: int
) -> tuple[optuna.Study, int, float]:
    tune_config = config.model_copy(deep=True)
    min_train = inner_min_train_events(len(set(data.event_ids[:n_rows].tolist())))
    tune_config.training.data.min_train_events = min_train
    tune_config.training.data.val_step_events = INNER_STEP_EVENTS
    tune_config.training.model.random_state = job.seed
    assert tune_config.tuning is not None
    tuner = OptunaTuner(
        study_name=f"retail_ev_nested_{job.variant}_s{job.seed}_f{job.fold}",
        direction=tune_config.tuning.direction,
        sampler=tune_config.tuning.sampler,
        pruner=tune_config.tuning.pruner,
        storage=None,
        tracking_config=None,
        sampler_seed=job.seed,
    )
    objective = create_objective(
        config=tune_config,
        X_train=data.X[:n_rows],
        y_train=residual_target(data)[:n_rows],
        feature_names=data.feature_names,
        event_ids_train=data.event_ids[:n_rows],
        sample_weights_train=data.loss_weights[:n_rows],
        commence_times_train=data.kickoffs[:n_rows],
    )
    t0 = time.monotonic()
    study = tuner.optimize(objective=objective, n_trials=n_trials)
    return study, min_train, time.monotonic() - t0


def fit_predict(
    config: MLTrainingConfig,
    params: dict[str, Any],
    seed: int,
    data: RunData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    fitted = config.model_copy(deep=True)
    apply_trial_params(fitted, params)
    fitted.training.model.random_state = seed
    strategy = XGBoostLineMovementStrategy()
    strategy.train_from_config(
        config=fitted,
        X_train=data.X[train_idx],
        y_train=residual_target(data)[train_idx],
        feature_names=data.feature_names,
        sample_weight_train=data.loss_weights[train_idx],
    )
    assert strategy.model is not None
    pred = strategy.model.predict(data.X[val_idx]).astype(np.float64)
    pred += data.rows["baseline"].to_numpy()[val_idx]
    return pred, _importances(strategy, data.feature_names)


def run_job(job: Job, n_trials: int, frozen_params: dict[str, Any]) -> dict[str, Any]:
    data, config = DATA[job.variant], CONFIGS[job.variant]
    train_idx, val_idx = outer_splits(data)[job.fold]
    n_rows = len(train_idx)
    if not np.array_equal(train_idx, np.arange(n_rows)):
        raise RuntimeError("an outer fold's training rows are not a prefix")
    study, min_train, seconds = tune_fold(config, data, n_rows, job, n_trials)
    nested_pred, nested_imp = fit_predict(
        config, study.best_params, job.seed, data, train_idx, val_idx
    )
    frozen_pred, frozen_imp = fit_predict(config, frozen_params, job.seed, data, train_idx, val_idx)

    y_val, w_val = data.y[val_idx], data.loss_weights[val_idx]
    b_val = data.rows["baseline"].to_numpy()[val_idx]
    tuning_live_rows = int((data.rows["event_type"].to_numpy()[:n_rows] == "live_collected").sum())
    inner = inner_references(data, n_rows, min_train)
    inner_baseline = inner["inner_baseline_mean_fold_wmse"]
    fold = {
        "variant": job.variant,
        "seed": job.seed,
        "fold": job.fold,
        "n_train_events": len(set(data.event_ids[train_idx].tolist())),
        "n_train_rows": n_rows,
        "train_live_rows": tuning_live_rows,
        "n_val_events": len(set(data.event_ids[val_idx].tolist())),
        "n_val_rows": len(val_idx),
        "first_val_kickoff": str(data.kickoffs[val_idx[0]]),
        "last_val_kickoff": str(data.kickoffs[val_idx[-1]]),
        "inner_min_train_events": min_train,
        **inner,
        "best_trial": study.best_trial.number,
        "inner_best_mean_fold_wmse": study.best_value,
        "trials_beating_baseline": sum(
            t.value is not None and t.value < inner_baseline for t in study.trials
        ),
        "tune_seconds": round(seconds, 1),
        **{f"param_{k}": v for k, v in study.best_params.items()},
        "wmse_baseline": weighted_mse(y_val, b_val, w_val) * METRIC_SCALE,
        "wmse_nested": weighted_mse(y_val, nested_pred, w_val) * METRIC_SCALE,
        "wmse_frozen": weighted_mse(y_val, frozen_pred, w_val) * METRIC_SCALE,
        "wstd_nested_minus_baseline": _wstd(nested_pred - b_val, w_val),
        "wstd_frozen_minus_baseline": _wstd(frozen_pred - b_val, w_val),
    }
    trials = study.trials_dataframe()
    trials.insert(0, "fold", job.fold)
    trials.insert(0, "seed", job.seed)
    trials.insert(0, "variant", job.variant)
    importance = pd.DataFrame(
        [
            {"tuning": tuning.value, "feature": name, "gain_share": share}
            for tuning, imp in ((Tuning.NESTED, nested_imp), (Tuning.FROZEN, frozen_imp))
            for name, share in imp.items()
        ]
    ).assign(variant=job.variant, seed=job.seed, fold=job.fold)
    return {
        "job": job,
        "val_idx": val_idx,
        "nested": nested_pred,
        "frozen": frozen_pred,
        "fold": fold,
        "trials": trials,
        "importance": importance,
    }


def column(variant: str, seed: int, tuning: Tuning) -> str:
    return f"{variant}:s{seed}:{tuning.value}"


@dataclass(frozen=True)
class Comparison:
    """Reference minus challenger in weighted MSE; both are prediction columns of the scored rows."""

    challenger: str
    reference: str

    @property
    def name(self) -> str:
        return f"{self.challenger}_vs_{self.reference}"


def comparisons() -> list[Comparison]:
    out: list[Comparison] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            nested = column(variant, seed, Tuning.NESTED)
            out.append(Comparison(nested, "baseline"))
            out.append(Comparison(nested, column(variant, seed, Tuning.FROZEN)))
            if variant != REFERENCE:
                out.append(Comparison(nested, column(REFERENCE, seed, Tuning.NESTED)))
    return out


def event_type_slices(oof: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    return [("all", "all", oof)] + [
        ("event_type", value, oof[oof["event_type"] == value])
        for value in sorted(oof["event_type"].unique())
    ]


def results_table(oof: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    slices = event_type_slices(oof) + [
        ("fold", str(int(fold)), oof[oof["fold"] == fold]) for fold in sorted(oof["fold"].unique())
    ]
    records: list[dict[str, Any]] = []
    for key, value, sub in slices:
        y, w = sub["y"].to_numpy(), sub["w_loss"].to_numpy()
        we, ev = sub["w_event"].to_numpy(), sub["event_id"].to_numpy()
        record: dict[str, Any] = {
            "slice": key,
            "value": value,
            "rows": len(sub),
            "events": sub["event_id"].nunique(),
            "live_events": sub.loc[sub["event_type"] == "live_collected", "event_id"].nunique(),
            "wmse_baseline": weighted_mse(y, sub["baseline"].to_numpy(), w) * METRIC_SCALE,
        }
        for variant in VARIANTS:
            for seed in SEEDS:
                for tuning in Tuning:
                    col = column(variant, seed, tuning)
                    record[f"wmse_{col}"] = weighted_mse(y, sub[col].to_numpy(), w) * METRIC_SCALE
        for comp in comparisons():
            pred, ref = sub[comp.challenger].to_numpy(), sub[comp.reference].to_numpy()
            record |= _est(
                f"mse_gain_{comp.name}", mse_gain(y, pred, ref, w, ev, n_boot), METRIC_SCALE
            )
            record |= _est(
                f"pick_gain_{comp.name}", pick_ev_gain(y, pred, ref, we, ev, n_boot), 1.0
            )
        records.append(record)
    return pd.DataFrame(records)


def bonferroni_table(oof: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Weighted-MSE gains with event-clustered CIs at level 1 - alpha/(every test in the table)."""
    slices = event_type_slices(oof)
    comps = comparisons()
    n_tests = len(slices) * len(comps)
    tail = FAMILY_ALPHA / n_tests / 2 * 100
    records: list[dict[str, Any]] = []
    for key, value, sub in slices:
        y, w = sub["y"].to_numpy(), sub["w_loss"].to_numpy()
        _, inverse = np.unique(sub["event_id"].to_numpy(), return_inverse=True)
        n_events = int(inverse.max()) + 1
        counts = np.random.default_rng(0).multinomial(
            n_events, np.full(n_events, 1.0 / n_events), size=n_boot
        )
        w_event = np.bincount(inverse, weights=w)
        for comp in comps:
            se_ref = np.bincount(inverse, weights=w * (y - sub[comp.reference].to_numpy()) ** 2)
            se_new = np.bincount(inverse, weights=w * (y - sub[comp.challenger].to_numpy()) ** 2)
            samples = (counts @ se_ref - counts @ se_new) / (counts @ w_event)
            low, high = np.percentile(samples, [tail, 100 - tail])
            records.append(
                {
                    "slice": key,
                    "value": value,
                    "challenger": comp.challenger,
                    "reference": comp.reference,
                    "n_tests": n_tests,
                    "level": 1 - FAMILY_ALPHA / n_tests,
                    "gain": (se_ref.sum() - se_new.sum()) / w_event.sum() * METRIC_SCALE,
                    "ci_low": low * METRIC_SCALE,
                    "ci_high": high * METRIC_SCALE,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(records)


def corner_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """How many outer folds' tuned settings sit in #545's heavy-regularisation corner."""
    corner = (
        (folds["param_min_child_weight"] >= CORNER_MIN_CHILD_WEIGHT)
        & (folds["param_n_estimators"] <= CORNER_MAX_ESTIMATORS)
        & (folds["param_learning_rate"] <= CORNER_MAX_LEARNING_RATE)
    )
    return (
        folds.assign(in_corner=corner)
        .groupby(["variant", "seed"], sort=False)
        .agg(
            folds=("fold", "size"),
            in_corner=("in_corner", "sum"),
            mcw_median=("param_min_child_weight", "median"),
            mcw_min=("param_min_child_weight", "min"),
            n_estimators_median=("param_n_estimators", "median"),
            learning_rate_median=("param_learning_rate", "median"),
            max_depth_median=("param_max_depth", "median"),
            folds_with_trial_beating_inner_baseline=(
                "trials_beating_baseline",
                lambda s: int((s > 0).sum()),
            ),
            wstd_nested_minus_baseline_median=("wstd_nested_minus_baseline", "median"),
            wstd_frozen_minus_baseline_median=("wstd_frozen_minus_baseline", "median"),
        )
        .reset_index()
    )


def frozen_params() -> dict[tuple[str, int], dict[str, Any]]:
    summary = json.loads(FROZEN_SUMMARY.read_text())
    return {
        (variant, seed): next(
            r["best_params"]
            for r in summary["variants"][variant]["runs"]
            if r["run"] == f"residual_s{seed}"
        )
        for variant in VARIANTS
        for seed in SEEDS
    }


async def main(n_trials: int, n_boot: int, n_boot_bonferroni: int) -> None:
    sha, dirty = git_sha(), git_dirty()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, groups in VARIANTS.items():
        CONFIGS[name] = variant_config(groups)
        DATA[name] = await load_data(CONFIGS[name])
        print(f"{name}: {len(DATA[name].y):,} rows, {len(DATA[name].feature_names)} features")
    reference = DATA[REFERENCE]
    for name, d in DATA.items():
        if not np.array_equal(d.event_ids, reference.event_ids) or not np.array_equal(
            d.y, reference.y
        ):
            raise ValueError(f"{name} rows differ from the price-only rows")
    prior = reference.kickoffs < _utc64(TUNING_END)
    if len(set(reference.event_ids[prior].tolist())) != PRIOR_SEASON_EVENTS:
        raise ValueError("the 2024-25 event count differs from PRIOR_SEASON_EVENTS")
    splits = outer_splits(reference)
    frozen = frozen_params()

    jobs = sorted(
        (Job(v, s, f) for v in VARIANTS for s in SEEDS for f in range(len(splits))),
        key=lambda j: -j.fold,
    )
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        futures = [pool.submit(run_job, j, n_trials, frozen[(j.variant, j.seed)]) for j in jobs]
        results = [f.result() for f in futures]

    oof = reference.rows.copy()
    fold_of_row = np.full(len(oof), -1)
    for fold_idx, (_, val_idx) in enumerate(splits):
        fold_of_row[val_idx] = fold_idx
    oof["fold"] = fold_of_row
    for r in results:
        job: Job = r["job"]
        for tuning in Tuning:
            oof.loc[r["val_idx"], column(job.variant, job.seed, tuning)] = r[tuning.value]
    oof = oof[oof["fold"] >= 0].reset_index(drop=True)

    table = results_table(oof, n_boot)
    table.to_csv(OUTPUT_DIR / "results.csv", index=False)
    bonferroni = bonferroni_table(oof, n_boot_bonferroni)
    bonferroni.to_csv(OUTPUT_DIR / "bonferroni.csv", index=False)
    folds = pd.DataFrame([r["fold"] for r in results]).sort_values(["variant", "seed", "fold"])
    folds.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    corner = corner_summary(folds)
    corner.to_csv(OUTPUT_DIR / "corner.csv", index=False)
    pd.concat([r["trials"] for r in results]).to_csv(OUTPUT_DIR / "trials.csv", index=False)
    importance = pd.concat([r["importance"] for r in results], ignore_index=True)
    (
        importance.groupby(["variant", "seed", "tuning", "feature"], sort=False)["gain_share"]
        .mean()
        .reset_index()
        .to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    )

    summary = {
        "git_sha": sha,
        "git_dirty": dirty,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "config": str(CONFIG_PATH.relative_to(REPO_DIR)),
        "frozen_params_from": str(FROZEN_SUMMARY.relative_to(REPO_DIR)),
        "data_sha256": data_fingerprint(reference),
        "n_trials": n_trials,
        "n_boot": n_boot,
        "n_boot_bonferroni": n_boot_bonferroni,
        "rows": len(reference.y),
        "last_kickoff": str(reference.kickoffs[-1]),
        "outer": {
            "min_train_events": PRIOR_SEASON_EVENTS + FIRST_HALF_EVENTS,
            "step_events": OUTER_STEP_EVENTS,
            "folds": len(splits),
            "scored_rows": len(oof),
            "scored_events": int(oof["event_id"].nunique()),
            "scored_live_events": int(
                oof.loc[oof["event_type"] == "live_collected", "event_id"].nunique()
            ),
        },
        "inner_step_events": INNER_STEP_EVENTS,
        "frozen_params": {f"{v}:s{s}": p for (v, s), p in frozen.items()},
        "variants": {name: list(groups) for name, groups in VARIANTS.items()},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(corner.round(4).to_string(index=False))
    print(bonferroni.round(1).to_string(index=False))
    gain_cols = [c for c in table.columns if c.startswith("mse_gain_") and "_ci_" not in c]
    print(table[["slice", "value", "events", "live_events", *gain_cols]].round(1).T.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-boot-bonferroni", type=int, default=20000)
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.n_boot, args.n_boot_bonferroni))
