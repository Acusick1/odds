"""Price-only XGBoost on the ADR-0012 retail-EV target against its baselines (#540).

Development data only, so nothing here is a keep-bar result. Rows come from the
retail_ev pipeline (`prepare_training_data_from_config`). Hyperparameters are tuned
with Optuna on 2024-25 events only, on expanding walk-forward folds inside that
season, then frozen. The scoring folds walk forward from the first 2025-26 event in
50-event blocks with the 7-day gap, so no tuned trial has seen a scored row.

Each scored row carries the martingale baseline (the exchange price at t as the
close), a constant shrinkage of it (baseline × β, β fitted by weighted least squares
on the fold's training rows) and four tuned models: two arms, each with two seeds.
The direct arm fits the EV target; the residual arm fits the target minus the
baseline and adds the baseline back, so a model that learns nothing is the baseline.
The script writes per-slice keep-bar metrics with event-clustered bootstrap CIs, fold
diagnostics, per-event contributions, feature importances and the trials.

Usage:
    uv run python experiments/scripts/retail_ev_price_only.py
    uv run python experiments/scripts/retail_ev_price_only.py --n-trials 3 --n-boot 200
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import structlog
from odds_analytics.feature_groups import filter_completed_events
from odds_analytics.retail_ev import DECISION_WINDOW, load_retail_ev_dataset
from odds_analytics.retail_ev_metrics import (
    BootstrapEstimate,
    mse_gain,
    pick_ev_gain,
    weighted_mse,
)
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.data_preparation import (
    filter_events_by_date_range,
    prepare_training_data_from_config,
)
from odds_analytics.training.tuner import OptunaTuner, apply_trial_params, create_objective
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from retail_ev_target_report import HORIZON_BUCKETS, horizon_bucket, season_of

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EXPERIMENTS_DIR.parent
CONFIG_PATH = EXPERIMENTS_DIR / "configs" / "xgboost_epl_retail_ev_price_only.yaml"
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "retail_ev_price_only"
OPTUNA_DIR = EXPERIMENTS_DIR / "optuna"
TUNING_END = datetime(2025, 8, 1, tzinfo=UTC)
TUNING_MIN_TRAIN_EVENTS = 130
SEEDS: tuple[int, ...] = (42, 7)
METRIC_SCALE = 1e6
TOP_EVENTS = 10


class Arm(StrEnum):
    DIRECT = "direct"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class ModelRun:
    arm: Arm
    seed: int

    @property
    def label(self) -> str:
        return f"{self.arm.value}_s{self.seed}"


MODEL_RUNS: tuple[ModelRun, ...] = tuple(ModelRun(arm, seed) for arm in Arm for seed in SEEDS)
MODEL_LABELS: tuple[str, ...] = tuple(run.label for run in MODEL_RUNS)
PREDICTIONS: tuple[str, ...] = ("baseline", "shrinkage", *MODEL_LABELS)
CHALLENGERS: tuple[str, ...] = PREDICTIONS[1:]


@dataclass(frozen=True)
class RunData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    event_ids: np.ndarray
    loss_weights: np.ndarray
    kickoffs: np.ndarray
    rows: pd.DataFrame


def _utc64(ts: datetime) -> np.datetime64:
    return np.datetime64(ts.astimezone(UTC).replace(tzinfo=None), "us")


async def load_data(config: MLTrainingConfig) -> RunData:
    """Training arrays from the pipeline, joined to each row's metadata from the same rows."""
    data_cfg = config.training.data
    async with async_session_maker() as session:
        prepared = await prepare_training_data_from_config(config, session)
        events = await filter_events_by_date_range(
            session,
            start_date=datetime.combine(data_cfg.start_date, datetime.min.time(), tzinfo=UTC),
            end_date=datetime.combine(data_cfg.end_date, datetime.max.time(), tzinfo=UTC),
            status=EventStatus.FINAL,
            sport_key=data_cfg.sport_key,
        )
        dataset = await load_retail_ev_dataset(
            filter_completed_events(events), session, config.training.features.primary_market
        )
    if prepared.X_test is not None or prepared.validation_data is not None:
        raise ValueError("expected every row in the training arrays (no val or test split)")
    assert prepared.event_ids_train is not None
    assert prepared.sample_weights_train is not None
    assert prepared.commence_times_train is not None
    if not np.array_equal(prepared.event_ids_train, dataset.event_ids):
        raise ValueError("pipeline rows and dataset rows differ in event order")
    if not np.allclose(prepared.y_train, dataset.targets, atol=1e-6):
        raise ValueError("pipeline targets differ from the dataset's")

    rows = pd.DataFrame(
        {
            "event_id": dataset.event_ids,
            "commence_time": pd.to_datetime(dataset.commence_times),
            "season": [season_of(r) for r in dataset.rows],
            "horizon": [horizon_bucket(r) for r in dataset.rows],
            "retail_source": [r.retail_kind.value for r in dataset.rows],
            "close_source": [r.close.kind.value for r in dataset.rows],
            "outcome": [r.outcome.value for r in dataset.rows],
            "book": [r.book for r in dataset.rows],
            "y": dataset.targets,
            "baseline": dataset.baselines,
            "w_event": dataset.event_weights,
            "w_loss": prepared.sample_weights_train,
        }
    )
    live_events = set(rows.loc[rows["retail_source"] != "fduk_opening", "event_id"])
    rows["event_type"] = np.where(rows["event_id"].isin(live_events), "live_collected", "fduk_only")
    kickoffs = prepared.commence_times_train
    if not (np.diff(kickoffs.astype("int64")) >= 0).all():
        raise ValueError("rows are not in kick-off order")
    return RunData(
        X=prepared.X_train,
        y=prepared.y_train,
        feature_names=prepared.feature_names,
        event_ids=prepared.event_ids_train,
        loss_weights=prepared.sample_weights_train,
        kickoffs=kickoffs,
        rows=rows,
    )


def tuning_prefix(data: RunData, config: MLTrainingConfig) -> int:
    """Rows before TUNING_END; they must be a prefix holding exactly min_train_events events."""
    in_tuning = data.kickoffs < _utc64(TUNING_END)
    n_rows = int(in_tuning.sum())
    if not in_tuning[:n_rows].all():
        raise ValueError("tuning-range rows are not a prefix of the data")
    n_events = len(set(data.event_ids[:n_rows].tolist()))
    if n_events != config.training.data.min_train_events:
        raise ValueError(
            f"{n_events} tuning-range events but min_train_events is "
            f"{config.training.data.min_train_events}; the first scored fold would not "
            f"start at {TUNING_END.date()}"
        )
    return n_rows


def tuning_fold_references(data: RunData, n_rows: int, val_step: int) -> dict[str, float]:
    """Mean fold weighted MSE of the baseline and the shrinkage on the tuner's folds."""
    splits = make_walk_forward_splits(
        event_ids=data.event_ids[:n_rows],
        min_train_events=TUNING_MIN_TRAIN_EVENTS,
        val_step_events=val_step,
        commence_times=data.kickoffs[:n_rows],
        gap=DECISION_WINDOW,
    )
    baseline = data.rows["baseline"].to_numpy()
    base_mse: list[float] = []
    shrink_mse: list[float] = []
    for train_idx, val_idx in splits:
        w_tr, b_tr = data.loss_weights[train_idx], baseline[train_idx]
        beta = np.sum(w_tr * data.y[train_idx] * b_tr) / np.sum(w_tr * b_tr**2)
        y_val, w_val, b_val = data.y[val_idx], data.loss_weights[val_idx], baseline[val_idx]
        base_mse.append(weighted_mse(y_val, b_val, w_val))
        shrink_mse.append(weighted_mse(y_val, beta * b_val, w_val))
    return {
        "n_folds": len(splits),
        "baseline_mean_fold_wmse": float(np.mean(base_mse)),
        "shrinkage_mean_fold_wmse": float(np.mean(shrink_mse)),
    }


def fit_target(data: RunData, arm: Arm) -> np.ndarray:
    if arm is Arm.DIRECT:
        return data.y
    return (data.y - data.rows["baseline"].to_numpy()).astype(np.float32)


def tune(
    config: MLTrainingConfig, data: RunData, n_rows: int, n_trials: int, run: ModelRun
) -> tuple[optuna.Study, float]:
    tune_config = config.model_copy(deep=True)
    tune_config.training.data.min_train_events = TUNING_MIN_TRAIN_EVENTS
    tune_config.training.model.random_state = run.seed
    assert tune_config.tuning is not None
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_DIR / f'retail_ev_price_only_{run.label}.db'}"
    study_name = f"retail_ev_price_only_{run.label}"
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
    tuner = OptunaTuner(
        study_name=study_name,
        direction=tune_config.tuning.direction,
        sampler=tune_config.tuning.sampler,
        pruner=tune_config.tuning.pruner,
        storage=storage,
        tracking_config=None,
        sampler_seed=run.seed,
    )
    objective = create_objective(
        config=tune_config,
        X_train=data.X[:n_rows],
        y_train=fit_target(data, run.arm)[:n_rows],
        feature_names=data.feature_names,
        event_ids_train=data.event_ids[:n_rows],
        sample_weights_train=data.loss_weights[:n_rows],
        commence_times_train=data.kickoffs[:n_rows],
    )
    t0 = time.monotonic()
    study = tuner.optimize(objective=objective, n_trials=n_trials)
    return study, time.monotonic() - t0


def _wstd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _importances(strategy: XGBoostLineMovementStrategy, names: list[str]) -> dict[str, float]:
    assert strategy.model is not None
    raw = strategy.model.get_booster().get_score(importance_type="total_gain")
    by_name = {
        names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in raw.items()
    }
    total = sum(by_name.values())
    return {n: by_name.get(n, 0.0) / total if total else 0.0 for n in names}


def score_run(
    config: MLTrainingConfig, data: RunData, n_rows: int, n_trials: int, run: ModelRun
) -> dict[str, Any]:
    """Tune on the tuning range, then refit the best trial on every scoring fold."""
    study, tune_seconds = tune(config, data, n_rows, n_trials, run)
    best = config.model_copy(deep=True)
    apply_trial_params(best, study.best_params)
    best.training.model.random_state = run.seed
    y_fit = fit_target(data, run.arm)

    wf = best.training.data
    assert wf.min_train_events is not None and wf.val_step_events is not None
    splits = make_walk_forward_splits(
        event_ids=data.event_ids,
        min_train_events=wf.min_train_events,
        val_step_events=wf.val_step_events,
        window_type=wf.window_type,
        max_train_events=wf.max_train_events,
        commence_times=data.kickoffs,
        gap=DECISION_WINDOW,
    )
    if splits[0][1][0] != n_rows:
        raise RuntimeError("the first scoring fold does not start at the first 2025-26 row")

    baseline = data.rows["baseline"].to_numpy()
    preds: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            config=best,
            X_train=data.X[train_idx],
            y_train=y_fit[train_idx],
            feature_names=data.feature_names,
            sample_weight_train=data.loss_weights[train_idx],
        )
        assert strategy.model is not None
        pred = strategy.model.predict(data.X[val_idx]).astype(np.float64)
        if run.arm is Arm.RESIDUAL:
            pred += baseline[val_idx]
        w_tr = data.loss_weights[train_idx]
        b_tr = baseline[train_idx]
        beta = float(np.sum(w_tr * data.y[train_idx] * b_tr) / np.sum(w_tr * b_tr**2))
        preds.append(
            pd.DataFrame(
                {
                    "row": val_idx,
                    "fold": fold_idx,
                    "model": pred,
                    "shrinkage": beta * baseline[val_idx],
                }
            )
        )
        importances.append(_importances(strategy, data.feature_names))

        y_val, w_val, b_val = data.y[val_idx], data.loss_weights[val_idx], baseline[val_idx]
        train_events = list(dict.fromkeys(data.event_ids[train_idx]))
        val_events = list(dict.fromkeys(data.event_ids[val_idx]))
        folds.append(
            {
                "run": run.label,
                "fold": fold_idx,
                "n_train_events": len(train_events),
                "n_train_rows": len(train_idx),
                "train_events_dropped_by_gap": len(set(data.event_ids[: val_idx[0]].tolist()))
                - len(train_events),
                "n_val_events": len(val_events),
                "n_val_rows": len(val_idx),
                "first_val_kickoff": str(data.kickoffs[val_idx[0]]),
                "last_val_kickoff": str(data.kickoffs[val_idx[-1]]),
                "beta": beta,
                "wmse_baseline": weighted_mse(y_val, b_val, w_val) * METRIC_SCALE,
                "wmse_shrinkage": weighted_mse(y_val, beta * b_val, w_val) * METRIC_SCALE,
                "wmse_model": weighted_mse(y_val, pred, w_val) * METRIC_SCALE,
                "wstd_target": _wstd(y_val, w_val),
                "wstd_baseline": _wstd(b_val, w_val),
                "wstd_model": _wstd(pred, w_val),
                "wstd_model_minus_baseline": _wstd(pred - b_val, w_val),
                "corr_model_baseline": float(np.corrcoef(pred, b_val)[0, 1]),
            }
        )

    trials = study.trials_dataframe()
    trials.insert(0, "run", run.label)
    imp = pd.DataFrame(importances, columns=data.feature_names)
    return {
        "run": run.label,
        "best_trial": study.best_trial.number,
        "best_tuning_cv_wmse": study.best_value,
        "best_params": study.best_params,
        "tune_seconds": round(tune_seconds, 1),
        "preds": pd.concat(preds, ignore_index=True),
        "folds": pd.DataFrame(folds),
        "trials": trials,
        "importance": pd.DataFrame(
            {
                "run": run.label,
                "feature": data.feature_names,
                "mean_gain_share": imp.mean().to_numpy(),
                "min_gain_share": imp.min().to_numpy(),
                "max_gain_share": imp.max().to_numpy(),
            }
        ),
    }


def _est(prefix: str, e: BootstrapEstimate, scale: float) -> dict[str, float]:
    return {
        prefix: e.point * scale,
        f"{prefix}_ci_low": e.ci_low * scale,
        f"{prefix}_ci_high": e.ci_high * scale,
    }


def slice_metrics(oof: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    y = oof["y"].to_numpy()
    w = oof["w_loss"].to_numpy()
    we = oof["w_event"].to_numpy()
    ev = oof["event_id"].to_numpy()
    out: dict[str, Any] = {
        "rows": len(oof),
        "events": oof["event_id"].nunique(),
        "sum_w_event": float(we.sum()),
    }
    for p in PREDICTIONS:
        out[f"wmse_{p}"] = weighted_mse(y, oof[p].to_numpy(), w) * METRIC_SCALE
    for p in CHALLENGERS:
        pred = oof[p].to_numpy()
        base = oof["baseline"].to_numpy()
        out |= _est(f"mse_gain_{p}", mse_gain(y, pred, base, w, ev, n_boot), METRIC_SCALE)
        out |= _est(f"pick_gain_{p}", pick_ev_gain(y, pred, base, we, ev, n_boot), 1.0)
    for p in MODEL_LABELS:
        shrink = oof["shrinkage"].to_numpy()
        out |= _est(
            f"mse_gain_{p}_vs_shrinkage",
            mse_gain(y, oof[p].to_numpy(), shrink, w, ev, n_boot),
            METRIC_SCALE,
        )
    for p in PREDICTIONS:
        picks = oof[p].to_numpy() > 0
        out[f"picks_{p}"] = int(picks.sum())
        out[f"pick_total_ev_{p}"] = float(np.sum(we * y * picks))
    return out


def slice_table(oof: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    bucket_order = [label for _, _, label in HORIZON_BUCKETS]
    keys: list[tuple[str, Callable[[list[str]], list[str]]]] = [
        ("season", sorted),
        ("horizon", lambda vals: [b for b in bucket_order if b in vals]),
        ("retail_source", sorted),
        ("close_source", sorted),
        ("event_type", sorted),
        ("outcome", sorted),
    ]
    records = [{"slice": "all", "value": "all", **slice_metrics(oof, n_boot)}]
    for key, order in keys:
        for value in order(oof[key].unique().tolist()):
            sub = oof[oof[key] == value]
            records.append({"slice": key, "value": value, **slice_metrics(sub, n_boot)})
    return pd.DataFrame(records)


def event_contributions(oof: pd.DataFrame) -> pd.DataFrame:
    """Each event's share of the pooled weighted-MSE gain and pick-EV gain over the baseline."""
    total_w = oof["w_loss"].sum()
    se = {p: oof["w_loss"] * (oof["y"] - oof[p]) ** 2 for p in PREDICTIONS}
    frame = pd.DataFrame(
        {"event_id": oof["event_id"], "event_type": oof["event_type"], "season": oof["season"]}
    )
    for p in CHALLENGERS:
        frame[f"mse_gain_{p}"] = (se["baseline"] - se[p]) / total_w * METRIC_SCALE
    for p in MODEL_LABELS:
        picks = (oof[p] > 0).astype(float) - (oof["baseline"] > 0)
        frame[f"pick_gain_{p}"] = oof["w_event"] * oof["y"] * picks
    frame["w_loss"] = oof["w_loss"]
    grouped = frame.groupby(["event_id", "event_type", "season"], sort=False).sum()
    grouped["rows"] = (
        oof.groupby("event_id", sort=False)
        .size()
        .reindex(grouped.index.get_level_values("event_id"))
        .to_numpy()
    )
    return grouped.reset_index().sort_values(f"mse_gain_{MODEL_LABELS[0]}", ascending=False)


def concentration(contrib: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in [f"{kind}_{p}" for kind in ("mse_gain", "pick_gain") for p in MODEL_LABELS]:
        ranked = contrib.sort_values(col, ascending=False)
        total = float(ranked[col].sum())
        top = ranked.head(TOP_EVENTS)
        out[col] = {
            "total": total,
            f"top_{TOP_EVENTS}_sum": float(top[col].sum()),
            f"total_without_top_{TOP_EVENTS}": total - float(top[col].sum()),
            f"top_{TOP_EVENTS}_live_collected": int((top["event_type"] == "live_collected").sum()),
            "positive_events": int((ranked[col] > 0).sum()),
            "negative_events": int((ranked[col] < 0).sum()),
        }
    return out


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()


def _fmt(table: pd.DataFrame, cols: list[str], digits: int = 1) -> str:
    return table[["slice", "value", *cols]].round(digits).to_string(index=False)


async def main(n_trials: int, n_boot: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = MLTrainingConfig.from_yaml(CONFIG_PATH)
    data = await load_data(config)
    n_rows = tuning_prefix(data, config)
    print(
        f"{len(data.y):,} rows, {data.rows['event_id'].nunique()} events, "
        f"{len(data.feature_names)} features; tuning range {n_rows:,} rows"
    )

    with ProcessPoolExecutor(max_workers=len(MODEL_RUNS)) as pool:
        futures = [pool.submit(score_run, config, data, n_rows, n_trials, r) for r in MODEL_RUNS]
        results = [f.result() for f in futures]

    oof = data.rows.copy()
    scored_rows = results[0]["preds"]["row"].to_numpy()
    for label, r in zip(MODEL_LABELS, results, strict=True):
        p = r["preds"]
        if not np.array_equal(p["row"].to_numpy(), scored_rows):
            raise RuntimeError("model runs scored different rows")
        oof.loc[p["row"], label] = p["model"].to_numpy()
    oof.loc[scored_rows, "shrinkage"] = results[0]["preds"]["shrinkage"].to_numpy()
    oof.loc[scored_rows, "fold"] = results[0]["preds"]["fold"].to_numpy()
    unscored = oof.drop(index=scored_rows)
    oof = oof.loc[scored_rows].reset_index(drop=True)

    table = slice_table(oof, n_boot)
    table.to_csv(OUTPUT_DIR / "results.csv", index=False)
    contrib = event_contributions(oof)
    contrib.to_csv(OUTPUT_DIR / "event_contributions.csv", index=False)
    pd.concat([r["folds"] for r in results]).to_csv(OUTPUT_DIR / "folds.csv", index=False)
    pd.concat([r["trials"] for r in results]).to_csv(OUTPUT_DIR / "trials.csv", index=False)
    importance = pd.concat([r["importance"] for r in results])
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    seed_diffs = {
        arm.value: _est(
            "gain",
            mse_gain(
                oof["y"].to_numpy(),
                oof[ModelRun(arm, SEEDS[1]).label].to_numpy(),
                oof[ModelRun(arm, SEEDS[0]).label].to_numpy(),
                oof["w_loss"].to_numpy(),
                oof["event_id"].to_numpy(),
                n_boot,
            ),
            METRIC_SCALE,
        )
        for arm in Arm
    }
    tuning_events = data.rows["event_id"].iloc[:n_rows].nunique()
    val_step = config.training.data.val_step_events
    assert val_step is not None
    summary = {
        "git_sha": git_sha(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "config": str(CONFIG_PATH.relative_to(REPO_DIR)),
        "n_trials": n_trials,
        "n_boot": n_boot,
        "rows": len(data.y),
        "events": int(data.rows["event_id"].nunique()),
        "features": data.feature_names,
        "tuning_range": {
            "end": TUNING_END.isoformat(),
            "events": int(tuning_events),
            "rows": n_rows,
            "min_train_events": TUNING_MIN_TRAIN_EVENTS,
            **tuning_fold_references(data, n_rows, val_step),
        },
        "scored": {
            "events": int(oof["event_id"].nunique()),
            "rows": len(oof),
            "first_kickoff": str(oof["commence_time"].min()),
            "last_kickoff": str(oof["commence_time"].max()),
        },
        "unscored_partial_block": {
            "events": int(unscored.loc[unscored.index >= n_rows, "event_id"].nunique()),
            "rows": int((unscored.index >= n_rows).sum()),
        },
        "runs": [
            {
                k: r[k]
                for k in (
                    "run",
                    "best_trial",
                    "best_tuning_cv_wmse",
                    "best_params",
                    "tune_seconds",
                )
            }
            for r in results
        ],
        f"seed_{SEEDS[1]}_minus_seed_{SEEDS[0]}_mse_gain_e6": seed_diffs,
        "concentration": concentration(contrib),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    pd.set_option("display.width", 250)
    print(json.dumps(summary["runs"], indent=2, default=str))
    print(_fmt(table, ["events", "rows", *[f"wmse_{p}" for p in PREDICTIONS]]))
    for p in CHALLENGERS:
        print(_fmt(table, [f"mse_gain_{p}", f"mse_gain_{p}_ci_low", f"mse_gain_{p}_ci_high"], 2))
    for p in CHALLENGERS:
        print(_fmt(table, [f"pick_gain_{p}", f"pick_gain_{p}_ci_low", f"pick_gain_{p}_ci_high"], 3))
    print(json.dumps(summary["concentration"], indent=2))
    print(importance.pivot(index="feature", columns="run", values="mean_gain_share").round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.n_boot))
