"""Validate hybrid sharp reference (Pinnacle + Betfair Exchange fallback).

Compares:
1. Pinnacle-only (1729 events) — current baseline
2. Hybrid: Pinnacle when available, BFE fallback (1800 events)

Both use expanding:50 with fixed hyperparams for direct comparison,
then tune each to find optimal hyperparams.

Usage:
    uv run python experiments/scripts/hybrid_sharp_validation.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import run_cv
from odds_analytics.training.data_preparation import (
    TrainingDataResult,
    prepare_training_data_from_config,
)
from odds_analytics.training.tuner import OptunaTuner, create_objective
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_core.database import async_session_maker

logger = structlog.get_logger()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "hybrid_sharp_validation"

# Best hyperparams from expanding:50 grid search (all-features, pinnacle-only)
BEST_PARAMS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.2104582297727651,
    "min_child_weight": 41,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "gamma": 0.0016454553957547594,
    "reg_alpha": 0.06136087356050856,
    "reg_lambda": 2.3136508277063026,
}

CONFIGS = [
    ("pinnacle-only", CONFIGS_DIR / "xgboost_epl_combined_all_tuning.yaml"),
    ("hybrid-sharp", CONFIGS_DIR / "xgboost_epl_hybrid_sharp.yaml"),
]


def prepare_data(data: TrainingDataResult) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Combine train+val for walk-forward CV."""
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
    return X_all, y_all, event_ids_all


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for label, config_path in CONFIGS:
        config = MLTrainingConfig.from_yaml(str(config_path))

        logger.info("loading_data", label=label)
        async with async_session_maker() as session:
            data = await prepare_training_data_from_config(config, session)

        X_all, y_all, event_ids_all = prepare_data(data)

        # --- Part 1: Fixed hyperparams (expanding:50) ---
        config_fixed = MLTrainingConfig.from_yaml(str(config_path))
        for param, value in BEST_PARAMS.items():
            setattr(config_fixed.training.model, param, value)
        config_fixed.training.data.window_type = "expanding"
        config_fixed.training.data.min_train_events = 700
        config_fixed.training.data.max_train_events = None
        config_fixed.training.data.val_step_events = 50
        config_fixed.training.model.n_jobs = 1

        strategy = XGBoostLineMovementStrategy()
        cv_fixed = run_cv(
            config=config_fixed,
            strategy=strategy,
            X=X_all,
            y=y_all,
            feature_names=data.feature_names,
            event_ids=event_ids_all,
        )

        print(f"\n{'=' * 70}")
        print(f"{label} — FIXED HYPERPARAMS (expanding:50)")
        print(f"  Events: {len(set(event_ids_all))}, Samples: {len(X_all)}")
        print(f"  Mean R²: {cv_fixed.mean_val_r2:.4f} ± {cv_fixed.std_val_r2:.4f}")
        print(f"  Mean MSE: {cv_fixed.mean_val_mse:.7f}")
        print(f"  Folds: {cv_fixed.n_folds}")
        print(f"{'Fold':>6} {'n_train':>8} {'n_val':>6} {'val_R²':>10} {'val_MSE':>12}")
        print("-" * 46)
        for f in cv_fixed.fold_results:
            print(
                f"{f.fold_idx:>6} {f.n_train:>8} {f.n_val:>6} {f.val_r2:>10.4f} {f.val_mse:>12.7f}"
            )

        # --- Part 2: Tuned hyperparams (expanding:50) ---
        config_tune = MLTrainingConfig.from_yaml(str(config_path))
        config_tune.training.data.window_type = "expanding"
        config_tune.training.data.min_train_events = 700
        config_tune.training.data.max_train_events = None
        config_tune.training.data.val_step_events = 50
        config_tune.training.model.n_jobs = 1

        tuner = OptunaTuner(
            study_name=f"hybrid_validation_{label}",
            direction=config_tune.tuning.direction,
            sampler=config_tune.tuning.sampler,
            pruner=config_tune.tuning.pruner,
            storage=None,
            tracking_config=None,
        )

        objective = create_objective(
            config=config_tune,
            X_train=X_all,
            y_train=y_all,
            feature_names=data.feature_names,
            X_val=None,
            y_val=None,
            static_train=None,
            static_val=None,
            event_ids_train=event_ids_all,
            event_ids_val=None,
        )

        t0 = time.monotonic()
        study = tuner.optimize(objective=objective, n_trials=100)
        elapsed = time.monotonic() - t0

        best_attrs = study.best_trial.user_attrs
        tuned_r2 = best_attrs.get("mean_val_r2")
        tuned_std = best_attrs.get("std_val_r2")
        tuned_mse = study.best_value

        print(f"\n{label} — TUNED (expanding:50, 100 trials, {elapsed:.0f}s)")
        print(f"  Best MSE: {tuned_mse:.7f}")
        print(f"  Mean R²: {tuned_r2:.4f} ± {tuned_std:.4f}")
        print(f"  Best params: {study.best_trial.params}")
        print(f"{'=' * 70}")

        all_results.append(
            {
                "label": label,
                "n_events": len(set(event_ids_all)),
                "n_samples": len(X_all),
                "n_features": len(data.feature_names),
                "fixed": {
                    "mean_r2": cv_fixed.mean_val_r2,
                    "std_r2": cv_fixed.std_val_r2,
                    "mean_mse": cv_fixed.mean_val_mse,
                    "n_folds": cv_fixed.n_folds,
                    "folds": [
                        {
                            "fold": f.fold_idx,
                            "n_train": f.n_train,
                            "n_val": f.n_val,
                            "val_r2": f.val_r2,
                            "val_mse": f.val_mse,
                            "train_r2": f.train_r2,
                        }
                        for f in cv_fixed.fold_results
                    ],
                },
                "tuned": {
                    "mean_r2": tuned_r2,
                    "std_r2": tuned_std,
                    "best_mse": tuned_mse,
                    "best_params": study.best_trial.params,
                    "elapsed_seconds": round(elapsed, 1),
                },
            }
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Label':<20} {'Events':>7} {'Fixed R²':>10} {'Tuned R²':>10} {'Tuned MSE':>12}")
    print("-" * 62)
    for r in all_results:
        print(
            f"{r['label']:<20} {r['n_events']:>7} "
            f"{r['fixed']['mean_r2']:>9.4f} "
            f"{r['tuned']['mean_r2']:>9.4f} "
            f"{r['tuned']['best_mse']:>12.7f}"
        )

    with open(OUTPUT_DIR / "validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'validation_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
