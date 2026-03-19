"""Analyze whether the recent performance degradation is due to missing Pinnacle.

Compares three scenarios on the 2024+ data:
1. Pinnacle sharp (590 events — excludes 71 post-shutdown events)
2. Betfair Exchange sharp (660 events — includes post-shutdown events)
3. Betfair Exchange sharp with Pinnacle history (full dataset, BFE for 2024+)

The key question: does having a sharp reference in the recent period help,
or is the regime shift about something else entirely?

Usage:
    uv run python experiments/scripts/sharp_regime_analysis.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.cross_validation import run_cv
from odds_analytics.training.data_preparation import prepare_training_data_from_config
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_core.database import async_session_maker

logger = structlog.get_logger()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sharp_regime_analysis"


# Best hyperparams from expanding:50 grid search (all-features)
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


async def load_and_run(
    label: str,
    config_path: Path,
    window_type: str,
    min_train: int,
    max_train: int | None,
    val_step: int,
) -> dict:
    """Load data and run CV with fixed hyperparams."""
    config = MLTrainingConfig.from_yaml(str(config_path))

    # Apply best params
    for param, value in BEST_PARAMS.items():
        setattr(config.training.model, param, value)

    # Apply CV protocol
    config.training.data.window_type = window_type
    config.training.data.min_train_events = min_train
    config.training.data.max_train_events = max_train
    config.training.data.val_step_events = val_step
    config.training.model.n_jobs = 1

    logger.info("loading_data", label=label)
    async with async_session_maker() as session:
        data = await prepare_training_data_from_config(config, session)

    # Combine train+val
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

    strategy = XGBoostLineMovementStrategy()

    logger.info("running_cv", label=label, n_samples=len(X_all), n_features=len(data.feature_names))
    cv_result = run_cv(
        config=config,
        strategy=strategy,
        X=X_all,
        y=y_all,
        feature_names=data.feature_names,
        event_ids=event_ids_all,
    )

    result = {
        "label": label,
        "n_samples": len(X_all),
        "n_events": len(set(event_ids_all)) if event_ids_all is not None else 0,
        "n_features": len(data.feature_names),
        "mean_r2": cv_result.mean_val_r2,
        "std_r2": cv_result.std_val_r2,
        "mean_mse": cv_result.mean_val_mse,
        "n_folds": cv_result.n_folds,
        "folds": [
            {
                "fold": f.fold_idx,
                "n_train": f.n_train,
                "n_val": f.n_val,
                "val_r2": f.val_r2,
                "val_mse": f.val_mse,
                "train_r2": f.train_r2,
            }
            for f in cv_result.fold_results
        ],
    }

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"  Samples: {result['n_samples']} ({result['n_events']} events)")
    print(f"  Mean R²: {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
    print(f"  Mean MSE: {result['mean_mse']:.7f}")
    print(
        f"{'Fold':>6} {'n_train':>8} {'n_val':>6} {'val_R²':>10} {'val_MSE':>12} {'train_R²':>10}"
    )
    print("-" * 56)
    for f in cv_result.fold_results:
        print(
            f"{f.fold_idx:>6} {f.n_train:>8} {f.n_val:>6} {f.val_r2:>10.4f} "
            f"{f.val_mse:>12.7f} {f.train_r2:>10.4f}"
        )
    print(f"{'=' * 70}")

    return result


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Pinnacle sharp, 2024+ only (baseline for this period)
    r = await load_and_run(
        label="pinnacle-sharp / 2024+ / expanding:50",
        config_path=CONFIGS_DIR / "xgboost_epl_pinnacle_sharp_2024.yaml",
        window_type="expanding",
        min_train=380,
        max_train=None,
        val_step=50,
    )
    results.append(r)

    # 2. Betfair Exchange sharp, 2024+ only (more events, includes post-shutdown)
    r = await load_and_run(
        label="betfair-exchange-sharp / 2024+ / expanding:50",
        config_path=CONFIGS_DIR / "xgboost_epl_betfair_sharp.yaml",
        window_type="expanding",
        min_train=380,
        max_train=None,
        val_step=50,
    )
    results.append(r)

    # 3. Pinnacle sharp, full history (our best known result)
    r = await load_and_run(
        label="pinnacle-sharp / full-history / expanding:50",
        config_path=CONFIGS_DIR / "xgboost_epl_combined_all_tuning.yaml",
        window_type="expanding",
        min_train=700,
        max_train=None,
        val_step=50,
    )
    results.append(r)

    # 4. Pinnacle sharp, full history, expanding:150
    r = await load_and_run(
        label="pinnacle-sharp / full-history / expanding:150",
        config_path=CONFIGS_DIR / "xgboost_epl_combined_all_tuning.yaml",
        window_type="expanding",
        min_train=700,
        max_train=None,
        val_step=150,
    )
    results.append(r)

    # Summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY (fixed hyperparams from expanding:50 grid search)")
    print(f"{'=' * 80}")
    print(f"{'Label':<50} {'Events':>7} {'R²':>8} {'±':>7} {'MSE':>10} {'Folds':>6}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['label']:<50} {r['n_events']:>7} {r['mean_r2']:>7.4f} "
            f"{r['std_r2']:>7.4f} {r['mean_mse']:>10.7f} {r['n_folds']:>6}"
        )

    with open(OUTPUT_DIR / "regime_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'regime_analysis.json'}")


if __name__ == "__main__":
    asyncio.run(main())
