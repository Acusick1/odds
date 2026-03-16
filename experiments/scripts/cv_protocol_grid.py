"""CV Protocol Grid Search.

Compares walk-forward CV configurations across independent axes:
1. Window type: expanding vs sliding
2. Window size (sliding only): quarter-season (95), half-season (190),
   1-season (380), 2-season (760)
3. Validation step size: 10, 50, 150 events

Loads data once, then runs Optuna tuning for each grid cell in parallel.
Results saved to experiments/results/cv_protocol_grid/.

Usage:
    uv run python experiments/scripts/cv_protocol_grid.py
    uv run python experiments/scripts/cv_protocol_grid.py --n-trials 50
    uv run python experiments/scripts/cv_protocol_grid.py --cells expanding:150 sliding-380:10
    uv run python experiments/scripts/cv_protocol_grid.py --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import prepare_training_data_from_config
from odds_analytics.training.tuner import OptunaTuner, create_objective
from odds_core.database import async_session_maker

logger = structlog.get_logger()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "cv_protocol_grid"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "xgboost_epl_combined_tuning.yaml"
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120


@dataclass(frozen=True)
class GridCell:
    """A single CV protocol configuration to evaluate."""

    window_type: str  # "expanding" or "sliding"
    max_train_events: int | None  # None for expanding
    val_step_events: int

    @property
    def label(self) -> str:
        if self.window_type == "expanding":
            return f"expanding:{self.val_step_events}"
        return f"sliding-{self.max_train_events}:{self.val_step_events}"

    @property
    def min_train_events(self) -> int:
        if self.window_type == "sliding":
            assert self.max_train_events is not None
            return self.max_train_events
        return 700


DEFAULT_GRID: list[GridCell] = [
    # Expanding window
    GridCell("expanding", None, 10),
    GridCell("expanding", None, 50),
    GridCell("expanding", None, 150),
    # Sliding quarter-season (95 events)
    GridCell("sliding", 95, 10),
    GridCell("sliding", 95, 50),
    GridCell("sliding", 95, 150),
    # Sliding half-season (190 events)
    GridCell("sliding", 190, 10),
    GridCell("sliding", 190, 50),
    GridCell("sliding", 190, 150),
    # Sliding 1-season (380 events)
    GridCell("sliding", 380, 10),
    GridCell("sliding", 380, 50),
    GridCell("sliding", 380, 150),
    # Sliding 2-season (760 events)
    GridCell("sliding", 760, 10),
    GridCell("sliding", 760, 50),
    GridCell("sliding", 760, 150),
]


def parse_cell_spec(spec: str) -> GridCell:
    """Parse a cell spec like 'expanding:150' or 'sliding-380:10'."""
    window_part, val_step_str = spec.split(":")
    val_step = int(val_step_str)

    if window_part == "expanding":
        return GridCell("expanding", None, val_step)

    if window_part.startswith("sliding-"):
        max_train = int(window_part.split("-")[1])
        return GridCell("sliding", max_train, val_step)

    msg = f"Unknown cell spec: {spec}. Expected 'expanding:N' or 'sliding-M:N'"
    raise ValueError(msg)


def apply_cv_config(config: MLTrainingConfig, cell: GridCell) -> MLTrainingConfig:
    """Create a modified config with the given CV parameters."""
    modified = config.model_copy(deep=True)
    modified.training.data.window_type = cell.window_type
    modified.training.data.min_train_events = cell.min_train_events
    modified.training.data.max_train_events = cell.max_train_events
    modified.training.data.val_step_events = cell.val_step_events
    # n_jobs=-1 has ~50x overhead on small datasets (<2K rows) due to thread spawning
    modified.training.model.n_jobs = 1
    return modified


def _run_cell_worker(
    cell: GridCell,
    config: MLTrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    static_train: np.ndarray | None,
    static_val: np.ndarray | None,
    event_ids_train: np.ndarray | None,
    event_ids_val: np.ndarray | None,
    n_trials: int,
) -> dict[str, Any]:
    """Worker function for process pool — runs tuning for a single cell."""
    modified_config = apply_cv_config(config, cell)

    tuner = OptunaTuner(
        study_name=f"cv_grid_{cell.label}",
        direction=modified_config.tuning.direction,
        sampler=modified_config.tuning.sampler,
        pruner=modified_config.tuning.pruner,
        storage=None,
        tracking_config=None,
    )

    objective = create_objective(
        config=modified_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
        static_train=static_train,
        static_val=static_val,
        event_ids_train=event_ids_train,
        event_ids_val=event_ids_val,
    )

    t0 = time.monotonic()
    study = tuner.optimize(objective=objective, n_trials=n_trials)
    elapsed = time.monotonic() - t0

    best_attrs = study.best_trial.user_attrs

    return {
        "cell": cell.label,
        "window_type": cell.window_type,
        "max_train_events": cell.max_train_events,
        "val_step_events": cell.val_step_events,
        "min_train_events": cell.min_train_events,
        "best_mse": study.best_value,
        "mean_r2": best_attrs.get("mean_val_r2"),
        "std_r2": best_attrs.get("std_val_r2"),
        "mean_mse": best_attrs.get("mean_val_mse", study.best_value),
        "std_mse": best_attrs.get("std_val_mse"),
        "n_folds": best_attrs.get("n_folds"),
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "elapsed_seconds": round(elapsed, 1),
        "n_trials": n_trials,
    }


def plot_results(results_df: pd.DataFrame) -> None:
    """Generate comparison plots."""
    results_df["window_label"] = results_df.apply(
        lambda r: (
            "expanding"
            if r["window_type"] == "expanding"
            else f"sliding-{int(r['max_train_events'])}"
        ),
        axis=1,
    )

    pivot = results_df.pivot(index="window_label", columns="val_step_events", values="best_mse")
    # Sort: expanding first, then sliding by window size
    order = ["expanding"] + [
        f"sliding-{s}"
        for s in sorted(
            results_df.loc[results_df["window_type"] == "sliding", "max_train_events"]
            .dropna()
            .unique()
            .astype(int)
        )
    ]
    pivot = pivot.reindex([o for o in order if o in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".6f",
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Best MSE by CV Protocol (lower = better)")
    ax.set_xlabel("val_step_events")
    ax.set_ylabel("Window Type")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mse_heatmap.png")
    plt.close(fig)
    logger.info("saved_plot", path="mse_heatmap.png")

    # Bar chart of best params per cell
    param_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "reg_lambda",
    ]
    params_data = []
    for _, row in results_df.iterrows():
        for k in param_keys:
            if k in row["best_params"]:
                params_data.append(
                    {
                        "cell": row["cell"],
                        "param": k,
                        "value": row["best_params"][k],
                    }
                )

    if params_data:
        params_df = pd.DataFrame(params_data)
        fig, axes = plt.subplots(1, len(param_keys), figsize=(4 * len(param_keys), 6))
        for i, param in enumerate(param_keys):
            subset = params_df[params_df["param"] == param]
            axes[i].barh(subset["cell"], subset["value"])
            axes[i].set_title(param)
            axes[i].tick_params(axis="y", labelsize=7)
        fig.suptitle("Best Hyperparameters by CV Protocol")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "params_comparison.png")
        plt.close(fig)
        logger.info("saved_plot", path="params_comparison.png")


async def main(n_trials: int, cells: list[GridCell], max_workers: int, config_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading_base_config", path=str(config_path))
    config = MLTrainingConfig.from_yaml(str(config_path))

    logger.info("loading_data")
    async with async_session_maker() as session:
        data = await prepare_training_data_from_config(config, session)
    logger.info(
        "data_loaded",
        n_train=data.num_train_samples,
        n_features=len(data.feature_names),
        features=data.feature_names,
    )

    # Extract numpy arrays for pickling across processes
    X_train = data.X_train
    y_train = data.y_train
    feature_names = data.feature_names
    X_val = data.X_val if data.num_val_samples > 0 else data.X_test
    y_val = data.y_val if data.num_val_samples > 0 else data.y_test
    static_train = data.static_train
    static_val = data.static_val if data.num_val_samples > 0 else data.static_test
    event_ids_train = data.event_ids_train
    event_ids_val = data.event_ids_val

    results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    if max_workers == 1:
        # Sequential mode
        for i, cell in enumerate(cells):
            logger.info("running_cell", cell=cell.label, progress=f"{i + 1}/{len(cells)}")
            result = _run_cell_worker(
                cell,
                config,
                X_train,
                y_train,
                feature_names,
                X_val,
                y_val,
                static_train,
                static_val,
                event_ids_train,
                event_ids_val,
                n_trials,
            )
            results.append(result)
            logger.info(
                "cell_complete",
                cell=cell.label,
                best_mse=f"{result['best_mse']:.6f}",
                elapsed=f"{result['elapsed_seconds']}s",
            )
    else:
        # Parallel mode
        logger.info("parallel_execution", max_workers=max_workers, n_cells=len(cells))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cell = {
                executor.submit(
                    _run_cell_worker,
                    cell,
                    config,
                    X_train,
                    y_train,
                    feature_names,
                    X_val,
                    y_val,
                    static_train,
                    static_val,
                    event_ids_train,
                    event_ids_val,
                    n_trials,
                ): cell
                for cell in cells
            }
            for future in as_completed(future_to_cell):
                cell = future_to_cell[future]
                result = future.result()
                results.append(result)
                logger.info(
                    "cell_complete",
                    cell=cell.label,
                    best_mse=f"{result['best_mse']:.6f}",
                    elapsed=f"{result['elapsed_seconds']}s",
                    done=f"{len(results)}/{len(cells)}",
                )

    total_elapsed = time.monotonic() - t_start

    # Sort results by grid order
    cell_order = {c.label: i for i, c in enumerate(cells)}
    results.sort(key=lambda r: cell_order.get(r["cell"], 999))

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)

    with open(OUTPUT_DIR / "grid_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary = results_df[
        [
            "cell",
            "window_type",
            "max_train_events",
            "val_step_events",
            "best_mse",
            "n_folds",
            "elapsed_seconds",
        ]
    ].to_string(index=False)
    logger.info("grid_complete", summary=f"\n{summary}")

    plot_results(results_df)

    print("\n" + "=" * 80)
    print("CV Protocol Grid Search Results")
    print("=" * 80)
    print(f"\nBase config: {config_path.name}")
    print(f"Trials per cell: {n_trials}")
    print(f"Workers: {max_workers}")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print(f"\n{summary}")
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Protocol Grid Search")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per grid cell (default: 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to base config YAML (default: xgboost_epl_combined_tuning.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "cells",
        nargs="*",
        help="Specific cells to run, e.g. 'expanding:150 sliding-380:10'. "
        "If omitted, runs the full default grid.",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH

    if args.cells:
        grid = [parse_cell_spec(c) for c in args.cells]
    else:
        grid = DEFAULT_GRID

    print(
        f"Running {len(grid)} grid cells with {args.n_trials} trials each ({args.workers} workers):"
    )
    for cell in grid:
        print(f"  {cell.label}")
    print()

    asyncio.run(main(args.n_trials, grid, args.workers, config_path))
