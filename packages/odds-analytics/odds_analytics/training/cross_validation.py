"""
Cross-Validation for ML Model Training.

This module provides cross-validation utilities for evaluating ML models,
including aggregate metrics computation (mean ± std).

Supports two cross-validation methods:
- KFold: Standard K-Fold cross-validation with optional shuffling
- TimeSeriesSplit: Walk-forward validation for temporal data (recommended for betting)

Example usage:
    ```python
    from odds_analytics.training.cross_validation import run_cv, CVResult

    # Run 5-fold time series cross-validation (default for temporal data)
    cv_result = run_cv(
        strategy=strategy,
        config=config,
        X=X_train,
        y=y_train,
        feature_names=feature_names,
    )

    # Access aggregated metrics
    print(f"MSE: {cv_result.mean_val_mse:.4f} ± {cv_result.std_val_mse:.4f}")
    print(f"R²: {cv_result.mean_val_r2:.4f} ± {cv_result.std_val_r2:.4f}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import structlog
from sklearn.model_selection import KFold, TimeSeriesSplit

if TYPE_CHECKING:
    from odds_analytics.training.config import MLTrainingConfig
    from odds_analytics.training.tracking import ExperimentTracker


@runtime_checkable
class TrainableStrategy(Protocol):
    def train_from_config(
        self,
        config: MLTrainingConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        X_val: np.ndarray | None = ...,
        y_val: np.ndarray | None = ...,
        tracker: ExperimentTracker | None = ...,
        trial: Any | None = ...,
    ) -> dict[str, Any]: ...


logger = structlog.get_logger()

__all__ = [
    "CVFoldResult",
    "CVResult",
    "TrainableStrategy",
    "run_cv",
    "train_with_cv",
]


@dataclass
class CVFoldResult:
    """
    Results from a single cross-validation fold.

    Attributes:
        fold_idx: Zero-indexed fold number
        train_mse: Mean squared error on training data
        train_mae: Mean absolute error on training data
        train_r2: R² score on training data
        val_mse: Mean squared error on validation fold
        val_mae: Mean absolute error on validation fold
        val_r2: R² score on validation fold
        n_train: Number of training samples
        n_val: Number of validation samples
    """

    fold_idx: int
    train_mse: float
    train_mae: float
    train_r2: float
    val_mse: float
    val_mae: float
    val_r2: float
    n_train: int
    n_val: int


@dataclass
class CVResult:
    """
    Aggregated cross-validation results.

    Contains per-fold results and computes aggregate statistics
    (mean and standard deviation) for all metrics.

    Attributes:
        fold_results: List of CVFoldResult for each fold
        n_folds: Number of folds used
        random_seed: Random seed used for shuffling (only used for kfold method)
        cv_method: Cross-validation method used ('kfold' or 'timeseries')

    Properties:
        mean_val_mse, std_val_mse: Validation MSE statistics
        mean_val_mae, std_val_mae: Validation MAE statistics
        mean_val_r2, std_val_r2: Validation R² statistics
        mean_train_mse, std_train_mse: Training MSE statistics
    """

    fold_results: list[CVFoldResult]
    n_folds: int
    random_seed: int
    cv_method: Literal["kfold", "timeseries", "group_timeseries"] = "kfold"
    _val_mse_stats: tuple[float, float] = field(init=False, repr=False)
    _val_mae_stats: tuple[float, float] = field(init=False, repr=False)
    _val_r2_stats: tuple[float, float] = field(init=False, repr=False)
    _train_mse_stats: tuple[float, float] = field(init=False, repr=False)
    _train_mae_stats: tuple[float, float] = field(init=False, repr=False)
    _train_r2_stats: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute aggregate statistics after initialization."""
        val_mse = [f.val_mse for f in self.fold_results]
        val_mae = [f.val_mae for f in self.fold_results]
        val_r2 = [f.val_r2 for f in self.fold_results]
        train_mse = [f.train_mse for f in self.fold_results]
        train_mae = [f.train_mae for f in self.fold_results]
        train_r2 = [f.train_r2 for f in self.fold_results]

        self._val_mse_stats = (float(np.mean(val_mse)), float(np.std(val_mse)))
        self._val_mae_stats = (float(np.mean(val_mae)), float(np.std(val_mae)))
        self._val_r2_stats = (float(np.mean(val_r2)), float(np.std(val_r2)))
        self._train_mse_stats = (float(np.mean(train_mse)), float(np.std(train_mse)))
        self._train_mae_stats = (float(np.mean(train_mae)), float(np.std(train_mae)))
        self._train_r2_stats = (float(np.mean(train_r2)), float(np.std(train_r2)))

    # Validation metrics
    @property
    def mean_val_mse(self) -> float:
        """Mean validation MSE across all folds."""
        return self._val_mse_stats[0]

    @property
    def std_val_mse(self) -> float:
        """Standard deviation of validation MSE across all folds."""
        return self._val_mse_stats[1]

    @property
    def mean_val_mae(self) -> float:
        """Mean validation MAE across all folds."""
        return self._val_mae_stats[0]

    @property
    def std_val_mae(self) -> float:
        """Standard deviation of validation MAE across all folds."""
        return self._val_mae_stats[1]

    @property
    def mean_val_r2(self) -> float:
        """Mean validation R² across all folds."""
        return self._val_r2_stats[0]

    @property
    def std_val_r2(self) -> float:
        """Standard deviation of validation R² across all folds."""
        return self._val_r2_stats[1]

    # Training metrics
    @property
    def mean_train_mse(self) -> float:
        """Mean training MSE across all folds."""
        return self._train_mse_stats[0]

    @property
    def std_train_mse(self) -> float:
        """Standard deviation of training MSE across all folds."""
        return self._train_mse_stats[1]

    @property
    def mean_train_mae(self) -> float:
        """Mean training MAE across all folds."""
        return self._train_mae_stats[0]

    @property
    def std_train_mae(self) -> float:
        """Standard deviation of training MAE across all folds."""
        return self._train_mae_stats[1]

    @property
    def mean_train_r2(self) -> float:
        """Mean training R² across all folds."""
        return self._train_r2_stats[0]

    @property
    def std_train_r2(self) -> float:
        """Standard deviation of training R² across all folds."""
        return self._train_r2_stats[1]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format for training history.

        Returns:
            Dictionary with cv_ prefix for all cross-validation metrics
        """
        return {
            "cv_n_folds": self.n_folds,
            "cv_random_seed": self.random_seed,
            "cv_method": self.cv_method,
            # Validation metrics (mean ± std)
            "cv_val_mse_mean": self.mean_val_mse,
            "cv_val_mse_std": self.std_val_mse,
            "cv_val_mae_mean": self.mean_val_mae,
            "cv_val_mae_std": self.std_val_mae,
            "cv_val_r2_mean": self.mean_val_r2,
            "cv_val_r2_std": self.std_val_r2,
            # Training metrics (mean ± std)
            "cv_train_mse_mean": self.mean_train_mse,
            "cv_train_mse_std": self.std_train_mse,
            "cv_train_mae_mean": self.mean_train_mae,
            "cv_train_mae_std": self.std_train_mae,
            "cv_train_r2_mean": self.mean_train_r2,
            "cv_train_r2_std": self.std_train_r2,
            # Per-fold details
            "cv_fold_results": [
                {
                    "fold": f.fold_idx,
                    "train_mse": f.train_mse,
                    "train_mae": f.train_mae,
                    "train_r2": f.train_r2,
                    "val_mse": f.val_mse,
                    "val_mae": f.val_mae,
                    "val_r2": f.val_r2,
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                }
                for f in self.fold_results
            ],
        }


def make_group_timeseries_splits(
    event_ids: np.ndarray,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return walk-forward row-level splits grouped by event boundary.

    Splits on unique event IDs using TimeSeriesSplit, then expands each
    event-level split back to row indices. Assumes event_ids are ordered
    chronologically (as returned by prepare_training_data).

    Args:
        event_ids: Per-row event identifiers in chronological order.
        n_folds: Number of CV folds.

    Returns:
        List of (train_row_indices, val_row_indices) tuples, one per fold.
    """
    unique_events = list(dict.fromkeys(event_ids))  # preserve chronological order
    event_indices = np.arange(len(unique_events))
    splits = []
    for ev_train_idx, ev_val_idx in TimeSeriesSplit(n_splits=n_folds).split(event_indices):
        train_events = {unique_events[i] for i in ev_train_idx}
        val_events = {unique_events[i] for i in ev_val_idx}
        splits.append(
            (
                np.where([eid in train_events for eid in event_ids])[0],
                np.where([eid in val_events for eid in event_ids])[0],
            )
        )
    return splits


def run_cv(
    strategy: TrainableStrategy,
    config: MLTrainingConfig,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    event_ids: np.ndarray | None = None,
) -> CVResult:
    """
    Run cross-validation on the provided data.

    Supports two cross-validation methods controlled by config.training.data.cv_method:
    - 'timeseries': Walk-forward validation using TimeSeriesSplit (default, recommended
      for temporal betting data). Training windows expand with each fold, preserving
      temporal order. Data must be pre-sorted chronologically.
    - 'kfold': Standard K-Fold cross-validation with optional shuffling.

    This function:
    1. Selects splitter based on cv_method (TimeSeriesSplit or KFold)
    2. For each fold: trains on training set, validates on validation fold
    3. Collects metrics from each fold
    4. Returns aggregated CVResult with mean ± std for all metrics

    Note: This function does NOT train a final model. After CV, call
    train_from_config() separately to train the final model on all data.

    Args:
        strategy: Strategy instance implementing train_from_config() that returns
            a history dict with keys: train_mse, train_mae, train_r2, val_mse,
            val_mae, val_r2. Supports XGBoostLineMovementStrategy and
            LSTMLineMovementStrategy.
        config: MLTrainingConfig with data.cv_method, data.n_folds, etc.
        X: Feature matrix (n_samples, n_features). Must be sorted chronologically
           when using cv_method='timeseries'.
        y: Target vector (n_samples,)
        feature_names: List of feature names

    Returns:
        CVResult with per-fold and aggregate metrics

    Example:
        >>> strategy = XGBoostLineMovementStrategy()
        >>> cv_result = run_cv(strategy, config, X_train, y_train, feature_names)
        >>> print(f"CV MSE: {cv_result.mean_val_mse:.4f} ± {cv_result.std_val_mse:.4f}")
    """
    data_config = config.training.data
    n_folds = data_config.n_folds
    cv_method = data_config.cv_method
    shuffle = data_config.kfold_shuffle
    random_seed = data_config.random_seed

    # Select cross-validation splitter based on method
    if cv_method == "group_timeseries" and event_ids is None:
        logger.warning(
            "group_timeseries_missing_event_ids",
            message="cv_method='group_timeseries' but event_ids not provided. "
            "Falling back to standard timeseries CV.",
        )
        cv_method = "timeseries"

    if cv_method == "group_timeseries" and event_ids is not None:
        group_splits = make_group_timeseries_splits(event_ids, n_folds)

        logger.info(
            "starting_group_timeseries_cv",
            n_folds=n_folds,
            n_samples=len(X),
            n_events=len(dict.fromkeys(event_ids)),
            n_features=len(feature_names),
            cv_method=cv_method,
        )

        fold_iter: list[tuple[np.ndarray, np.ndarray]] = group_splits

    elif cv_method == "timeseries":
        if shuffle:
            logger.warning(
                "timeseries_cv_ignoring_shuffle",
                message="kfold_shuffle=True is ignored when cv_method='timeseries'. "
                "TimeSeriesSplit preserves temporal order by design.",
            )

        splitter = TimeSeriesSplit(n_splits=n_folds)
        logger.info(
            "starting_timeseries_cv",
            n_folds=n_folds,
            n_samples=len(X),
            n_features=len(feature_names),
            cv_method=cv_method,
            message="Using walk-forward validation. Ensure data is sorted chronologically.",
        )
        fold_iter = list(splitter.split(X))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
        logger.info(
            "starting_kfold_cv",
            n_folds=n_folds,
            n_samples=len(X),
            n_features=len(feature_names),
            shuffle=shuffle,
            random_seed=random_seed,
            cv_method=cv_method,
        )
        fold_iter = list(splitter.split(X))

    fold_results: list[CVFoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        logger.debug(
            "training_fold",
            fold=fold_idx + 1,
            n_folds=n_folds,
            n_train=len(train_idx),
            n_val=len(val_idx),
            cv_method=cv_method,
        )

        # Train model on this fold
        history = strategy.train_from_config(
            config=config,
            X_train=X_train_fold,
            y_train=y_train_fold,
            feature_names=feature_names,
            X_val=X_val_fold,
            y_val=y_val_fold,
        )

        fold_result = CVFoldResult(
            fold_idx=fold_idx,
            train_mse=history["train_mse"],
            train_mae=history["train_mae"],
            train_r2=history["train_r2"],
            val_mse=history["val_mse"],
            val_mae=history["val_mae"],
            val_r2=history["val_r2"],
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        fold_results.append(fold_result)

        logger.info(
            "fold_complete",
            fold=fold_idx + 1,
            n_folds=n_folds,
            val_mse=fold_result.val_mse,
            val_r2=fold_result.val_r2,
            cv_method=cv_method,
        )

    cv_result = CVResult(
        fold_results=fold_results,
        n_folds=n_folds,
        random_seed=random_seed,
        cv_method=cv_method,
    )

    logger.info(
        "cv_complete",
        cv_method=cv_method,
        n_folds=n_folds,
        mean_val_mse=cv_result.mean_val_mse,
        std_val_mse=cv_result.std_val_mse,
        mean_val_r2=cv_result.mean_val_r2,
        std_val_r2=cv_result.std_val_r2,
    )

    return cv_result


def train_with_cv(
    strategy: TrainableStrategy,
    config: MLTrainingConfig,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    event_ids: np.ndarray | None = None,
) -> tuple[dict[str, Any], CVResult]:
    """Run CV, train final model on all data, merge CV metrics into history."""
    logger.info(
        "starting_train_with_cv",
        experiment_name=config.experiment.name,
        n_folds=config.training.data.n_folds,
        n_samples=len(X),
        n_features=len(feature_names),
    )

    cv_result = run_cv(
        strategy=strategy,
        config=config,
        X=X,
        y=y,
        feature_names=feature_names,
        event_ids=event_ids,
    )

    logger.info(
        "cv_complete_training_final",
        cv_val_mse=f"{cv_result.mean_val_mse:.6f} ± {cv_result.std_val_mse:.6f}",
        cv_val_r2=f"{cv_result.mean_val_r2:.4f} ± {cv_result.std_val_r2:.4f}",
    )

    history = strategy.train_from_config(
        config=config,
        X_train=X,
        y_train=y,
        feature_names=feature_names,
        X_val=X_test,
        y_val=y_test,
    )

    history.update(cv_result.to_dict())

    logger.info(
        "train_with_cv_complete",
        experiment_name=config.experiment.name,
        cv_val_mse_mean=cv_result.mean_val_mse,
        final_train_mse=history.get("train_mse"),
        final_test_mse=history.get("val_mse"),
    )

    return history, cv_result
