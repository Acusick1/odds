"""
Config-driven data preparation for ML model training.

This module bridges the MLTrainingConfig schema with data preparation functions,
enabling configuration-file-based training workflows using composable feature groups.

Key Features:
- Unified entry point using composable feature groups
- Event filtering by date range from DataConfig
- Feature group composition via FeatureConfig.feature_groups
- Train/test splitting from configuration

Example usage:
    ```python
    from odds_analytics.training import MLTrainingConfig
    from odds_analytics.training.data_preparation import prepare_training_data_from_config

    # Load config with feature_groups: ["tabular"]
    config = MLTrainingConfig.from_yaml("experiments/xgboost_v1.yaml")

    # Prepare data
    async with async_session_maker() as session:
        result = await prepare_training_data_from_config(config, session)

        # For XGBoost
        X_train, X_test = result.X_train, result.X_test
        y_train, y_test = result.y_train, result.y_test
        feature_names = result.feature_names

        # For LSTM
        masks_train, masks_test = result.masks_train, result.masks_test
    ```
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import structlog
from odds_core.models import Event, EventStatus
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.training.config import LSTMConfig, MLTrainingConfig

logger = structlog.get_logger()


def _split_aligned_arrays(
    arrays: list[np.ndarray | None],
    **split_kwargs: float | int | bool | None,
) -> list[np.ndarray | None]:
    """Split multiple aligned arrays via train_test_split, skipping Nones.

    Returns a flat list of split results in the same order as input,
    with two entries per input array (train, test). None inputs produce
    (None, None) pairs in the output.
    """
    present = [(i, arr) for i, arr in enumerate(arrays) if arr is not None]
    if not present:
        return [None, None] * len(arrays)

    _, to_split = zip(*present, strict=True)
    split_result = train_test_split(*to_split, **split_kwargs)

    # split_result is [arr0_train, arr0_test, arr1_train, arr1_test, ...]
    result: list[np.ndarray | None] = [None] * (len(arrays) * 2)
    for pair_idx, (orig_idx, _) in enumerate(present):
        result[orig_idx * 2] = split_result[pair_idx * 2]
        result[orig_idx * 2 + 1] = split_result[pair_idx * 2 + 1]
    return result


__all__ = [
    "prepare_training_data_from_config",
    "filter_events_by_date_range",
    "TrainingDataResult",
]


class TrainingDataResult:
    """
    Container for training data preparation results.

    Provides a unified interface for both tabular (XGBoost) and sequence (LSTM) data.

    Attributes:
        X_train: Training features
        X_val: Validation features (optional, None if no validation split)
        X_test: Test features
        y_train: Training targets
        y_val: Validation targets (optional, None if no validation split)
        y_test: Test targets
        feature_names: List of feature names
        masks_train: Training masks (LSTM only, None for XGBoost)
        masks_val: Validation masks (LSTM only, None for XGBoost)
        masks_test: Test masks (LSTM only, None for XGBoost)
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples (0 if no validation split)
        num_test_samples: Number of test samples
        strategy_type: Type of strategy (xgboost_line_movement, lstm_line_movement, etc.)
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        strategy_type: str,
        masks_train: np.ndarray | None = None,
        masks_test: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        masks_val: np.ndarray | None = None,
        event_ids_train: np.ndarray | None = None,
        event_ids_val: np.ndarray | None = None,
        event_ids_test: np.ndarray | None = None,
        static_train: np.ndarray | None = None,
        static_val: np.ndarray | None = None,
        static_test: np.ndarray | None = None,
        static_feature_names: list[str] | None = None,
    ):
        """Initialize training data result."""
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_names
        self.strategy_type = strategy_type
        self.masks_train = masks_train
        self.masks_val = masks_val
        self.masks_test = masks_test
        self.event_ids_train = event_ids_train
        self.event_ids_val = event_ids_val
        self.event_ids_test = event_ids_test
        self.static_train = static_train
        self.static_val = static_val
        self.static_test = static_test
        self.static_feature_names = static_feature_names

    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.X_train)

    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        return len(self.X_val) if self.X_val is not None else 0

    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        return len(self.X_test)

    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
            "strategy_type": self.strategy_type,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "num_test_samples": self.num_test_samples,
            "num_features": self.num_features,
        }
        if self.X_val is not None:
            result["X_val"] = self.X_val
        if self.y_val is not None:
            result["y_val"] = self.y_val
        if self.masks_train is not None:
            result["masks_train"] = self.masks_train
        if self.masks_val is not None:
            result["masks_val"] = self.masks_val
        if self.masks_test is not None:
            result["masks_test"] = self.masks_test
        return result


async def filter_events_by_date_range(
    session: AsyncSession,
    start_date: datetime,
    end_date: datetime,
    status: EventStatus = EventStatus.FINAL,
    data_source: str | None = None,
    min_snapshots: int | None = None,
) -> list[Event]:
    """
    Filter events by date range from the database.

    Args:
        session: Async database session
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        status: Event status to filter by (default: FINAL)
        data_source: Filter by data source ('oddsportal', 'oddsapi', or None)
        min_snapshots: Minimum number of snapshots required per event

    Returns:
        List of Event objects within the date range
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)

    # oddsportal events use 'op_' ID prefix; push to SQL for efficiency
    event_id_pattern = "op_%" if data_source == "oddsportal" else None

    events = await reader.get_events_by_date_range(
        start_date=start_date,
        end_date=end_date,
        status=status,
        event_id_pattern=event_id_pattern,
        min_snapshots=min_snapshots,
    )

    # oddsapi = everything that isn't oddsportal; filter in-memory
    if data_source == "oddsapi":
        events = [e for e in events if not e.id.startswith("op_")]

    logger.info(
        "filtered_events_by_date_range",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        num_events=len(events),
        status=status.value,
        data_source=data_source,
        min_snapshots=min_snapshots,
    )

    return events


async def prepare_training_data_from_config(
    config: MLTrainingConfig,
    session: AsyncSession,
) -> TrainingDataResult:
    """
    Prepare training data using configuration object.

    This is the main entry point for config-driven data preparation. It:
    1. Extracts date range from DataConfig
    2. Filters events by date range
    3. Uses composable feature groups from FeatureConfig.feature_groups
    4. Performs train/test split based on DataConfig

    Args:
        config: MLTrainingConfig object with all training parameters
        session: Async database session

    Returns:
        TrainingDataResult containing train/test splits and metadata

    Raises:
        ValueError: If no valid events found or no valid training data

    Example:
        >>> config = MLTrainingConfig.from_yaml("experiments/config.yaml")
        >>> async with async_session_maker() as session:
        ...     result = await prepare_training_data_from_config(config, session)
        ...     print(f"Training samples: {result.num_train_samples}")
        ...     print(f"Test samples: {result.num_test_samples}")
    """
    from odds_analytics.feature_groups import prepare_training_data

    # Extract configuration
    training_config = config.training
    data_config = training_config.data
    features_config = training_config.features
    strategy_type = training_config.strategy_type

    logger.info(
        "preparing_training_data_from_config",
        experiment_name=config.experiment.name,
        strategy_type=strategy_type,
        feature_groups=features_config.feature_groups,
        target_type=features_config.target_type,
        sampling_strategy=features_config.sampling.strategy,
        start_date=data_config.start_date.isoformat(),
        end_date=data_config.end_date.isoformat(),
    )

    # Convert date to datetime with timezone
    start_datetime = datetime.combine(
        data_config.start_date,
        datetime.min.time(),
        tzinfo=UTC,
    )
    end_datetime = datetime.combine(
        data_config.end_date,
        datetime.max.time(),
        tzinfo=UTC,
    )

    # Normalize data_source: treat "all" as None (no filter)
    data_source = data_config.data_source
    if data_source == "all":
        data_source = None

    # Filter events by date range
    events = await filter_events_by_date_range(
        session=session,
        start_date=start_datetime,
        end_date=end_datetime,
        status=EventStatus.FINAL,
        data_source=data_source,
        min_snapshots=data_config.min_snapshots,
    )

    if not events:
        raise ValueError(
            f"No events found in date range {data_config.start_date} to {data_config.end_date}"
        )

    prep_result = await prepare_training_data(
        events=events,
        session=session,
        config=features_config,
    )

    X = prep_result.X
    y = prep_result.y
    feature_names = prep_result.feature_names
    masks = prep_result.masks
    event_ids = prep_result.event_ids
    static = prep_result.static_features

    if len(X) == 0:
        raise ValueError(f"No valid training data after processing {len(events)} events")

    # Filter events with insufficient valid timesteps (LSTM only)
    if masks is not None and isinstance(training_config.model, LSTMConfig):
        min_ts = training_config.model.min_valid_timesteps
        valid_counts = masks.sum(axis=1)
        keep = valid_counts >= min_ts
        n_before = len(X)
        X = X[keep]
        y = y[keep]
        masks = masks[keep]
        if event_ids is not None:
            event_ids = event_ids[keep]
        if static is not None:
            static = static[keep]
        logger.info(
            "filtered_by_min_valid_timesteps",
            min_valid_timesteps=min_ts,
            kept=int(keep.sum()),
            removed=n_before - int(keep.sum()),
        )
        if len(X) == 0:
            raise ValueError(
                f"No events with >= {min_ts} valid timesteps (all {n_before} events filtered out)"
            )

    # Split: event-level for multi-horizon, row-level for raw
    event_ids_train = None
    event_ids_test = None

    if event_ids is not None:
        # Event-level split: sort by event commence time, split at event boundary
        unique_events = list(dict.fromkeys(event_ids))  # preserve order
        n_test_events = max(1, int(len(unique_events) * data_config.test_split))
        train_events = set(unique_events[:-n_test_events])

        train_mask = np.array([eid in train_events for eid in event_ids])
        test_mask = ~train_mask

        X_trainval = X[train_mask]
        y_trainval = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        event_ids_train = event_ids[train_mask]
        event_ids_test = event_ids[test_mask]
        masks_trainval = masks[train_mask] if masks is not None else None
        masks_test = masks[test_mask] if masks is not None else None
        static_trainval = static[train_mask] if static is not None else None
        static_test = static[test_mask] if static is not None else None
    else:
        split_result = _split_aligned_arrays(
            [X, y, masks, static],
            test_size=data_config.test_split,
            random_state=data_config.random_seed,
            shuffle=data_config.shuffle,
        )
        X_trainval, X_test = split_result[0], split_result[1]
        y_trainval, y_test = split_result[2], split_result[3]
        masks_trainval, masks_test = split_result[4], split_result[5]
        static_trainval, static_test = split_result[6], split_result[7]

    # Second split: separate validation set from training set (if validation_split > 0)
    X_val = None
    y_val = None
    masks_val = None
    masks_train = None
    static_val = None
    static_train = None
    event_ids_val: np.ndarray | None = None

    if data_config.validation_split > 0:
        remaining = 1.0 - data_config.test_split
        val_size_relative = data_config.validation_split / remaining

        if event_ids_train is not None:
            # Event-level val split (matches event-level test split above)
            unique_trainval_events = list(dict.fromkeys(event_ids_train))
            n_val_events = max(1, int(len(unique_trainval_events) * val_size_relative))
            train_events = set(unique_trainval_events[:-n_val_events])

            tv_train_mask = np.array([eid in train_events for eid in event_ids_train])
            tv_val_mask = ~tv_train_mask

            X_train = X_trainval[tv_train_mask]
            y_train = y_trainval[tv_train_mask]
            X_val = X_trainval[tv_val_mask]
            y_val = y_trainval[tv_val_mask]
            masks_train = masks_trainval[tv_train_mask] if masks_trainval is not None else None
            masks_val = masks_trainval[tv_val_mask] if masks_trainval is not None else None
            static_train = static_trainval[tv_train_mask] if static_trainval is not None else None
            static_val = static_trainval[tv_val_mask] if static_trainval is not None else None
            event_ids_val = event_ids_train[tv_val_mask]
            event_ids_train = event_ids_train[tv_train_mask]
        else:
            val_result = _split_aligned_arrays(
                [X_trainval, y_trainval, masks_trainval, static_trainval],
                test_size=val_size_relative,
                random_state=data_config.random_seed,
                shuffle=data_config.shuffle,
            )
            X_train, X_val = val_result[0], val_result[1]
            y_train, y_val = val_result[2], val_result[3]
            masks_train, masks_val = val_result[4], val_result[5]
            static_train, static_val = val_result[6], val_result[7]
    else:
        X_train = X_trainval
        y_train = y_trainval
        masks_train = masks_trainval
        static_train = static_trainval

    logger.info(
        "training_data_prepared_from_config",
        strategy_type=strategy_type,
        feature_groups=features_config.feature_groups,
        target_type=features_config.target_type,
        sampling_strategy=features_config.sampling.strategy,
        total_samples=len(X),
        train_samples=len(X_train),
        val_samples=len(X_val) if X_val is not None else 0,
        test_samples=len(X_test),
        num_features=len(feature_names),
        num_static_features=static.shape[1] if static is not None else 0,
        test_split=data_config.test_split,
        validation_split=data_config.validation_split,
    )

    return TrainingDataResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        strategy_type=strategy_type,
        masks_train=masks_train,
        masks_test=masks_test,
        X_val=X_val,
        y_val=y_val,
        masks_val=masks_val,
        event_ids_train=event_ids_train,
        event_ids_val=event_ids_val,
        event_ids_test=event_ids_test,
        static_train=static_train,
        static_val=static_val,
        static_test=static_test,
        static_feature_names=prep_result.static_feature_names,
    )
