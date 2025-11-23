"""
Config-driven data preparation for ML model training.

This module bridges the MLTrainingConfig schema with data preparation functions,
enabling configuration-file-based training workflows.

Key Features:
- Unified entry point that routes to appropriate preparation function
- Event filtering by date range from DataConfig
- Feature parameter extraction from FeatureConfig
- Backward compatibility with legacy function signatures
- Train/test splitting from configuration

Example usage:
    ```python
    from odds_analytics.training import MLTrainingConfig
    from odds_analytics.training.data_preparation import prepare_training_data_from_config

    # Load config
    config = MLTrainingConfig.from_yaml("experiments/xgboost_v1.yaml")

    # Prepare data
    async with get_async_session() as session:
        result = await prepare_training_data_from_config(config, session)

        # For XGBoost
        X_train, X_test = result["X_train"], result["X_test"]
        y_train, y_test = result["y_train"], result["y_test"]
        feature_names = result["feature_names"]

        # For LSTM
        masks_train, masks_test = result["masks_train"], result["masks_test"]
    ```
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import structlog
from odds_core.models import Event, EventStatus
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.training.config import FeatureConfig, MLTrainingConfig

logger = structlog.get_logger()

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
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        feature_names: List of feature names
        masks_train: Training masks (LSTM only, None for XGBoost)
        masks_test: Test masks (LSTM only, None for XGBoost)
        num_train_samples: Number of training samples
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
    ):
        """Initialize training data result."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.strategy_type = strategy_type
        self.masks_train = masks_train
        self.masks_test = masks_test

    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.X_train)

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
            "num_test_samples": self.num_test_samples,
            "num_features": self.num_features,
        }
        if self.masks_train is not None:
            result["masks_train"] = self.masks_train
        if self.masks_test is not None:
            result["masks_test"] = self.masks_test
        return result


async def filter_events_by_date_range(
    session: AsyncSession,
    start_date: datetime,
    end_date: datetime,
    status: EventStatus = EventStatus.FINAL,
) -> list[Event]:
    """
    Filter events by date range from the database.

    Args:
        session: Async database session
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        status: Event status to filter by (default: FINAL)

    Returns:
        List of Event objects within the date range

    Example:
        >>> from datetime import date
        >>> start = datetime(2024, 10, 1, tzinfo=timezone.utc)
        >>> end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        >>> events = await filter_events_by_date_range(session, start, end)
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)

    events = await reader.get_events_by_date_range(
        start_date=start_date,
        end_date=end_date,
        status=status,
    )

    logger.info(
        "filtered_events_by_date_range",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        num_events=len(events),
        status=status.value,
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
    3. Routes to appropriate preparation function based on strategy_type
    4. Passes feature parameters from FeatureConfig
    5. Performs train/test split based on DataConfig

    Args:
        config: MLTrainingConfig object with all training parameters
        session: Async database session

    Returns:
        TrainingDataResult containing train/test splits and metadata

    Raises:
        ValueError: If no valid events found or strategy type not supported

    Example:
        >>> config = MLTrainingConfig.from_yaml("experiments/config.yaml")
        >>> async with get_async_session() as session:
        ...     result = await prepare_training_data_from_config(config, session)
        ...     print(f"Training samples: {result.num_train_samples}")
        ...     print(f"Test samples: {result.num_test_samples}")
    """
    # Extract configuration
    training_config = config.training
    data_config = training_config.data
    features_config = training_config.features
    strategy_type = training_config.strategy_type

    logger.info(
        "preparing_training_data_from_config",
        experiment_name=config.experiment.name,
        strategy_type=strategy_type,
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

    # Filter events by date range
    events = await filter_events_by_date_range(
        session=session,
        start_date=start_datetime,
        end_date=end_datetime,
        status=EventStatus.FINAL,
    )

    if not events:
        raise ValueError(
            f"No events found in date range {data_config.start_date} to {data_config.end_date}"
        )

    # Route to appropriate preparation function based on strategy type
    if strategy_type in ("xgboost", "xgboost_line_movement"):
        X, y, feature_names = await _prepare_tabular_data(
            events=events,
            session=session,
            features_config=features_config,
        )
        masks = None
    elif strategy_type in ("lstm", "lstm_line_movement"):
        X, y, masks, feature_names = await _prepare_sequence_data(
            events=events,
            session=session,
            features_config=features_config,
            model_config=training_config.model,
        )
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

    if len(X) == 0:
        raise ValueError(f"No valid training data after processing {len(events)} events")

    # Perform train/test split
    if masks is not None:
        # LSTM with masks
        X_train, X_test, y_train, y_test, masks_train, masks_test = train_test_split(
            X,
            y,
            masks,
            test_size=data_config.test_split,
            random_state=data_config.random_seed,
            shuffle=data_config.shuffle,
        )
    else:
        # Tabular without masks
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=data_config.test_split,
            random_state=data_config.random_seed,
            shuffle=data_config.shuffle,
        )
        masks_train = None
        masks_test = None

    logger.info(
        "training_data_prepared_from_config",
        strategy_type=strategy_type,
        total_samples=len(X),
        train_samples=len(X_train),
        test_samples=len(X_test),
        num_features=len(feature_names),
        test_split=data_config.test_split,
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
    )


async def _prepare_tabular_data(
    events: list[Event],
    session: AsyncSession,
    features_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare tabular training data for XGBoost models.

    Args:
        events: List of events with final scores
        session: Async database session
        features_config: Feature extraction configuration

    Returns:
        Tuple of (X, y, feature_names)
    """
    from odds_analytics.feature_extraction import TabularFeatureExtractor
    from odds_analytics.xgboost_line_movement import prepare_tabular_training_data

    # Use the first market from config (h2h is most common)
    market = features_config.markets[0] if features_config.markets else "h2h"

    # Create extractor from config using factory method
    # This extractor can be used for feature extraction and to get feature names
    extractor = TabularFeatureExtractor.from_config(features_config)

    X, y, feature_names = await prepare_tabular_training_data(
        events=events,
        session=session,
        outcome=features_config.outcome,
        market=market,
        opening_hours_before=features_config.opening_hours_before,
        closing_hours_before=features_config.closing_hours_before,
        sharp_bookmakers=extractor.sharp_bookmakers,
        retail_bookmakers=extractor.retail_bookmakers,
    )

    return X, y, feature_names


async def _prepare_sequence_data(
    events: list[Event],
    session: AsyncSession,
    features_config: FeatureConfig,
    model_config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare sequence training data for LSTM models.

    Args:
        events: List of events with final scores
        session: Async database session
        features_config: Feature extraction configuration
        model_config: LSTM model configuration (for lookback_hours, timesteps)

    Returns:
        Tuple of (X, y, masks, feature_names)
    """
    from odds_analytics.feature_extraction import SequenceFeatureExtractor
    from odds_analytics.sequence_loader import (
        TargetType,
        prepare_lstm_training_data,
    )

    # Use the first market from config
    market = features_config.markets[0] if features_config.markets else "h2h"

    # Create extractor from config using factory method
    extractor = SequenceFeatureExtractor.from_config(features_config)

    X, y, masks = await prepare_lstm_training_data(
        events=events,
        session=session,
        outcome=features_config.outcome,
        market=market,
        lookback_hours=features_config.lookback_hours,
        timesteps=features_config.timesteps,
        sharp_bookmakers=features_config.sharp_bookmakers,
        retail_bookmakers=features_config.retail_bookmakers,
        target_type=TargetType.REGRESSION,
        opening_hours_before=features_config.opening_hours_before,
        closing_hours_before=features_config.closing_hours_before,
    )

    # Get feature names from extractor
    feature_names = extractor.get_feature_names()

    return X, y, masks, feature_names
