"""Feature extraction abstractions.

Defines how features are extracted from observations for different model types.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from .observations import Observation
from .problems import PredictionProblem


class FeatureSet(ABC):
    """Base class for feature containers.

    Different model types require features in different formats:
    - Tree-based models (XGBoost, RandomForest): tabular/flat features
    - Sequential models (LSTM, RNN): time series windows
    - Other models may have other requirements

    This abstraction allows seamless switching between model types.
    """

    @abstractmethod
    def get_shape(self) -> tuple[int, ...]:
        """Return the shape of the features.

        Returns:
            Tuple describing feature dimensions
            - Tabular: (n_features,)
            - Sequential: (sequence_length, n_features)
            - Image: (height, width, channels)
        """
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array.

        Returns:
            Numpy array containing the features, shape matches get_shape()
        """
        pass


class TabularFeatureSet(FeatureSet):
    """Features as a single flat vector - for tree-based models.

    Used by:
    - XGBoost
    - Random Forest
    - Logistic Regression
    - Any model that expects a feature vector

    Example:
        features = {
            'sharp_prob': 0.52,
            'retail_prob': 0.48,
            'market_hold': 0.045,
            'hours_to_game': 2.5,
        }
    """

    def __init__(self, features: dict[str, float]):
        """Initialize tabular feature set.

        Args:
            features: Dictionary mapping feature names to values
        """
        self.features = features
        self._feature_names = list(features.keys())

    def get_shape(self) -> tuple[int]:
        """Return shape of features."""
        return (len(self.features),)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([self.features[name] for name in self._feature_names])

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return self._feature_names.copy()


class SequentialFeatureSet(FeatureSet):
    """Features as a time series window - for sequential models.

    Used by:
    - LSTM
    - GRU
    - Transformer
    - Any model that expects sequential data

    Example:
        # 10 time steps, 4 features each
        sequences = [
            [0.50, 0.48, 0.045, 12.0],  # t-9
            [0.51, 0.48, 0.044, 11.0],  # t-8
            ...
            [0.52, 0.48, 0.043, 2.0],   # t-0 (most recent)
        ]
        shape: (10, 4) = (sequence_length, n_features)
    """

    def __init__(
        self,
        sequences: np.ndarray,
        feature_names: list[str],
        timestamps: list[datetime],
    ):
        """Initialize sequential feature set.

        Args:
            sequences: 2D array of shape (sequence_length, n_features)
            feature_names: Names of features (length = n_features)
            timestamps: Timestamps for each sequence step (length = sequence_length)
        """
        if sequences.ndim != 2:
            raise ValueError(f"sequences must be 2D, got shape {sequences.shape}")

        if len(feature_names) != sequences.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"sequences second dimension ({sequences.shape[1]})"
            )

        if len(timestamps) != sequences.shape[0]:
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match "
                f"sequence_length ({sequences.shape[0]})"
            )

        self.sequences = sequences
        self.feature_names = feature_names
        self.timestamps = timestamps

    def get_shape(self) -> tuple[int, int]:
        """Return shape of sequences."""
        return self.sequences.shape

    def to_array(self) -> np.ndarray:
        """Return sequences array."""
        return self.sequences

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        return self.feature_names.copy()

    def get_timestamps(self) -> list[datetime]:
        """Return list of timestamps for each sequence step."""
        return self.timestamps.copy()


class FeatureExtractor(ABC):
    """Base class for feature extraction.

    Extracts features from observations about a prediction problem.

    Critical constraint: MUST prevent look-ahead bias by only using observations
    that occurred before the decision_time.
    """

    @abstractmethod
    def extract(
        self,
        problem: PredictionProblem,
        observations: list[Observation],
        decision_time: datetime,
    ) -> FeatureSet:
        """Extract features from observations.

        Args:
            problem: The prediction problem
            observations: List of observations (may include future data!)
            decision_time: The time at which we're making a decision

        Returns:
            FeatureSet (tabular or sequential) containing extracted features

        CRITICAL: Implementation MUST filter observations to only include those
        where observation_time <= decision_time to prevent look-ahead bias.
        """
        pass

    def filter_observations(
        self,
        observations: list[Observation],
        decision_time: datetime,
    ) -> list[Observation]:
        """Filter observations to prevent look-ahead bias.

        Args:
            observations: List of all observations
            decision_time: The decision time cutoff

        Returns:
            List of observations that occurred at or before decision_time

        This is a utility method that implementations can use to ensure
        no look-ahead bias.
        """
        return [obs for obs in observations if obs.is_at_or_before(decision_time)]
