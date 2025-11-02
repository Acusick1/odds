"""Sequential feature extraction for RNNs/LSTMs."""

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np

from analytics.core import FeatureExtractor, Observation, PredictionProblem, SequentialFeatureSet


class SequentialFeatureExtractor(FeatureExtractor):
    """Extract sequential features for RNN/LSTM models.

    This extractor creates a time series window by sampling observations at regular
    intervals before the decision time. Each time step gets features computed by
    the provided feature computer functions.

    Perfect for:
    - LSTM
    - GRU
    - Transformer
    - Any model expecting sequential/time series data

    Example:
        def compute_odds_at_time(problem, observations, decision_time):
            # Get most recent observation
            if observations:
                recent = max(observations, key=lambda o: o.observation_time)
                return {'odds': float(recent.get_data()['odds'])}
            return {'odds': 0.0}

        extractor = SequentialFeatureExtractor(
            feature_computers=[compute_odds_at_time],
            sequence_length=10,  # 10 time steps
            step_size=timedelta(hours=1)  # 1 hour apart
        )
        features = extractor.extract(problem, observations, decision_time)
        # features.sequences.shape = (10, 1)  # 10 time steps, 1 feature each
    """

    def __init__(
        self,
        feature_computers: list[
            Callable[[PredictionProblem, list[Observation], datetime], dict[str, float]]
        ],
        sequence_length: int = 10,
        step_size: timedelta = timedelta(hours=1),
    ):
        """Initialize sequential feature extractor.

        Args:
            feature_computers: List of functions that compute features at each time step
            sequence_length: Number of time steps in the sequence
            step_size: Time between each step (e.g., 1 hour, 30 minutes)
        """
        self.feature_computers = feature_computers
        self.sequence_length = sequence_length
        self.step_size = step_size

    def extract(
        self,
        problem: PredictionProblem,
        observations: list[Observation],
        decision_time: datetime,
    ) -> SequentialFeatureSet:
        """Extract sequential features from observations.

        Args:
            problem: The prediction problem
            observations: List of observations
            decision_time: Time at which we're making a decision

        Returns:
            SequentialFeatureSet with time series features
        """
        # Create time windows going backwards from decision time
        window_times = self._create_time_windows(decision_time)

        # Compute features at each time step
        sequences = []
        feature_names = None

        for window_time in window_times:
            # Filter observations up to this window time (prevent look-ahead)
            valid_obs = [obs for obs in observations if obs.is_at_or_before(window_time)]

            # Compute features at this time step
            time_features = {}
            for computer in self.feature_computers:
                try:
                    features = computer(problem, valid_obs, window_time)
                    if not isinstance(features, dict):
                        raise ValueError(f"Feature computer {computer.__name__} must return dict")
                    time_features.update(features)
                except Exception as e:
                    # Log but continue
                    print(
                        f"Warning: Feature computer {computer.__name__} failed at {window_time}: {e}"
                    )
                    continue

            # Extract feature names from first time step
            if feature_names is None:
                feature_names = list(time_features.keys())

            # Convert to array (maintain consistent ordering)
            feature_array = [time_features.get(name, 0.0) for name in feature_names]
            sequences.append(feature_array)

        # Convert to numpy array
        sequences_array = np.array(sequences)

        # Handle case where no features were computed
        if sequences_array.size == 0:
            sequences_array = np.zeros((self.sequence_length, 1))
            feature_names = ["empty"]

        return SequentialFeatureSet(
            sequences=sequences_array,
            feature_names=feature_names,
            timestamps=window_times,
        )

    def _create_time_windows(self, decision_time: datetime) -> list[datetime]:
        """Create evenly-spaced time windows before decision time.

        Args:
            decision_time: The decision time

        Returns:
            List of timestamps, ordered from earliest to most recent
        """
        windows = []
        for i in range(self.sequence_length - 1, -1, -1):
            window_time = decision_time - (i * self.step_size)
            windows.append(window_time)
        return windows

    def get_sequence_config(self) -> dict:
        """Get configuration of the sequence.

        Returns:
            Dictionary with sequence_length and step_size
        """
        return {
            "sequence_length": self.sequence_length,
            "step_size_seconds": self.step_size.total_seconds(),
        }
