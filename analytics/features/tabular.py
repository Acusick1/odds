"""Tabular feature extraction for tree-based models (XGBoost, Random Forest, etc.)."""

from collections.abc import Callable
from datetime import datetime

from analytics.core import FeatureExtractor, Observation, PredictionProblem, TabularFeatureSet


class TabularFeatureExtractor(FeatureExtractor):
    """Extract tabular features for tree-based models.

    This extractor takes a list of feature computer functions and applies them
    to observations to create a single feature vector. Each feature computer
    function receives the problem, observations, and decision time, and returns
    a dictionary of features.

    Perfect for:
    - XGBoost
    - Random Forest
    - Logistic Regression
    - Any model expecting a flat feature vector

    Example:
        def compute_avg_odds(problem, observations, decision_time):
            valid_obs = [o for o in observations if o.observation_time <= decision_time]
            avg = sum([o.get_data()['odds'] for o in valid_obs]) / len(valid_obs)
            return {'avg_odds': avg}

        extractor = TabularFeatureExtractor([compute_avg_odds])
        features = extractor.extract(problem, observations, decision_time)
        # features.features = {'avg_odds': -105.5}
    """

    def __init__(
        self,
        feature_computers: list[
            Callable[[PredictionProblem, list[Observation], datetime], dict[str, float]]
        ],
    ):
        """Initialize tabular feature extractor.

        Args:
            feature_computers: List of functions that compute features.
                Each function signature: (problem, observations, decision_time) -> dict[str, float]
        """
        self.feature_computers = feature_computers

    def extract(
        self,
        problem: PredictionProblem,
        observations: list[Observation],
        decision_time: datetime,
    ) -> TabularFeatureSet:
        """Extract tabular features from observations.

        Args:
            problem: The prediction problem
            observations: List of observations (will be filtered to prevent look-ahead)
            decision_time: Time at which we're making a decision

        Returns:
            TabularFeatureSet with all computed features
        """
        # Filter observations to prevent look-ahead bias
        valid_obs = self.filter_observations(observations, decision_time)

        # Compute features by applying all feature computers
        all_features = {}
        for computer in self.feature_computers:
            try:
                features = computer(problem, valid_obs, decision_time)
                if not isinstance(features, dict):
                    raise ValueError(
                        f"Feature computer {computer.__name__} must return dict, got {type(features)}"
                    )
                all_features.update(features)
            except Exception as e:
                # Log the error but continue with other features
                print(f"Warning: Feature computer {computer.__name__} failed: {e}")
                continue

        return TabularFeatureSet(all_features)

    def get_feature_names(self) -> list[str]:
        """Get list of feature names that will be computed.

        Note: This is a best-effort method - actual features may differ if
        feature computers are conditional.

        Returns:
            List of expected feature names
        """
        # This requires running the feature computers, which we can't do without data
        # Could be implemented by having feature computers declare their output names
        raise NotImplementedError(
            "Cannot determine feature names without running extraction. "
            "Feature computers should document their output keys."
        )
