"""Unit tests for core time series abstractions."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from analytics.core import (
    ContinuousPredictionProblem,
    DiscreteEventProblem,
    FeatureExtractor,
    ModelPredictor,
    Observation,
    Prediction,
    SequentialFeatureSet,
    TabularFeatureSet,
)

# Test implementations for abstract classes


class MockDiscreteEvent(DiscreteEventProblem):
    """Mock discrete event for testing."""

    def __init__(self, event_id: str, event_time: datetime, outcome: str | None = None):
        self.id = event_id
        self.timestamp = event_time
        self.event_time = event_time
        self.outcome = outcome


class MockContinuousPrediction(ContinuousPredictionProblem):
    """Mock continuous prediction for testing."""

    def __init__(
        self,
        prob_id: str,
        start: datetime,
        end: datetime,
        value: float | None = None,
    ):
        self.id = prob_id
        self.timestamp = start
        self.prediction_window_start = start
        self.prediction_window_end = end
        self.realized_value = value


class MockObservation(Observation):
    """Mock observation for testing."""

    def __init__(self, problem_id: str, obs_time: datetime, data: dict):
        self.problem_id = problem_id
        self.observation_time = obs_time
        self._data = data

    def get_data(self) -> dict:
        return self._data


class MockTabularExtractor(FeatureExtractor):
    """Mock feature extractor that returns tabular features."""

    def extract(self, problem, observations, decision_time):
        valid_obs = self.filter_observations(observations, decision_time)
        # Simple feature: count of observations
        return TabularFeatureSet({"obs_count": float(len(valid_obs))})


class MockModelPredictor(ModelPredictor):
    """Mock model predictor for testing."""

    def predict(self, features):
        # Simple mock: predict based on feature count
        if isinstance(features, TabularFeatureSet):
            count = features.features.get("obs_count", 0)
            prob = min(count / 10.0, 1.0)
            return Prediction(
                predictions={"positive": prob, "negative": 1 - prob},
                confidence=abs(0.5 - prob) + 0.5,
                metadata={"model_type": "mock"},
            )
        raise ValueError("Mock model requires TabularFeatureSet")

    def get_required_feature_type(self) -> str:
        return "tabular"


# Tests for PredictionProblem classes


class TestDiscreteEventProblem:
    """Tests for DiscreteEventProblem."""

    def test_creation(self):
        """Test creating a discrete event problem."""
        event_time = datetime(2024, 1, 15, 19, 0)
        event = MockDiscreteEvent("game1", event_time, outcome="home_win")

        assert event.id == "game1"
        assert event.event_time == event_time
        assert event.get_outcome() == "home_win"
        assert event.get_problem_type() == "discrete_event"

    def test_outcome_initially_none(self):
        """Test that outcome can be None initially."""
        event = MockDiscreteEvent("game2", datetime.now())

        assert event.get_outcome() is None


class TestContinuousPredictionProblem:
    """Tests for ContinuousPredictionProblem."""

    def test_creation(self):
        """Test creating a continuous prediction problem."""
        start = datetime(2024, 1, 15, 9, 0)
        end = datetime(2024, 1, 15, 16, 0)
        prob = MockContinuousPrediction("trade1", start, end, value=0.025)

        assert prob.id == "trade1"
        assert prob.prediction_window_start == start
        assert prob.prediction_window_end == end
        assert prob.get_outcome() == 0.025
        assert prob.get_problem_type() == "continuous_prediction"

    def test_value_initially_none(self):
        """Test that realized_value can be None initially."""
        start = datetime.now()
        end = start + timedelta(hours=1)
        prob = MockContinuousPrediction("trade2", start, end)

        assert prob.get_outcome() is None


# Tests for Observation


class TestObservation:
    """Tests for Observation base class."""

    def test_creation(self):
        """Test creating an observation."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = MockObservation("game1", obs_time, {"value": 100})

        assert obs.problem_id == "game1"
        assert obs.observation_time == obs_time
        assert obs.get_data() == {"value": 100}

    def test_is_before(self):
        """Test is_before method."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = MockObservation("game1", obs_time, {})

        later_time = datetime(2024, 1, 15, 19, 0)
        earlier_time = datetime(2024, 1, 15, 17, 0)

        assert obs.is_before(later_time) is True
        assert obs.is_before(earlier_time) is False
        assert obs.is_before(obs_time) is False

    def test_is_at_or_before(self):
        """Test is_at_or_before method."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = MockObservation("game1", obs_time, {})

        later_time = datetime(2024, 1, 15, 19, 0)
        earlier_time = datetime(2024, 1, 15, 17, 0)

        assert obs.is_at_or_before(later_time) is True
        assert obs.is_at_or_before(earlier_time) is False
        assert obs.is_at_or_before(obs_time) is True  # Equal time


# Tests for FeatureSet classes


class TestTabularFeatureSet:
    """Tests for TabularFeatureSet."""

    def test_creation(self):
        """Test creating tabular features."""
        features = {"feat1": 1.0, "feat2": 2.0, "feat3": 3.0}
        feat_set = TabularFeatureSet(features)

        assert feat_set.features == features
        assert feat_set.get_shape() == (3,)

    def test_to_array(self):
        """Test converting to numpy array."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        feat_set = TabularFeatureSet(features)

        arr = feat_set.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert list(arr) == [1.0, 2.0, 3.0]

    def test_get_feature_names(self):
        """Test getting feature names."""
        features = {"feat1": 1.0, "feat2": 2.0}
        feat_set = TabularFeatureSet(features)

        names = feat_set.get_feature_names()
        assert names == ["feat1", "feat2"]


class TestSequentialFeatureSet:
    """Tests for SequentialFeatureSet."""

    def test_creation(self):
        """Test creating sequential features."""
        sequences = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        feature_names = ["feat1", "feat2"]
        timestamps = [datetime(2024, 1, 15, i, 0) for i in range(3)]

        feat_set = SequentialFeatureSet(sequences, feature_names, timestamps)

        assert feat_set.get_shape() == (3, 2)
        assert np.array_equal(feat_set.to_array(), sequences)
        assert feat_set.get_feature_names() == feature_names
        assert feat_set.get_timestamps() == timestamps

    def test_validation_wrong_dimensions(self):
        """Test that 1D array raises error."""
        sequences = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D
        feature_names = ["feat1"]
        timestamps = [datetime.now()]

        with pytest.raises(ValueError, match="sequences must be 2D"):
            SequentialFeatureSet(sequences, feature_names, timestamps)

    def test_validation_feature_names_mismatch(self):
        """Test that feature names length must match."""
        sequences = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["feat1"]  # Should be 2
        timestamps = [datetime.now(), datetime.now()]

        with pytest.raises(ValueError, match="feature_names length"):
            SequentialFeatureSet(sequences, feature_names, timestamps)

    def test_validation_timestamps_mismatch(self):
        """Test that timestamps length must match sequence length."""
        sequences = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["feat1", "feat2"]
        timestamps = [datetime.now()]  # Should be 2

        with pytest.raises(ValueError, match="timestamps length"):
            SequentialFeatureSet(sequences, feature_names, timestamps)


# Tests for FeatureExtractor


class TestFeatureExtractor:
    """Tests for FeatureExtractor base class."""

    def test_filter_observations(self):
        """Test filtering observations to prevent look-ahead bias."""
        decision_time = datetime(2024, 1, 15, 18, 0)

        obs1 = MockObservation("game1", datetime(2024, 1, 15, 17, 0), {})  # Before
        obs2 = MockObservation("game1", datetime(2024, 1, 15, 18, 0), {})  # Exactly at
        obs3 = MockObservation("game1", datetime(2024, 1, 15, 19, 0), {})  # After

        extractor = MockTabularExtractor()
        filtered = extractor.filter_observations([obs1, obs2, obs3], decision_time)

        assert len(filtered) == 2
        assert obs1 in filtered
        assert obs2 in filtered
        assert obs3 not in filtered

    def test_extract_prevents_lookahead(self):
        """Test that extract only uses observations before decision time."""
        decision_time = datetime(2024, 1, 15, 18, 0)
        event = MockDiscreteEvent("game1", datetime(2024, 1, 15, 19, 0))

        # Create observations before, at, and after decision time
        observations = [
            MockObservation("game1", datetime(2024, 1, 15, 17, 0), {"v": 1}),
            MockObservation("game1", datetime(2024, 1, 15, 18, 0), {"v": 2}),
            MockObservation("game1", datetime(2024, 1, 15, 19, 0), {"v": 3}),
        ]

        extractor = MockTabularExtractor()
        features = extractor.extract(event, observations, decision_time)

        # Should only count the first 2 observations
        assert features.features["obs_count"] == 2.0


# Tests for Prediction


class TestPrediction:
    """Tests for Prediction model."""

    def test_creation(self):
        """Test creating a prediction."""
        pred = Prediction(
            predictions={"home_win": 0.6, "away_win": 0.4},
            confidence=0.6,
            metadata={"model": "test"},
        )

        assert pred.predictions["home_win"] == 0.6
        assert pred.confidence == 0.6
        assert pred.metadata["model"] == "test"

    def test_get_most_likely_outcome(self):
        """Test getting most likely outcome."""
        pred = Prediction(
            predictions={"a": 0.2, "b": 0.7, "c": 0.1},
            confidence=0.7,
        )

        outcome, prob = pred.get_most_likely_outcome()
        assert outcome == "b"
        assert prob == 0.7

    def test_get_prediction_for(self):
        """Test getting prediction for specific outcome."""
        pred = Prediction(
            predictions={"home": 0.6, "away": 0.4},
            confidence=0.6,
        )

        assert pred.get_prediction_for("home") == 0.6
        assert pred.get_prediction_for("away") == 0.4
        assert pred.get_prediction_for("draw") is None

    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            Prediction(predictions={"a": 1.0}, confidence=1.5)

        with pytest.raises(ValueError):
            Prediction(predictions={"a": 1.0}, confidence=-0.1)


# Tests for ModelPredictor


class TestModelPredictor:
    """Tests for ModelPredictor base class."""

    def test_predict(self):
        """Test making a prediction."""
        model = MockModelPredictor()
        features = TabularFeatureSet({"obs_count": 5.0})

        prediction = model.predict(features)

        assert isinstance(prediction, Prediction)
        assert "positive" in prediction.predictions
        assert "negative" in prediction.predictions
        assert 0.0 <= prediction.confidence <= 1.0

    def test_get_model_type(self):
        """Test getting model type."""
        model = MockModelPredictor()
        assert "Mock" in model.get_model_type()

    def test_get_required_feature_type(self):
        """Test getting required feature type."""
        model = MockModelPredictor()
        assert model.get_required_feature_type() == "tabular"
