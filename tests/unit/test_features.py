"""Unit tests for feature engineering modules."""

from datetime import datetime, timedelta

from analytics.betting import BettingEvent, OddsObservation
from analytics.features import (
    SequentialFeatureExtractor,
    TabularFeatureExtractor,
    compute_consensus_odds,
    compute_market_hold,
    compute_timing_features,
)
from analytics.features.validators import validate_all, validate_no_inf, validate_no_nan


# Helper feature computer for testing
def simple_feature_computer(problem, observations, decision_time):
    """Simple feature: count of observations."""
    return {"obs_count": float(len(observations))}


def odds_feature_computer(problem, observations, decision_time):
    """Feature: average odds if available."""
    if not observations:
        return {"avg_odds": 0.0}

    odds_obs = [obs for obs in observations if isinstance(obs, OddsObservation)]
    if not odds_obs:
        return {"avg_odds": 0.0}

    avg = sum(obs.odds for obs in odds_obs) / len(odds_obs)
    return {"avg_odds": float(avg)}


class TestTabularFeatureExtractor:
    """Tests for TabularFeatureExtractor."""

    def test_creation(self):
        """Test creating a tabular extractor."""
        extractor = TabularFeatureExtractor([simple_feature_computer])
        assert len(extractor.feature_computers) == 1

    def test_extract_basic(self):
        """Test basic feature extraction."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),
        )

        observations = [
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 18, 0),
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-110,
            ),
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 18, 30),
                bookmaker="draftkings",
                market="h2h",
                outcome="Lakers",
                odds=-105,
            ),
        ]

        decision_time = datetime(2024, 1, 15, 18, 45)

        extractor = TabularFeatureExtractor([simple_feature_computer])
        features = extractor.extract(event, observations, decision_time)

        assert features.features["obs_count"] == 2.0
        assert features.get_shape() == (1,)

    def test_prevents_lookahead_bias(self):
        """Test that extractor filters future observations."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),
        )

        observations = [
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 17, 0),  # Before
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-110,
            ),
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 19, 0),  # After
                bookmaker="draftkings",
                market="h2h",
                outcome="Lakers",
                odds=-105,
            ),
        ]

        decision_time = datetime(2024, 1, 15, 18, 0)

        extractor = TabularFeatureExtractor([simple_feature_computer])
        features = extractor.extract(event, observations, decision_time)

        # Should only count the first observation
        assert features.features["obs_count"] == 1.0

    def test_multiple_feature_computers(self):
        """Test using multiple feature computers."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),
        )

        observations = [
            OddsObservation(
                "game1", datetime(2024, 1, 15, 18, 0), "fanduel", "h2h", "Lakers", -110
            ),
        ]

        decision_time = datetime(2024, 1, 15, 18, 30)

        extractor = TabularFeatureExtractor([simple_feature_computer, odds_feature_computer])
        features = extractor.extract(event, observations, decision_time)

        assert "obs_count" in features.features
        assert "avg_odds" in features.features
        assert features.features["obs_count"] == 1.0
        assert features.features["avg_odds"] == -110.0


class TestSequentialFeatureExtractor:
    """Tests for SequentialFeatureExtractor."""

    def test_creation(self):
        """Test creating a sequential extractor."""
        extractor = SequentialFeatureExtractor(
            [simple_feature_computer],
            sequence_length=5,
            step_size=timedelta(hours=1),
        )

        assert extractor.sequence_length == 5
        assert extractor.step_size == timedelta(hours=1)

    def test_extract_basic(self):
        """Test basic sequential feature extraction."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),
        )

        observations = [
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 15, 0),
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-110,
            ),
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 17, 0),
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-105,
            ),
        ]

        decision_time = datetime(2024, 1, 15, 18, 0)

        extractor = SequentialFeatureExtractor(
            [simple_feature_computer],
            sequence_length=3,
            step_size=timedelta(hours=1),
        )

        features = extractor.extract(event, observations, decision_time)

        # Should have 3 time steps (18:00 - 2 hours to 18:00)
        assert features.get_shape() == (3, 1)  # 3 steps, 1 feature
        assert len(features.timestamps) == 3

        # Windows are: 16:00, 17:00, 18:00
        # At 16:00: only obs at 15:00 (1 obs)
        # At 17:00: obs at 15:00 and 17:00 (2 obs)
        # At 18:00: both obs (2 obs)
        assert features.sequences[0, 0] == 1.0  # First window (16:00): 1 observation
        assert features.sequences[-1, 0] == 2.0  # Most recent (18:00): 2 observations

    def test_time_window_creation(self):
        """Test that time windows are created correctly."""
        decision_time = datetime(2024, 1, 15, 18, 0)

        extractor = SequentialFeatureExtractor(
            [simple_feature_computer],
            sequence_length=5,
            step_size=timedelta(hours=1),
        )

        windows = extractor._create_time_windows(decision_time)

        assert len(windows) == 5
        assert windows[0] == datetime(2024, 1, 15, 14, 0)  # 4 hours before
        assert windows[-1] == datetime(2024, 1, 15, 18, 0)  # decision time

    def test_prevents_lookahead_per_window(self):
        """Test that each window only uses observations before that window time."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),
        )

        # Observations at different times
        observations = [
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 15, 0),
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-110,
            ),
            OddsObservation(
                event_id="game1",
                observation_time=datetime(2024, 1, 15, 17, 0),
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-105,
            ),
        ]

        decision_time = datetime(2024, 1, 15, 18, 0)

        extractor = SequentialFeatureExtractor(
            [simple_feature_computer],
            sequence_length=4,
            step_size=timedelta(hours=1),
        )

        features = extractor.extract(event, observations, decision_time)

        # Window at 15:00 should have 0 or 1 observations
        # Window at 17:00 should have 1 observation
        # Window at 18:00 should have 2 observations
        assert features.sequences[0, 0] <= 1.0  # Earliest window
        assert features.sequences[-1, 0] == 2.0  # Latest window


class TestBettingFeatures:
    """Tests for betting-specific feature computers."""

    def test_compute_timing_features(self):
        """Test timing feature computation."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime(2024, 1, 15, 19, 0),  # Monday at 7pm
        )

        decision_time = datetime(2024, 1, 15, 17, 0)  # 2 hours before

        features = compute_timing_features(event, [], decision_time)

        assert features["hours_to_game"] == 2.0
        assert features["day_of_week"] == 0.0  # Monday

    def test_compute_consensus_odds(self):
        """Test consensus odds computation."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
        )

        observations = [
            OddsObservation("game1", datetime.now(), "fanduel", "h2h", "Lakers", -110),  # ~52.4%
            OddsObservation("game1", datetime.now(), "draftkings", "h2h", "Lakers", -120),  # ~54.5%
        ]

        features = compute_consensus_odds(event, observations, datetime.now())

        # Should average the implied probabilities
        assert "consensus_home_prob" in features
        assert 0.52 < features["consensus_home_prob"] < 0.55

    def test_compute_market_hold(self):
        """Test market hold computation."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
        )

        observations = [
            OddsObservation("game1", datetime.now(), "fanduel", "h2h", "Lakers", -110),
            OddsObservation("game1", datetime.now(), "fanduel", "h2h", "Celtics", -110),
        ]

        features = compute_market_hold(event, observations, datetime.now())

        # Standard -110/-110 market has ~4.5% hold
        assert "market_hold_avg" in features
        assert 0.04 < features["market_hold_avg"] < 0.06


class TestFeatureValidators:
    """Tests for feature validation utilities."""

    def test_validate_no_inf(self):
        """Test infinite value detection."""
        from analytics.core import TabularFeatureSet

        # Valid features
        valid_features = TabularFeatureSet({"a": 1.0, "b": 2.0})
        is_valid, problems = validate_no_inf(valid_features)
        assert is_valid is True
        assert len(problems) == 0

        # Invalid features with inf
        invalid_features = TabularFeatureSet({"a": float("inf"), "b": 2.0})
        is_valid, problems = validate_no_inf(invalid_features)
        assert is_valid is False
        assert len(problems) > 0

    def test_validate_no_nan(self):
        """Test NaN value detection."""
        from analytics.core import TabularFeatureSet

        # Valid features
        valid_features = TabularFeatureSet({"a": 1.0, "b": 2.0})
        is_valid, problems = validate_no_nan(valid_features)
        assert is_valid is True

        # Invalid features with NaN
        invalid_features = TabularFeatureSet({"a": float("nan"), "b": 2.0})
        is_valid, problems = validate_no_nan(invalid_features)
        assert is_valid is False

    def test_validate_all(self):
        """Test combined validation."""
        from analytics.core import TabularFeatureSet

        # Valid features
        valid_features = TabularFeatureSet({"a": 1.0, "b": 2.0})
        is_valid, problems = validate_all(valid_features)
        assert is_valid is True

        # Invalid features
        invalid_features = TabularFeatureSet({"a": float("nan"), "b": float("inf")})
        is_valid, problems = validate_all(invalid_features)
        assert is_valid is False
        assert len(problems) == 2
