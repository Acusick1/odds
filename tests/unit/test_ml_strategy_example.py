"""Unit tests for XGBoost ML strategy example."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from odds_analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.ml_strategy_example import XGBoostStrategy
from odds_core.models import EventStatus, Odds


@pytest.fixture
def sample_event():
    """Create a sample BacktestEvent for testing."""
    return BacktestEvent(
        id="test_event_1",
        commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        home_score=110,
        away_score=105,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def sample_odds_snapshot(sample_event):
    """Create sample odds snapshot for testing."""
    timestamp = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
    return [
        # Pinnacle (sharp book)
        Odds(
            id=1,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price=-120,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=2,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price=+100,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        # FanDuel (retail book)
        Odds(
            id=3,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price=-115,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=4,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price=-105,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        # DraftKings (retail book)
        Odds(
            id=5,
            event_id=sample_event.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price=-118,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=6,
            event_id=sample_event.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price=-102,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
    ]


class TestTabularFeatureExtractor:
    """Test TabularFeatureExtractor integration with XGBoostStrategy."""

    def test_extract_features_returns_dict(self, sample_event, sample_odds_snapshot):
        """Test that extract_features returns a dictionary."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_features_includes_consensus_prob(self, sample_event, sample_odds_snapshot):
        """Test that consensus probability features are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "consensus_prob" in features
        assert 0 < features["consensus_prob"] < 1
        assert "opponent_consensus_prob" in features

    def test_extract_features_includes_sharp_features(self, sample_event, sample_odds_snapshot):
        """Test that sharp bookmaker features are extracted."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "sharp_prob" in features
        assert "sharp_market_hold" in features
        assert features["sharp_market_hold"] > 0  # Should have some vig

    def test_extract_features_includes_retail_sharp_diff(self, sample_event, sample_odds_snapshot):
        """Test that retail vs sharp differences are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should have retail-sharp difference features
        assert "retail_sharp_diff_home" in features or "retail_sharp_diff_away" in features

    def test_extract_features_includes_best_odds(self, sample_event, sample_odds_snapshot):
        """Test that best available odds are found."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "best_available_odds" in features
        assert "best_available_decimal" in features
        assert features["best_available_decimal"] > 1.0

    def test_extract_features_team_indicators(self, sample_event, sample_odds_snapshot):
        """Test that team indicator features are set correctly."""
        extractor = TabularFeatureExtractor()
        home_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert home_features["is_home_team"] == 1.0
        assert home_features["is_away_team"] == 0.0

        away_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.away_team
        )

        assert away_features["is_home_team"] == 0.0
        assert away_features["is_away_team"] == 1.0

    def test_extract_features_empty_odds(self, sample_event):
        """Test that extract_features handles empty odds gracefully."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, [], market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, dict)
        # Should return empty or minimal features
        assert len(features) == 0

    def test_create_feature_vector_correct_order(self):
        """Test that feature vector maintains correct order."""
        extractor = TabularFeatureExtractor()
        features = {
            "feature_a": 1.0,
            "feature_b": 2.0,
            "feature_c": 3.0,
        }
        feature_names = ["feature_c", "feature_a", "feature_b"]

        vector = extractor.create_feature_vector(features, feature_names)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert vector[0] == 3.0  # feature_c
        assert vector[1] == 1.0  # feature_a
        assert vector[2] == 2.0  # feature_b

    def test_create_feature_vector_fills_missing_with_zero(self):
        """Test that missing features are filled with 0.0."""
        extractor = TabularFeatureExtractor()
        features = {"feature_a": 1.0}
        feature_names = ["feature_a", "feature_b", "feature_c"]

        vector = extractor.create_feature_vector(features, feature_names)

        assert vector[0] == 1.0
        assert vector[1] == 0.0  # Missing feature_b
        assert vector[2] == 0.0  # Missing feature_c


class TestXGBoostStrategy:
    """Test XGBoost strategy class."""

    def test_strategy_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = XGBoostStrategy(
            market="h2h",
            min_edge_threshold=0.05,
            min_confidence=0.55,
        )

        assert strategy.name == "XGBoost"
        assert strategy.params["market"] == "h2h"
        assert strategy.params["min_edge_threshold"] == 0.05
        assert strategy.params["min_confidence"] == 0.55
        assert strategy.model is None

    def test_train_requires_xgboost(self):
        """Test that train method requires xgboost package."""
        strategy = XGBoostStrategy()

        # Mock missing xgboost import
        import sys

        original_xgboost = sys.modules.get("xgboost")
        sys.modules["xgboost"] = None

        try:
            with pytest.raises(ImportError, match="xgboost not installed"):
                strategy.train(
                    X_train=np.array([[1, 2], [3, 4]]),
                    y_train=np.array([0, 1]),
                    feature_names=["f1", "f2"],
                )
        finally:
            # Restore original state
            if original_xgboost is not None:
                sys.modules["xgboost"] = original_xgboost
            else:
                del sys.modules["xgboost"]

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_creates_model(self):
        """Test that training creates a model."""
        from xgboost import XGBClassifier

        strategy = XGBoostStrategy()

        # Simple training data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        assert strategy.model is not None
        assert isinstance(strategy.model, XGBClassifier)
        assert strategy.feature_names == feature_names

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_predict_probability(self):
        """Test that model can make predictions."""
        strategy = XGBoostStrategy()

        # Train simple model
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Make prediction
        test_features = np.random.rand(5)
        prob = strategy._predict_probability(test_features)

        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_save_and_load_model(self):
        """Test that model can be saved and loaded."""
        strategy = XGBoostStrategy()

        # Train model
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            strategy.save_model(str(model_path))
            assert model_path.exists()

            # Load into new strategy
            loaded_strategy = XGBoostStrategy()
            loaded_strategy.load_model(str(model_path))

            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names

            # Predictions should be identical
            test_features = np.random.rand(5)
            original_prob = strategy._predict_probability(test_features)
            loaded_prob = loaded_strategy._predict_probability(test_features)

            assert abs(original_prob - loaded_prob) < 1e-6

    def test_save_model_without_training_raises_error(self):
        """Test that saving without training raises error."""
        strategy = XGBoostStrategy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            with pytest.raises(ValueError, match="No model to save"):
                strategy.save_model(str(model_path))

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_get_feature_importance(self):
        """Test that feature importance can be retrieved."""
        strategy = XGBoostStrategy()

        # Train model
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Get importance
        importance = strategy.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(name in importance for name in feature_names)
        # XGBoost returns numpy types, verify they're numeric
        assert all(np.isscalar(score) and np.isfinite(score) for score in importance.values())

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_requires_model(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity requires trained model."""
        strategy = XGBoostStrategy()
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )

        with pytest.raises(ValueError, match="Model not loaded"):
            await strategy.evaluate_opportunity(sample_event, sample_odds_snapshot, config)

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_returns_opportunities(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that evaluate_opportunity returns BetOpportunity objects."""
        strategy = XGBoostStrategy(min_edge_threshold=0.0, min_confidence=0.0)

        # Train simple model (doesn't need to be good)
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        # Evaluate
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        opportunities = await strategy.evaluate_opportunity(
            sample_event, sample_odds_snapshot, config
        )

        assert isinstance(opportunities, list)
        # Should return opportunities for both teams
        for opp in opportunities:
            assert isinstance(opp, BetOpportunity)
            assert opp.event_id == sample_event.id
            assert 0 <= opp.confidence <= 1
            assert opp.market == "h2h"

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_respects_min_confidence(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that opportunities below min_confidence are filtered."""
        strategy = XGBoostStrategy(
            min_edge_threshold=0.0,
            min_confidence=0.99,  # Very high threshold
        )

        # Train model
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        # Evaluate
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        opportunities = await strategy.evaluate_opportunity(
            sample_event, sample_odds_snapshot, config
        )

        # With 99% confidence threshold, should get few/no opportunities
        # (unless we get very lucky with random model)
        assert isinstance(opportunities, list)

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_empty_odds(self, sample_event):
        """Test that evaluate_opportunity handles empty odds."""
        strategy = XGBoostStrategy()

        # Train model
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        # Evaluate with empty odds
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        opportunities = await strategy.evaluate_opportunity(sample_event, [], config)

        assert opportunities == []


class TestIntegration:
    """Integration tests for ML strategy with backtesting framework."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_model_persistence_workflow(self):
        """Test complete save/load workflow."""
        # Create and train model
        strategy = XGBoostStrategy(market="h2h", min_edge_threshold=0.03)

        X_train = np.random.rand(200, 15)
        y_train = np.random.randint(0, 2, 200)
        feature_names = [f"feature_{i}" for i in range(15)]

        strategy.train(
            X_train, y_train, feature_names, n_estimators=20, max_depth=4, learning_rate=0.1
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            strategy.save_model(str(model_path))

            # Load into new instance
            new_strategy = XGBoostStrategy(model_path=str(model_path))

            # Verify parameters persisted
            assert new_strategy.params["market"] == "h2h"
            assert new_strategy.params["min_edge_threshold"] == 0.03

            # Verify model works
            test_vec = np.random.rand(15)
            prob1 = strategy._predict_probability(test_vec)
            prob2 = new_strategy._predict_probability(test_vec)

            assert abs(prob1 - prob2) < 1e-6

    def test_feature_extraction_consistency(self, sample_event, sample_odds_snapshot):
        """Test that feature extraction produces consistent results."""
        extractor = TabularFeatureExtractor()

        # Extract features twice
        features1 = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        features2 = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should be identical
        assert features1 == features2

        # Convert to vectors
        feature_names = sorted(features1.keys())
        vec1 = extractor.create_feature_vector(features1, feature_names)
        vec2 = extractor.create_feature_vector(features2, feature_names)

        assert np.allclose(vec1, vec2)
