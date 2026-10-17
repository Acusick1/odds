"""Unit tests for XGBoost Line Movement Predictor strategy."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from odds_analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity
from odds_analytics.feature_extraction import TabularFeatures
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
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


class TestXGBoostLineMovementStrategy:
    """Test XGBoost Line Movement strategy class."""

    def test_strategy_initialization(self):
        """Test that strategy initializes correctly."""
        strategy = XGBoostLineMovementStrategy(
            market="h2h",
            min_predicted_movement=0.03,
            movement_confidence_scale=4.0,
            base_confidence=0.55,
        )

        assert strategy.name == "XGBoostLineMovement"
        assert strategy.params["market"] == "h2h"
        assert strategy.params["min_predicted_movement"] == 0.03
        assert strategy.params["movement_confidence_scale"] == 4.0
        assert strategy.params["base_confidence"] == 0.55
        assert strategy.model is None

    def test_strategy_default_parameters(self):
        """Test that default parameters are set correctly."""
        strategy = XGBoostLineMovementStrategy()

        assert strategy.params["market"] == "h2h"
        assert strategy.params["min_predicted_movement"] == 0.02
        assert strategy.params["movement_confidence_scale"] == 5.0
        assert strategy.params["base_confidence"] == 0.52
        assert len(strategy.params["bookmakers"]) == 8

    def test_train_requires_xgboost(self):
        """Test that train method requires xgboost package."""
        strategy = XGBoostLineMovementStrategy()

        # Mock missing xgboost import
        import sys

        original_xgboost = sys.modules.get("xgboost")
        sys.modules["xgboost"] = None

        try:
            with pytest.raises(ImportError, match="xgboost not installed"):
                strategy.train(
                    X_train=np.array([[1, 2], [3, 4]]),
                    y_train=np.array([0.1, 0.2]),
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
    def test_train_creates_regressor_model(self):
        """Test that training creates a regressor model."""
        from xgboost import XGBRegressor

        strategy = XGBoostLineMovementStrategy()

        # Regression training data (continuous targets)
        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)  # Line movement deltas
        feature_names = [f"feature_{i}" for i in range(5)]

        history = strategy.train(X_train, y_train, feature_names, n_estimators=10)

        assert strategy.model is not None
        assert isinstance(strategy.model, XGBRegressor)
        assert strategy.feature_names == feature_names
        assert "train_mse" in history
        assert "train_mae" in history
        assert "train_r2" in history

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_with_validation_data(self):
        """Test training with validation data."""
        strategy = XGBoostLineMovementStrategy()

        X_train = np.random.rand(80, 5)
        y_train = np.random.uniform(-0.1, 0.1, 80)
        X_val = np.random.rand(20, 5)
        y_val = np.random.uniform(-0.1, 0.1, 20)
        feature_names = [f"feature_{i}" for i in range(5)]

        history = strategy.train(
            X_train, y_train, feature_names, X_val=X_val, y_val=y_val, n_estimators=10
        )

        assert "val_mse" in history
        assert "val_mae" in history
        assert "val_r2" in history

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_predict_movement(self):
        """Test that model can predict line movement."""
        strategy = XGBoostLineMovementStrategy()

        # Train simple model
        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Make prediction
        test_features = np.random.rand(5)
        movement = strategy._predict_movement(test_features)

        assert isinstance(movement, float)
        # Movement should be in reasonable range for probability deltas
        assert -1 <= movement <= 1

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_save_and_load_model(self):
        """Test that model can be saved and loaded."""
        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.03, movement_confidence_scale=6.0
        )

        # Train model
        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            strategy.save_model(str(model_path))
            assert model_path.exists()

            # Load into new strategy
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names
            # Params should be loaded
            assert loaded_strategy.params["min_predicted_movement"] == 0.03
            assert loaded_strategy.params["movement_confidence_scale"] == 6.0

            # Predictions should be identical
            test_features = np.random.rand(5)
            original_movement = strategy._predict_movement(test_features)
            loaded_movement = loaded_strategy._predict_movement(test_features)

            assert abs(original_movement - loaded_movement) < 1e-6

    def test_save_model_without_training_raises_error(self):
        """Test that saving without training raises error."""
        strategy = XGBoostLineMovementStrategy()

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
        strategy = XGBoostLineMovementStrategy()

        # Train model
        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Get importance
        importance = strategy.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(name in importance for name in feature_names)
        assert all(np.isscalar(score) and np.isfinite(score) for score in importance.values())

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_requires_model(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity requires trained model."""
        strategy = XGBoostLineMovementStrategy()
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
        # Very low threshold to ensure we get opportunities
        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.0,
            base_confidence=0.52,
        )

        # Train with correct feature count
        num_features = len(TabularFeatures.get_feature_names())
        X_train = np.random.rand(100, num_features)
        # Generate targets that will give us positive predictions
        y_train = np.random.uniform(0.0, 0.15, 100)
        feature_names = TabularFeatures.get_feature_names()

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
        # With very low threshold, should get opportunities
        for opp in opportunities:
            assert isinstance(opp, BetOpportunity)
            assert opp.event_id == sample_event.id
            assert 0.5 <= opp.confidence <= 0.95  # Should be in clamped range
            assert opp.market == "h2h"
            assert "Predicted movement" in opp.rationale

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_respects_min_movement(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that opportunities below min_predicted_movement are filtered."""
        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.99,  # Very high threshold
        )

        # Train model that predicts small movements
        num_features = len(TabularFeatures.get_feature_names())
        X_train = np.random.rand(100, num_features)
        y_train = np.random.uniform(-0.01, 0.01, 100)  # Small targets
        feature_names = TabularFeatures.get_feature_names()

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

        # Should get no opportunities with very high threshold
        assert opportunities == []

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_confidence_clamping(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that confidence is properly clamped to 0.5-0.95 range."""
        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.0,
            movement_confidence_scale=100.0,  # High scale to test clamping
            base_confidence=0.52,
        )

        # Train with high positive targets
        num_features = len(TabularFeatures.get_feature_names())
        X_train = np.random.rand(100, num_features)
        y_train = np.random.uniform(0.1, 0.2, 100)  # High movements
        feature_names = TabularFeatures.get_feature_names()

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

        # All confidences should be clamped
        for opp in opportunities:
            assert 0.5 <= opp.confidence <= 0.95

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_empty_odds(self, sample_event):
        """Test that evaluate_opportunity handles empty odds."""
        strategy = XGBoostLineMovementStrategy()

        # Train model
        num_features = len(TabularFeatures.get_feature_names())
        X_train = np.random.rand(100, num_features)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = TabularFeatures.get_feature_names()

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        # Evaluate with empty odds
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        opportunities = await strategy.evaluate_opportunity(sample_event, [], config)

        assert opportunities == []

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    async def test_evaluate_opportunity_handles_nan_features(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that NaN features are handled gracefully."""
        strategy = XGBoostLineMovementStrategy(min_predicted_movement=0.0)

        # Train model
        num_features = len(TabularFeatures.get_feature_names())
        X_train = np.random.rand(100, num_features)
        y_train = np.random.uniform(0.0, 0.1, 100)
        feature_names = TabularFeatures.get_feature_names()

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        # Create sparse odds that will result in NaN features
        sparse_odds = [sample_odds_snapshot[0], sample_odds_snapshot[1]]  # Only one bookmaker

        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )

        # Should not raise error, even with NaN features
        opportunities = await strategy.evaluate_opportunity(sample_event, sparse_odds, config)
        assert isinstance(opportunities, list)


class TestModelPersistence:
    """Test model persistence workflow."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_complete_save_load_workflow(self):
        """Test complete save/load workflow with all parameters."""
        # Create and train model
        strategy = XGBoostLineMovementStrategy(
            market="h2h",
            min_predicted_movement=0.03,
            movement_confidence_scale=4.0,
            base_confidence=0.54,
        )

        X_train = np.random.rand(200, 15)
        y_train = np.random.uniform(-0.1, 0.1, 200)
        feature_names = [f"feature_{i}" for i in range(15)]

        strategy.train(
            X_train, y_train, feature_names, n_estimators=20, max_depth=4, learning_rate=0.1
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            strategy.save_model(str(model_path))

            # Load into new instance using model_path in constructor
            new_strategy = XGBoostLineMovementStrategy(model_path=str(model_path))

            # Verify parameters persisted
            assert new_strategy.params["market"] == "h2h"
            assert new_strategy.params["min_predicted_movement"] == 0.03
            assert new_strategy.params["movement_confidence_scale"] == 4.0
            assert new_strategy.params["base_confidence"] == 0.54

            # Verify model works
            test_vec = np.random.rand(15)
            movement1 = strategy._predict_movement(test_vec)
            movement2 = new_strategy._predict_movement(test_vec)

            assert abs(movement1 - movement2) < 1e-6

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_model_creates_directory_if_not_exists(self):
        """Test that save_model creates parent directories."""
        strategy = XGBoostLineMovementStrategy()

        X_train = np.random.rand(50, 5)
        y_train = np.random.uniform(-0.1, 0.1, 50)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with non-existent subdirectory
            model_path = Path(tmpdir) / "subdir" / "model.pkl"

            # Should create subdir automatically
            strategy.save_model(str(model_path))
            assert model_path.exists()


class TestModelPersistenceWithConfig:
    """Test model persistence with YAML config files."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_save_model_creates_yaml_config(self):
        """Test that save_model creates both model and YAML config files."""
        import yaml

        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.03, movement_confidence_scale=6.0
        )

        # Train model
        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            strategy.save_model(str(model_path))

            # Check both files exist
            assert model_path.exists()
            config_path = model_path.with_suffix(".yaml")
            assert config_path.exists()

            # Verify config content
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            assert config_data["model_type"] == "XGBoostLineMovement"
            assert "saved_at" in config_data
            assert config_data["params"]["min_predicted_movement"] == 0.03
            assert config_data["params"]["movement_confidence_scale"] == 6.0
            assert config_data["feature_names"] == feature_names
            assert config_data["n_features"] == 5

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_load_model_with_config(self):
        """Test that load_model logs config information when available."""
        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.04, movement_confidence_scale=7.0
        )

        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            strategy.save_model(str(model_path))

            # Load into new strategy
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            # Verify params were loaded correctly
            assert loaded_strategy.params["min_predicted_movement"] == 0.04
            assert loaded_strategy.params["movement_confidence_scale"] == 7.0
            assert loaded_strategy.feature_names == feature_names

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_load_model_backward_compatibility_pickle(self):
        """Test that load_model can still load old pickle format models."""
        import pickle

        strategy = XGBoostLineMovementStrategy(
            min_predicted_movement=0.05, movement_confidence_scale=8.0
        )

        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save using old pickle format directly
            model_path = Path(tmpdir) / "old_model.pkl"
            model_data = {
                "model": strategy.model,
                "feature_names": strategy.feature_names,
                "params": strategy.params,
            }
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            # Load into new strategy
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            # Verify model loaded correctly
            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names
            assert loaded_strategy.params["min_predicted_movement"] == 0.05

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_load_model_without_config_file(self):
        """Test that load_model works when config file doesn't exist."""
        import joblib

        strategy = XGBoostLineMovementStrategy()

        X_train = np.random.rand(100, 5)
        y_train = np.random.uniform(-0.1, 0.1, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        strategy.train(X_train, y_train, feature_names, n_estimators=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save only model file, no config
            model_path = Path(tmpdir) / "model_only.joblib"
            model_data = {
                "model": strategy.model,
                "feature_names": strategy.feature_names,
                "params": strategy.params,
            }
            joblib.dump(model_data, model_path)

            # Load - should work without config file
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names
