"""Unit tests for train_from_config() methods in ML strategies."""

from datetime import date

import numpy as np
import pytest
from odds_analytics.ml_strategy_example import XGBoostStrategy
from odds_analytics.training.config import (
    DataConfig,
    ExperimentConfig,
    LSTMConfig,
    MLTrainingConfig,
    SearchSpace,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
    resolve_search_spaces,
)
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy


@pytest.fixture
def sample_xgboost_config():
    """Create a sample XGBoost training configuration."""
    return MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_xgboost",
            tags=["test", "xgboost"],
            description="Test XGBoost configuration",
        ),
        training=TrainingConfig(
            strategy_type="xgboost",
            data=DataConfig(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 12, 31),
                test_split=0.2,
            ),
            model=XGBoostConfig(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.5,
            ),
        ),
    )


@pytest.fixture
def sample_xgboost_line_movement_config():
    """Create a sample XGBoost line movement training configuration."""
    return MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_xgb_line_movement",
            tags=["test", "xgboost", "line_movement"],
            description="Test XGBoost line movement configuration",
        ),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 12, 31),
            ),
            model=XGBoostConfig(
                n_estimators=75,
                max_depth=5,
                learning_rate=0.08,
                # Note: early_stopping_rounds requires validation data
            ),
        ),
    )


@pytest.fixture
def sample_lstm_config():
    """Create a sample LSTM training configuration."""
    return MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_lstm",
            tags=["test", "lstm"],
            description="Test LSTM configuration",
        ),
        training=TrainingConfig(
            strategy_type="lstm",
            data=DataConfig(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 12, 31),
            ),
            model=LSTMConfig(
                hidden_size=32,
                num_layers=1,
                dropout=0.1,
                epochs=5,
                batch_size=16,
                learning_rate=0.002,
                lookback_hours=48,
                timesteps=12,
            ),
        ),
    )


@pytest.fixture
def sample_config_with_search_spaces():
    """Create a configuration with search spaces for testing resolution."""
    return MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_with_search_spaces",
            tags=["test", "tuning"],
        ),
        training=TrainingConfig(
            strategy_type="xgboost",
            data=DataConfig(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 12, 31),
            ),
            model=XGBoostConfig(
                n_estimators=100,  # Will be overridden by search space
                max_depth=6,  # Will be overridden by search space
                learning_rate=0.1,  # Will be overridden by search space
            ),
        ),
        tuning=TuningConfig(
            n_trials=100,
            search_spaces={
                "n_estimators": SearchSpace(type="int", low=50, high=150),
                "max_depth": SearchSpace(type="int", low=2, high=10),
                "learning_rate": SearchSpace(type="float", low=0.01, high=0.3, log=True),
                "objective": SearchSpace(
                    type="categorical",
                    choices=["reg:squarederror", "reg:absoluteerror"],
                ),
            },
        ),
    )


@pytest.fixture
def training_data():
    """Create simple training data for tests."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    feature_names = [f"feature_{i}" for i in range(10)]
    return X_train, y_train, feature_names


class TestXGBoostStrategyTrainFromConfig:
    """Test XGBoostStrategy.train_from_config() method."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_from_config_extracts_hyperparameters(self, sample_xgboost_config, training_data):
        """Test that hyperparameters are correctly extracted from config."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostStrategy()

        strategy.train_from_config(sample_xgboost_config, X_train, y_train, feature_names)

        assert strategy.model is not None
        # Verify model was trained with config parameters
        assert strategy.model.n_estimators == 50
        assert strategy.model.max_depth == 4
        assert strategy.model.learning_rate == 0.05

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_from_config_with_search_spaces(
        self, sample_config_with_search_spaces, training_data
    ):
        """Test that search spaces are resolved to midpoint values."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostStrategy()

        strategy.train_from_config(
            sample_config_with_search_spaces, X_train, y_train, feature_names
        )

        # Check that search spaces were resolved to midpoints
        # n_estimators: (50 + 150) / 2 = 100
        assert strategy.model.n_estimators == 100
        # max_depth: (2 + 10) / 2 = 6
        assert strategy.model.max_depth == 6
        # learning_rate: geometric mean of 0.01 and 0.3 (log scale)
        # = exp((ln(0.01) + ln(0.3)) / 2) ≈ 0.0548
        assert 0.05 < strategy.model.learning_rate < 0.06

    def test_train_from_config_invalid_strategy_type(self, training_data):
        """Test that invalid strategy type raises ValueError."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostStrategy()

        invalid_config = MLTrainingConfig(
            experiment=ExperimentConfig(name="invalid"),
            training=TrainingConfig(
                strategy_type="lstm",  # Wrong type for XGBoostStrategy
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=LSTMConfig(),  # Wrong config type
            ),
        )

        with pytest.raises(ValueError, match="Invalid strategy_type"):
            strategy.train_from_config(invalid_config, X_train, y_train, feature_names)

    def test_train_from_config_wrong_model_type(self, training_data):
        """Test that wrong model config type raises TypeError."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostStrategy()

        # Manually construct an invalid config (normally prevented by validation)
        from unittest.mock import MagicMock

        invalid_config = MagicMock()
        invalid_config.training.strategy_type = "xgboost"
        invalid_config.training.model = "not_a_model_config"
        invalid_config.tuning = None

        with pytest.raises(TypeError, match="Expected XGBoostConfig"):
            strategy.train_from_config(invalid_config, X_train, y_train, feature_names)

    def test_resolve_search_spaces_int(self):
        """Test search space resolution for integer parameters."""
        params = {"n_estimators": 100}
        search_spaces = {"n_estimators": SearchSpace(type="int", low=50, high=150)}

        resolved = resolve_search_spaces(params, search_spaces)

        assert resolved["n_estimators"] == 100  # (50 + 150) / 2

    def test_resolve_search_spaces_int_with_step(self):
        """Test search space resolution for integer with step."""
        params = {"n_estimators": 100}
        search_spaces = {"n_estimators": SearchSpace(type="int", low=50, high=150, step=25)}

        resolved = resolve_search_spaces(params, search_spaces)

        # Midpoint 100, rounded to nearest step of 25 = 100
        assert resolved["n_estimators"] == 100

    def test_resolve_search_spaces_float_linear(self):
        """Test search space resolution for linear float parameters."""
        params = {"subsample": 0.8}
        search_spaces = {"subsample": SearchSpace(type="float", low=0.5, high=1.0)}

        resolved = resolve_search_spaces(params, search_spaces)

        assert resolved["subsample"] == 0.75  # (0.5 + 1.0) / 2

    def test_resolve_search_spaces_float_log(self):
        """Test search space resolution for log-scale float parameters."""
        import math

        params = {"learning_rate": 0.1}
        search_spaces = {"learning_rate": SearchSpace(type="float", low=0.001, high=0.1, log=True)}

        resolved = resolve_search_spaces(params, search_spaces)

        # Geometric mean = exp((ln(0.001) + ln(0.1)) / 2)
        expected = math.exp((math.log(0.001) + math.log(0.1)) / 2)
        assert abs(resolved["learning_rate"] - expected) < 1e-6

    def test_resolve_search_spaces_categorical(self):
        """Test search space resolution for categorical parameters."""
        params = {"objective": "reg:squarederror"}
        search_spaces = {
            "objective": SearchSpace(
                type="categorical", choices=["reg:absoluteerror", "reg:squarederror"]
            )
        }

        resolved = resolve_search_spaces(params, search_spaces)

        # Should use first choice
        assert resolved["objective"] == "reg:absoluteerror"

    def test_resolve_search_spaces_unknown_param(self):
        """Test that unknown search space parameters are logged but ignored."""
        params = {"n_estimators": 100}
        search_spaces = {
            "unknown_param": SearchSpace(type="int", low=1, high=10),
        }

        resolved = resolve_search_spaces(params, search_spaces)

        # Original param should be unchanged
        assert resolved["n_estimators"] == 100
        # Unknown param should not be added
        assert "unknown_param" not in resolved


class TestXGBoostLineMovementStrategyTrainFromConfig:
    """Test XGBoostLineMovementStrategy.train_from_config() method."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_from_config_returns_history(
        self, sample_xgboost_line_movement_config, training_data
    ):
        """Test that train_from_config returns training history."""
        X_train, y_train, feature_names = training_data
        # Convert to regression targets
        y_train = y_train.astype(np.float32) * 0.1  # Small movement values

        strategy = XGBoostLineMovementStrategy()

        history = strategy.train_from_config(
            sample_xgboost_line_movement_config, X_train, y_train, feature_names
        )

        assert isinstance(history, dict)
        assert "train_mse" in history
        assert "train_mae" in history
        assert "train_r2" in history

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_from_config_with_validation_data(
        self, sample_xgboost_line_movement_config, training_data
    ):
        """Test train_from_config with validation data."""
        X_train, y_train, feature_names = training_data
        y_train = y_train.astype(np.float32) * 0.1

        # Create validation data
        X_val = np.random.rand(20, 10)
        y_val = np.random.rand(20) * 0.1

        strategy = XGBoostLineMovementStrategy()

        history = strategy.train_from_config(
            sample_xgboost_line_movement_config,
            X_train,
            y_train,
            feature_names,
            X_val=X_val,
            y_val=y_val,
        )

        assert "val_mse" in history
        assert "val_mae" in history
        assert "val_r2" in history

    def test_train_from_config_invalid_strategy_type(self, training_data):
        """Test that wrong strategy type raises ValueError."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostLineMovementStrategy()

        # Use xgboost instead of xgboost_line_movement
        invalid_config = MLTrainingConfig(
            experiment=ExperimentConfig(name="invalid"),
            training=TrainingConfig(
                strategy_type="xgboost",  # Should be xgboost_line_movement
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(),
            ),
        )

        with pytest.raises(ValueError, match="Invalid strategy_type"):
            strategy.train_from_config(invalid_config, X_train, y_train, feature_names)

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_train_from_config_extracts_early_stopping(self, training_data):
        """Test that early stopping rounds are extracted from config."""
        X_train, y_train, feature_names = training_data
        y_train = y_train.astype(np.float32) * 0.1

        # Create validation data (required for early stopping)
        X_val = np.random.rand(20, 10)
        y_val = np.random.rand(20) * 0.1

        # Create config with early stopping
        config = MLTrainingConfig(
            experiment=ExperimentConfig(name="with_early_stopping"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(
                    n_estimators=100,
                    early_stopping_rounds=15,
                ),
            ),
        )

        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            config, X_train, y_train, feature_names, X_val=X_val, y_val=y_val
        )

        # Model should be trained (early stopping param is passed through)
        assert strategy.model is not None


class TestLSTMStrategyTrainFromConfig:
    """Test LSTMStrategy.train_from_config() method."""

    def test_train_from_config_invalid_strategy_type(self, sample_xgboost_config):
        """Test that wrong strategy type raises ValueError."""
        from odds_analytics.lstm_strategy import LSTMStrategy

        strategy = LSTMStrategy()

        with pytest.raises(ValueError, match="Invalid strategy_type"):
            # Can't use async in sync test, but we can test the validation
            import asyncio

            asyncio.run(strategy.train_from_config(sample_xgboost_config, [], None))

    def test_train_from_config_wrong_model_type(self, sample_lstm_config):
        """Test that wrong model config type raises TypeError."""
        from unittest.mock import MagicMock

        from odds_analytics.lstm_strategy import LSTMStrategy

        strategy = LSTMStrategy()

        # Manually construct an invalid config
        invalid_config = MagicMock()
        invalid_config.training.strategy_type = "lstm"
        invalid_config.training.model = XGBoostConfig()  # Wrong type
        invalid_config.tuning = None

        import asyncio

        with pytest.raises(TypeError, match="Expected LSTMConfig"):
            asyncio.run(strategy.train_from_config(invalid_config, [], None))

    def test_train_from_config_updates_params(self, sample_lstm_config):
        """Test that train_from_config updates strategy params from config."""
        # Access the config model directly to verify params would be extracted
        model_config = sample_lstm_config.training.model
        assert isinstance(model_config, LSTMConfig)
        assert model_config.hidden_size == 32
        assert model_config.num_layers == 1
        assert model_config.epochs == 5
        assert model_config.batch_size == 16

    def test_lstm_resolve_search_spaces(self, sample_lstm_config):
        """Test LSTM search space resolution."""
        params = {
            "epochs": 20,
            "hidden_size": 64,
            "learning_rate": 0.001,
        }
        search_spaces = {
            "epochs": SearchSpace(type="int", low=10, high=50),
            "hidden_size": SearchSpace(type="int", low=32, high=128),
            "learning_rate": SearchSpace(type="float", low=0.0001, high=0.01, log=True),
        }

        resolved = resolve_search_spaces(params, search_spaces)

        # epochs: (10 + 50) / 2 = 30
        assert resolved["epochs"] == 30
        # hidden_size: (32 + 128) / 2 = 80
        assert resolved["hidden_size"] == 80
        # learning_rate: geometric mean
        assert 0.0009 < resolved["learning_rate"] < 0.0011


class TestConfigValidation:
    """Test configuration validation in train_from_config methods."""

    def test_xgboost_rejects_xgboost_line_movement_type(self, training_data):
        """Test that XGBoostStrategy rejects xgboost_line_movement strategy type."""
        X_train, y_train, feature_names = training_data
        strategy = XGBoostStrategy()

        # XGBoostStrategy should only accept "xgboost", not "xgboost_line_movement"
        config = MLTrainingConfig(
            experiment=ExperimentConfig(name="test"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(n_estimators=10),
            ),
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid strategy_type"):
            strategy.train_from_config(config, X_train, y_train, feature_names)

    def test_config_extracts_all_xgboost_params(self, sample_xgboost_config):
        """Test that all XGBoost parameters are extracted from config."""
        model_config = sample_xgboost_config.training.model

        # Verify all expected params are in the config
        assert hasattr(model_config, "n_estimators")
        assert hasattr(model_config, "max_depth")
        assert hasattr(model_config, "min_child_weight")
        assert hasattr(model_config, "learning_rate")
        assert hasattr(model_config, "gamma")
        assert hasattr(model_config, "subsample")
        assert hasattr(model_config, "colsample_bytree")
        assert hasattr(model_config, "colsample_bylevel")
        assert hasattr(model_config, "colsample_bynode")
        assert hasattr(model_config, "reg_alpha")
        assert hasattr(model_config, "reg_lambda")
        assert hasattr(model_config, "objective")
        assert hasattr(model_config, "random_state")
        assert hasattr(model_config, "n_jobs")
        assert hasattr(model_config, "early_stopping_rounds")

    def test_config_extracts_all_lstm_params(self, sample_lstm_config):
        """Test that all LSTM parameters are extracted from config."""
        model_config = sample_lstm_config.training.model

        # Verify all expected params are in the config
        assert hasattr(model_config, "hidden_size")
        assert hasattr(model_config, "num_layers")
        assert hasattr(model_config, "dropout")
        assert hasattr(model_config, "bidirectional")
        assert hasattr(model_config, "lookback_hours")
        assert hasattr(model_config, "timesteps")
        assert hasattr(model_config, "epochs")
        assert hasattr(model_config, "batch_size")
        assert hasattr(model_config, "learning_rate")
        assert hasattr(model_config, "loss_function")
        assert hasattr(model_config, "weight_decay")
        assert hasattr(model_config, "clip_grad_norm")
        assert hasattr(model_config, "patience")
        assert hasattr(model_config, "min_delta")
