"""
Unit tests for ML training configuration schema.

Tests cover:
- All config classes with proper type hints
- YAML/JSON file loading and validation
- Clear validation error messages
- YAML/JSON serialization capability
- Search space validation alongside static values
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    LSTMConfig,
    MLTrainingConfig,
    SearchSpace,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)
from odds_lambda.fetch_tier import FetchTier

# =============================================================================
# ExperimentConfig Tests
# =============================================================================


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_basic_creation(self):
        """Test creating an experiment config with required fields."""
        config = ExperimentConfig(name="test_experiment")
        assert config.name == "test_experiment"
        assert config.tags == []
        assert config.description == ""

    def test_with_all_fields(self):
        """Test creating with all fields specified."""
        config = ExperimentConfig(
            name="full_experiment",
            tags=["xgboost", "line_movement"],
            description="Test experiment for validation",
        )
        assert config.name == "full_experiment"
        assert config.tags == ["xgboost", "line_movement"]
        assert config.description == "Test experiment for validation"

    def test_empty_name_fails(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValueError, match="at least 1 character"):
            ExperimentConfig(name="")


# =============================================================================
# DataConfig Tests
# =============================================================================


class TestDataConfig:
    """Tests for DataConfig."""

    def test_basic_creation(self):
        """Test creating a data config with required fields."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert config.start_date == date(2024, 1, 1)
        assert config.end_date == date(2024, 12, 31)
        assert config.test_split == 0.2
        assert config.validation_split == 0.1
        assert config.random_seed == 42
        assert config.shuffle is True

    def test_custom_splits(self):
        """Test custom train/test splits."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            test_split=0.3,
            validation_split=0.15,
        )
        assert config.test_split == 0.3
        assert config.validation_split == 0.15

    def test_invalid_date_range(self):
        """Test that start_date >= end_date raises error."""
        with pytest.raises(ValueError, match="must be before"):
            DataConfig(
                start_date=date(2024, 12, 31),
                end_date=date(2024, 1, 1),
            )

    def test_same_dates_fails(self):
        """Test that identical dates raise error."""
        with pytest.raises(ValueError, match="must be before"):
            DataConfig(
                start_date=date(2024, 6, 1),
                end_date=date(2024, 6, 1),
            )

    def test_splits_exceed_one(self):
        """Test that splits exceeding 1.0 raise error."""
        with pytest.raises(ValueError, match="must be less than 1.0"):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                test_split=0.6,
                validation_split=0.5,
            )

    def test_test_split_bounds(self):
        """Test test_split boundary validation."""
        # Valid boundary
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            test_split=0.0,
            validation_split=0.0,
        )
        assert config.test_split == 0.0

        # Invalid upper bound
        with pytest.raises(ValueError):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                test_split=1.1,
            )

    def test_kfold_defaults(self):
        """Test K-Fold cross-validation default values."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert config.use_kfold is False
        assert config.n_folds == 5
        assert config.kfold_shuffle is True

    def test_kfold_enabled(self):
        """Test enabling K-Fold cross-validation."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            use_kfold=True,
            n_folds=10,
            kfold_shuffle=False,
        )
        assert config.use_kfold is True
        assert config.n_folds == 10
        assert config.kfold_shuffle is False

    def test_kfold_ignores_validation_split(self):
        """Test that validation_split check is skipped when use_kfold is True."""
        # This should NOT raise even though test_split + validation_split >= 1.0
        # because validation_split is ignored when use_kfold=True
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            test_split=0.2,
            validation_split=0.9,  # Would exceed 1.0 normally
            use_kfold=True,
        )
        assert config.use_kfold is True
        assert config.validation_split == 0.9  # Still stored, just ignored

    def test_kfold_n_folds_bounds(self):
        """Test n_folds boundary validation."""
        # Minimum valid
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            use_kfold=True,
            n_folds=2,
        )
        assert config.n_folds == 2

        # Maximum valid
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            use_kfold=True,
            n_folds=20,
        )
        assert config.n_folds == 20

        # Invalid - too low
        with pytest.raises(ValueError):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                n_folds=1,
            )

        # Invalid - too high
        with pytest.raises(ValueError):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                n_folds=21,
            )

    def test_cv_method_default(self):
        """Test cv_method defaults to 'timeseries'."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert config.cv_method == "timeseries"

    def test_cv_method_kfold(self):
        """Test cv_method can be set to 'kfold'."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            cv_method="kfold",
        )
        assert config.cv_method == "kfold"

    def test_cv_method_timeseries(self):
        """Test cv_method can be explicitly set to 'timeseries'."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            cv_method="timeseries",
        )
        assert config.cv_method == "timeseries"

    def test_cv_method_invalid(self):
        """Test invalid cv_method value raises error."""
        with pytest.raises(ValueError):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                cv_method="invalid",
            )

    def test_cv_method_with_use_kfold(self):
        """Test cv_method works with use_kfold=True."""
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            use_kfold=True,
            cv_method="kfold",
            n_folds=5,
        )
        assert config.use_kfold is True
        assert config.cv_method == "kfold"
        assert config.n_folds == 5

    def test_cv_method_timeseries_with_shuffle(self):
        """Test kfold_shuffle can be set with timeseries (will be ignored at runtime)."""
        # This is allowed at config level - the warning is logged at runtime
        config = DataConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            cv_method="timeseries",
            kfold_shuffle=True,
        )
        assert config.cv_method == "timeseries"
        assert config.kfold_shuffle is True  # Stored but ignored for timeseries


# =============================================================================
# XGBoostConfig Tests
# =============================================================================


class TestXGBoostConfig:
    """Tests for XGBoostConfig."""

    def test_default_values(self):
        """Test default XGBoost parameters."""
        config = XGBoostConfig()
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.subsample == 0.8
        assert config.colsample_bytree == 0.8
        assert config.objective == "reg:squarederror"
        assert config.random_state == 42
        assert config.n_jobs == -1

    def test_custom_parameters(self):
        """Test custom XGBoost parameters."""
        config = XGBoostConfig(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            reg_alpha=0.1,
            reg_lambda=2.0,
            early_stopping_rounds=50,
        )
        assert config.n_estimators == 500
        assert config.max_depth == 10
        assert config.learning_rate == 0.05
        assert config.reg_alpha == 0.1
        assert config.early_stopping_rounds == 50

    def test_invalid_n_estimators(self):
        """Test invalid n_estimators validation."""
        with pytest.raises(ValueError):
            XGBoostConfig(n_estimators=0)

        with pytest.raises(ValueError):
            XGBoostConfig(n_estimators=20000)

    def test_invalid_learning_rate(self):
        """Test invalid learning_rate validation."""
        with pytest.raises(ValueError):
            XGBoostConfig(learning_rate=0.0)

        with pytest.raises(ValueError):
            XGBoostConfig(learning_rate=1.5)

    def test_invalid_subsample(self):
        """Test invalid subsample validation."""
        with pytest.raises(ValueError):
            XGBoostConfig(subsample=0.0)

        with pytest.raises(ValueError):
            XGBoostConfig(subsample=1.5)


# =============================================================================
# LSTMConfig Tests
# =============================================================================


class TestLSTMConfig:
    """Tests for LSTMConfig."""

    def test_default_values(self):
        """Test default LSTM parameters."""
        config = LSTMConfig()
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.dropout == 0.2
        assert config.bidirectional is False
        assert config.lookback_hours == 72
        assert config.timesteps == 24
        assert config.epochs == 20
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.loss_function == "mse"

    def test_custom_parameters(self):
        """Test custom LSTM parameters."""
        config = LSTMConfig(
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            lookback_hours=48,
            timesteps=12,
            epochs=50,
            batch_size=64,
            learning_rate=0.0001,
            loss_function="huber",
            patience=10,
        )
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.bidirectional is True
        assert config.loss_function == "huber"
        assert config.patience == 10

    def test_invalid_hidden_size(self):
        """Test invalid hidden_size validation."""
        with pytest.raises(ValueError):
            LSTMConfig(hidden_size=4)

        with pytest.raises(ValueError):
            LSTMConfig(hidden_size=2048)

    def test_invalid_dropout(self):
        """Test invalid dropout validation."""
        with pytest.raises(ValueError):
            LSTMConfig(dropout=-0.1)

        with pytest.raises(ValueError):
            LSTMConfig(dropout=1.0)

    def test_invalid_loss_function(self):
        """Test invalid loss function literal."""
        with pytest.raises(ValueError):
            LSTMConfig(loss_function="invalid")


# =============================================================================
# FeatureConfig Tests
# =============================================================================


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_values(self):
        """Test default feature config values."""
        config = FeatureConfig()
        assert config.sharp_bookmakers == ["pinnacle"]
        assert config.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]
        assert config.markets == ["h2h", "spreads", "totals"]
        assert config.outcome == "home"
        assert config.opening_tier == FetchTier.EARLY
        assert config.closing_tier == FetchTier.CLOSING

    def test_custom_bookmakers(self):
        """Test custom bookmaker lists."""
        config = FeatureConfig(
            sharp_bookmakers=["pinnacle", "circasports"],
            retail_bookmakers=["fanduel"],
        )
        assert config.sharp_bookmakers == ["pinnacle", "circasports"]
        assert config.retail_bookmakers == ["fanduel"]

    def test_invalid_tier_order(self):
        """Test invalid tier order validation (opening must be earlier than closing)."""
        with pytest.raises(ValueError, match="must be earlier than"):
            FeatureConfig(
                opening_tier=FetchTier.CLOSING,
                closing_tier=FetchTier.EARLY,
            )

    def test_same_tier_fails(self):
        """Test same opening/closing tier fails."""
        with pytest.raises(ValueError, match="must be earlier than"):
            FeatureConfig(
                opening_tier=FetchTier.SHARP,
                closing_tier=FetchTier.SHARP,
            )

    def test_empty_bookmakers_fails(self):
        """Test that empty bookmaker lists fail."""
        with pytest.raises(ValueError):
            FeatureConfig(sharp_bookmakers=[])


# =============================================================================
# SearchSpace Tests
# =============================================================================


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_int_search_space(self):
        """Test integer search space."""
        space = SearchSpace(
            type="int",
            low=50,
            high=500,
            step=50,
        )
        assert space.type == "int"
        assert space.low == 50
        assert space.high == 500
        assert space.step == 50

    def test_float_search_space(self):
        """Test float search space with log scale."""
        space = SearchSpace(
            type="float",
            low=0.001,
            high=0.3,
            log=True,
        )
        assert space.type == "float"
        assert space.low == 0.001
        assert space.high == 0.3
        assert space.log is True

    def test_categorical_search_space(self):
        """Test categorical search space."""
        space = SearchSpace(
            type="categorical",
            choices=["reg:squarederror", "reg:absoluteerror"],
        )
        assert space.type == "categorical"
        assert space.choices == ["reg:squarederror", "reg:absoluteerror"]

    def test_int_missing_bounds(self):
        """Test int type requires low and high."""
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            SearchSpace(type="int", low=50)

    def test_float_missing_bounds(self):
        """Test float type requires low and high."""
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            SearchSpace(type="float", high=100)

    def test_invalid_bounds(self):
        """Test low >= high fails."""
        with pytest.raises(ValueError, match="must be less than"):
            SearchSpace(type="int", low=100, high=50)

    def test_log_scale_negative(self):
        """Test log scale requires positive low."""
        with pytest.raises(ValueError, match="Log scale requires"):
            SearchSpace(type="float", low=-1, high=1, log=True)

    def test_categorical_missing_choices(self):
        """Test categorical requires choices."""
        with pytest.raises(ValueError, match="requires non-empty 'choices'"):
            SearchSpace(type="categorical")

    def test_categorical_with_bounds_fails(self):
        """Test categorical should not have bounds."""
        with pytest.raises(ValueError, match="should not have 'low'"):
            SearchSpace(type="categorical", choices=["a", "b"], low=0)


# =============================================================================
# TuningConfig Tests
# =============================================================================


class TestTuningConfig:
    """Tests for TuningConfig."""

    def test_default_values(self):
        """Test default tuning config values."""
        config = TuningConfig()
        assert config.n_trials == 100
        assert config.timeout is None
        assert config.direction == "minimize"
        assert config.metric == "val_mse"
        assert config.pruner == "median"
        assert config.sampler == "tpe"

    def test_with_search_spaces(self):
        """Test tuning config with search spaces."""
        config = TuningConfig(
            n_trials=50,
            direction="maximize",
            metric="val_r2",
            search_spaces={
                "n_estimators": SearchSpace(type="int", low=50, high=500),
                "learning_rate": SearchSpace(type="float", low=0.001, high=0.3, log=True),
            },
        )
        assert config.n_trials == 50
        assert config.direction == "maximize"
        assert "n_estimators" in config.search_spaces
        assert config.search_spaces["n_estimators"].type == "int"


# =============================================================================
# TrackingConfig Tests
# =============================================================================


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_default_values(self):
        """Test default tracking config values."""
        config = TrackingConfig()
        assert config.enabled is False
        assert config.tracking_uri == "mlruns"
        assert config.log_model is True
        assert config.log_params is True
        assert config.log_metrics is True

    def test_enabled_config(self):
        """Test enabled tracking config."""
        config = TrackingConfig(
            enabled=True,
            tracking_uri="http://mlflow.example.com",
            experiment_name="my_experiment",
            run_name="run_001",
        )
        assert config.enabled is True
        assert config.tracking_uri == "http://mlflow.example.com"
        assert config.experiment_name == "my_experiment"


# =============================================================================
# TrainingConfig Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_xgboost_config(self):
        """Test training config with XGBoost model."""
        config = TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            ),
            model=XGBoostConfig(n_estimators=200),
        )
        assert config.strategy_type == "xgboost_line_movement"
        assert isinstance(config.model, XGBoostConfig)
        assert config.model.n_estimators == 200

    def test_lstm_config(self):
        """Test training config with LSTM model."""
        config = TrainingConfig(
            strategy_type="lstm_line_movement",
            data=DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            ),
            model=LSTMConfig(hidden_size=128, epochs=30),
        )
        assert config.strategy_type == "lstm_line_movement"
        assert isinstance(config.model, LSTMConfig)
        assert config.model.hidden_size == 128
        assert config.model.epochs == 30

    def test_model_type_mismatch(self):
        """Test that model type must match strategy type."""
        with pytest.raises(ValueError, match="requires XGBoostConfig"):
            TrainingConfig(
                strategy_type="xgboost",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=LSTMConfig(),
            )

        with pytest.raises(ValueError, match="requires LSTMConfig"):
            TrainingConfig(
                strategy_type="lstm",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(),
            )

    def test_default_strategy_params(self):
        """Test default strategy parameters."""
        config = TrainingConfig(
            strategy_type="xgboost",
            data=DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            ),
            model=XGBoostConfig(),
        )
        assert config.min_predicted_movement == 0.02
        assert config.movement_confidence_scale == 5.0
        assert config.base_confidence == 0.52
        assert config.output_path == "models"


# =============================================================================
# MLTrainingConfig Tests
# =============================================================================


class TestMLTrainingConfig:
    """Tests for MLTrainingConfig top-level container."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic test configuration."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="test_experiment",
                tags=["test"],
                description="Test configuration",
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(n_estimators=100),
            ),
        )

    def test_basic_creation(self, basic_config):
        """Test creating a basic configuration."""
        assert basic_config.experiment.name == "test_experiment"
        assert basic_config.training.strategy_type == "xgboost_line_movement"
        assert basic_config.tuning is None
        assert basic_config.tracking is None

    def test_full_config(self):
        """Test creating a full configuration with all sections."""
        config = MLTrainingConfig(
            experiment=ExperimentConfig(name="full_test"),
            training=TrainingConfig(
                strategy_type="lstm_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=LSTMConfig(),
                features=FeatureConfig(sharp_bookmakers=["pinnacle", "circasports"]),
            ),
            tuning=TuningConfig(
                n_trials=50,
                search_spaces={
                    "hidden_size": SearchSpace(type="int", low=32, high=256),
                },
            ),
            tracking=TrackingConfig(enabled=True),
        )
        assert config.experiment.name == "full_test"
        assert config.tuning.n_trials == 50
        assert config.tracking.enabled is True


# =============================================================================
# YAML Loading Tests
# =============================================================================


class TestYAMLLoading:
    """Tests for YAML file loading."""

    @pytest.fixture
    def yaml_config_file(self):
        """Create a temporary YAML config file."""
        config_data = {
            "experiment": {
                "name": "yaml_test",
                "tags": ["test", "yaml"],
                "description": "Test YAML loading",
            },
            "training": {
                "strategy_type": "xgboost_line_movement",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "test_split": 0.2,
                    "random_seed": 42,
                },
                "model": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                },
                "features": {
                    "sharp_bookmakers": ["pinnacle"],
                    "retail_bookmakers": ["fanduel", "draftkings"],
                },
                "output_path": "models/test",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        Path(f.name).unlink()

    def test_load_from_yaml(self, yaml_config_file):
        """Test loading configuration from YAML file."""
        config = MLTrainingConfig.from_yaml(yaml_config_file)

        assert config.experiment.name == "yaml_test"
        assert config.experiment.tags == ["test", "yaml"]
        assert config.training.strategy_type == "xgboost_line_movement"
        assert config.training.data.start_date == date(2024, 1, 1)
        assert config.training.data.end_date == date(2024, 12, 31)
        assert isinstance(config.training.model, XGBoostConfig)
        assert config.training.model.n_estimators == 200
        assert config.training.model.max_depth == 8
        assert config.training.features.sharp_bookmakers == ["pinnacle"]

    def test_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            MLTrainingConfig.from_yaml("nonexistent.yaml")

    def test_empty_yaml_file(self):
        """Test error when YAML file is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Empty configuration"):
                MLTrainingConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_yaml_with_lstm(self):
        """Test YAML loading with LSTM config."""
        config_data = {
            "experiment": {"name": "lstm_yaml_test"},
            "training": {
                "strategy_type": "lstm_line_movement",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "model": {
                    "hidden_size": 128,
                    "num_layers": 3,
                    "epochs": 50,
                    "loss_function": "huber",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = MLTrainingConfig.from_yaml(temp_path)
            assert config.training.strategy_type == "lstm_line_movement"
            assert isinstance(config.training.model, LSTMConfig)
            assert config.training.model.hidden_size == 128
            assert config.training.model.loss_function == "huber"
        finally:
            Path(temp_path).unlink()


# =============================================================================
# JSON Loading Tests
# =============================================================================


class TestJSONLoading:
    """Tests for JSON file loading."""

    @pytest.fixture
    def json_config_file(self):
        """Create a temporary JSON config file."""
        config_data = {
            "experiment": {
                "name": "json_test",
                "tags": ["test", "json"],
            },
            "training": {
                "strategy_type": "xgboost",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                    "test_split": 0.25,
                },
                "model": {
                    "n_estimators": 150,
                    "learning_rate": 0.08,
                },
            },
            "tuning": {
                "n_trials": 50,
                "direction": "minimize",
                "search_spaces": {
                    "n_estimators": {
                        "type": "int",
                        "low": 50,
                        "high": 300,
                    },
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.2,
                        "log": True,
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        Path(temp_path).unlink()

    def test_load_from_json(self, json_config_file):
        """Test loading configuration from JSON file."""
        config = MLTrainingConfig.from_json(json_config_file)

        assert config.experiment.name == "json_test"
        assert config.training.strategy_type == "xgboost"
        assert config.training.data.test_split == 0.25
        assert config.training.model.n_estimators == 150
        assert config.tuning is not None
        assert config.tuning.n_trials == 50
        assert "n_estimators" in config.tuning.search_spaces
        assert config.tuning.search_spaces["learning_rate"].log is True

    def test_json_file_not_found(self):
        """Test error when JSON file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            MLTrainingConfig.from_json("nonexistent.json")


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for YAML/JSON serialization."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="serialization_test",
                tags=["serialize", "test"],
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 3, 1),
                    end_date=date(2024, 9, 30),
                    test_split=0.15,
                ),
                model=XGBoostConfig(
                    n_estimators=250,
                    max_depth=7,
                ),
            ),
            tuning=TuningConfig(
                n_trials=30,
                search_spaces={
                    "n_estimators": SearchSpace(type="int", low=100, high=500),
                },
            ),
        )

    def test_to_yaml(self, config):
        """Test serializing to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            config.to_yaml(filepath)

            # Load back and verify
            loaded = MLTrainingConfig.from_yaml(filepath)
            assert loaded.experiment.name == "serialization_test"
            assert loaded.training.model.n_estimators == 250
            assert loaded.tuning.n_trials == 30

    def test_to_json(self, config):
        """Test serializing to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            config.to_json(filepath)

            # Load back and verify
            loaded = MLTrainingConfig.from_json(filepath)
            assert loaded.experiment.name == "serialization_test"
            assert loaded.training.data.start_date == date(2024, 3, 1)

    def test_to_dict(self, config):
        """Test converting to dictionary."""
        data = config.to_dict()

        assert data["experiment"]["name"] == "serialization_test"
        assert data["training"]["strategy_type"] == "xgboost_line_movement"
        assert data["training"]["data"]["start_date"] == "2024-03-01"
        assert data["training"]["model"]["n_estimators"] == 250

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "experiment": {"name": "dict_test"},
            "training": {
                "strategy_type": "xgboost",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "model": {"n_estimators": 100},
            },
        }

        config = MLTrainingConfig.from_dict(data)
        assert config.experiment.name == "dict_test"
        assert config.training.model.n_estimators == 100

    def test_roundtrip_yaml(self, config):
        """Test YAML roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "roundtrip.yaml"
            config.to_yaml(filepath)
            loaded = MLTrainingConfig.from_yaml(filepath)

            # Verify all fields match
            assert loaded.experiment.name == config.experiment.name
            assert loaded.experiment.tags == config.experiment.tags
            assert loaded.training.strategy_type == config.training.strategy_type
            assert loaded.training.data.start_date == config.training.data.start_date
            assert loaded.training.model.n_estimators == config.training.model.n_estimators
            assert loaded.tuning.n_trials == config.tuning.n_trials

    def test_creates_parent_directories(self, config):
        """Test that serialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dirs" / "config.yaml"
            config.to_yaml(filepath)
            assert filepath.exists()


# =============================================================================
# Validation Error Message Tests
# =============================================================================


class TestValidationErrorMessages:
    """Tests for clear validation error messages."""

    def test_date_range_error_message(self):
        """Test clear error message for invalid date range."""
        with pytest.raises(ValueError) as excinfo:
            DataConfig(
                start_date=date(2024, 12, 31),
                end_date=date(2024, 1, 1),
            )
        assert "must be before" in str(excinfo.value)
        assert "2024-12-31" in str(excinfo.value)
        assert "2024-01-01" in str(excinfo.value)

    def test_splits_error_message(self):
        """Test clear error message for excessive splits."""
        with pytest.raises(ValueError) as excinfo:
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                test_split=0.5,
                validation_split=0.6,
            )
        assert "must be less than 1.0" in str(excinfo.value)

    def test_search_space_bounds_error(self):
        """Test clear error message for search space bounds."""
        with pytest.raises(ValueError) as excinfo:
            SearchSpace(type="int", low=100, high=50)
        assert "must be less than" in str(excinfo.value)

    def test_model_type_mismatch_error(self):
        """Test clear error message for model type mismatch."""
        with pytest.raises(ValueError) as excinfo:
            TrainingConfig(
                strategy_type="xgboost",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=LSTMConfig(),
            )
        assert "requires XGBoostConfig" in str(excinfo.value)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = MLTrainingConfig(
            experiment=ExperimentConfig(name="minimal"),
            training=TrainingConfig(
                strategy_type="xgboost",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 2),  # Minimum 1 day
                ),
                model=XGBoostConfig(),
            ),
        )
        assert config.experiment.name == "minimal"

    def test_all_strategy_types(self):
        """Test all supported strategy types."""
        for strategy in ["xgboost", "xgboost_line_movement"]:
            config = TrainingConfig(
                strategy_type=strategy,
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(),
            )
            assert config.strategy_type == strategy

        for strategy in ["lstm", "lstm_line_movement"]:
            config = TrainingConfig(
                strategy_type=strategy,
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=LSTMConfig(),
            )
            assert config.strategy_type == strategy

    def test_date_string_parsing_in_yaml(self):
        """Test that date strings are properly parsed from YAML."""
        config_data = {
            "experiment": {"name": "date_test"},
            "training": {
                "strategy_type": "xgboost",
                "data": {
                    "start_date": "2024-10-15",  # Specific date format
                    "end_date": "2025-01-20",
                },
                "model": {},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = MLTrainingConfig.from_yaml(temp_path)
            assert config.training.data.start_date == date(2024, 10, 15)
            assert config.training.data.end_date == date(2025, 1, 20)
        finally:
            Path(temp_path).unlink()

    def test_special_characters_in_names(self):
        """Test experiment names with special characters."""
        config = ExperimentConfig(
            name="experiment_v1.0-beta_2024",
            description="Test with special chars: @#$%",
        )
        assert config.name == "experiment_v1.0-beta_2024"

    def test_unicode_in_description(self):
        """Test unicode characters in description."""
        config = ExperimentConfig(
            name="unicode_test",
            description="Test with unicode: αβγδ 中文 日本語",
        )
        assert "αβγδ" in config.description


# =============================================================================
# Unknown Field Rejection Tests
# =============================================================================


class TestUnknownFieldRejection:
    """Tests for rejecting unknown fields in configurations."""

    def test_experiment_config_unknown_field_rejected(self):
        """Test that ExperimentConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            ExperimentConfig(name="test", unknown_param=42)

    def test_data_config_unknown_field_rejected(self):
        """Test that DataConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                unknown_param="invalid",
            )

    def test_xgboost_config_unknown_field_rejected(self):
        """Test that XGBoostConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            XGBoostConfig(n_estimators=100, unknown_param=42)

    def test_lstm_config_unknown_field_rejected(self):
        """Test that LSTMConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            LSTMConfig(hidden_size=64, unknown_param="invalid")

    def test_feature_config_unknown_field_rejected(self):
        """Test that FeatureConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            FeatureConfig(unknown_param=123)

    def test_search_space_unknown_field_rejected(self):
        """Test that SearchSpace rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            SearchSpace(type="int", low=0, high=100, unknown_param="invalid")

    def test_tuning_config_unknown_field_rejected(self):
        """Test that TuningConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TuningConfig(n_trials=50, unknown_param=True)

    def test_tracking_config_unknown_field_rejected(self):
        """Test that TrackingConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TrackingConfig(enabled=True, unknown_param="invalid")

    def test_training_config_unknown_field_rejected(self):
        """Test that TrainingConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TrainingConfig(
                strategy_type="xgboost",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                model=XGBoostConfig(),
                unknown_param="invalid",
            )

    def test_ml_training_config_unknown_field_rejected(self):
        """Test that MLTrainingConfig rejects unknown fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            MLTrainingConfig(
                experiment=ExperimentConfig(name="test"),
                training=TrainingConfig(
                    strategy_type="xgboost",
                    data=DataConfig(
                        start_date=date(2024, 1, 1),
                        end_date=date(2024, 12, 31),
                    ),
                    model=XGBoostConfig(),
                ),
                unknown_param="invalid",
            )

    def test_yaml_unknown_field_rejected(self):
        """Test that unknown fields in YAML are rejected."""
        config_data = {
            "experiment": {
                "name": "test",
                "unknown_field": "should_fail",
            },
            "training": {
                "strategy_type": "xgboost",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "model": {},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Extra inputs are not permitted"):
                MLTrainingConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Config Inheritance Tests
# =============================================================================


class TestConfigInheritance:
    """Tests for configuration inheritance via base field."""

    @pytest.fixture
    def base_config_data(self):
        """Base config data for inheritance tests."""
        return {
            "experiment": {
                "name": "base_experiment",
                "tags": ["base", "xgboost"],
                "description": "Base experiment configuration",
            },
            "training": {
                "strategy_type": "xgboost_line_movement",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "test_split": 0.2,
                    "random_seed": 42,
                },
                "model": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                },
                "features": {
                    "sharp_bookmakers": ["pinnacle"],
                    "retail_bookmakers": ["fanduel", "draftkings", "betmgm"],
                },
                "output_path": "models/base",
            },
        }

    def test_basic_inheritance(self, base_config_data):
        """Test basic config inheritance from a base file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Create child config that inherits from base
            child_data = {
                "base": "base.yaml",
                "experiment": {
                    "name": "child_experiment",
                },
                "training": {
                    "model": {
                        "n_estimators": 200,
                    },
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            # Load child config
            config = MLTrainingConfig.from_yaml(child_path)

            # Child values should override base
            assert config.experiment.name == "child_experiment"
            assert config.training.model.n_estimators == 200

            # Base values should be inherited
            assert config.experiment.tags == ["base", "xgboost"]
            assert config.experiment.description == "Base experiment configuration"
            assert config.training.model.max_depth == 6
            assert config.training.model.learning_rate == 0.1
            assert config.training.data.start_date == date(2024, 1, 1)

    def test_deep_merge_nested_dicts(self, base_config_data):
        """Test that nested dictionaries are deep-merged correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Child with nested overrides
            child_data = {
                "base": "base.yaml",
                "experiment": {
                    "name": "nested_test",
                },
                "training": {
                    "data": {
                        "start_date": "2024-06-01",
                        # end_date should be inherited
                    },
                    "model": {
                        "learning_rate": 0.05,
                        # n_estimators and max_depth should be inherited
                    },
                    "features": {
                        "retail_bookmakers": ["caesars"],
                        # sharp_bookmakers should be inherited
                    },
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            # Child overrides
            assert config.training.data.start_date == date(2024, 6, 1)
            assert config.training.model.learning_rate == 0.05
            assert config.training.features.retail_bookmakers == ["caesars"]

            # Inherited from base
            assert config.training.data.end_date == date(2024, 12, 31)
            assert config.training.model.n_estimators == 100
            assert config.training.model.max_depth == 6
            assert config.training.features.sharp_bookmakers == ["pinnacle"]

    def test_multi_level_inheritance(self, base_config_data):
        """Test inheritance chains with multiple levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Level 1: base
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Level 2: intermediate inherits from base
            intermediate_data = {
                "base": "base.yaml",
                "experiment": {
                    "name": "intermediate",
                    "tags": ["intermediate"],
                },
                "training": {
                    "model": {
                        "n_estimators": 150,
                    },
                },
            }
            intermediate_path = Path(tmpdir) / "intermediate.yaml"
            with open(intermediate_path, "w") as f:
                yaml.dump(intermediate_data, f)

            # Level 3: final inherits from intermediate
            final_data = {
                "base": "intermediate.yaml",
                "experiment": {
                    "name": "final",
                },
                "training": {
                    "model": {
                        "learning_rate": 0.01,
                    },
                },
            }
            final_path = Path(tmpdir) / "final.yaml"
            with open(final_path, "w") as f:
                yaml.dump(final_data, f)

            config = MLTrainingConfig.from_yaml(final_path)

            # Final overrides
            assert config.experiment.name == "final"
            assert config.training.model.learning_rate == 0.01

            # From intermediate
            assert config.experiment.tags == ["intermediate"]
            assert config.training.model.n_estimators == 150

            # From base
            assert config.training.model.max_depth == 6
            assert config.training.data.start_date == date(2024, 1, 1)

    def test_circular_inheritance_detection(self, base_config_data):
        """Test that circular inheritance is detected and rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Config A inherits from B
            config_a_data = {
                "base": "config_b.yaml",
                "experiment": {"name": "config_a"},
                "training": base_config_data["training"],
            }
            config_a_path = Path(tmpdir) / "config_a.yaml"
            with open(config_a_path, "w") as f:
                yaml.dump(config_a_data, f)

            # Config B inherits from A (circular!)
            config_b_data = {
                "base": "config_a.yaml",
                "experiment": {"name": "config_b"},
                "training": base_config_data["training"],
            }
            config_b_path = Path(tmpdir) / "config_b.yaml"
            with open(config_b_path, "w") as f:
                yaml.dump(config_b_data, f)

            with pytest.raises(ValueError, match="Circular inheritance detected"):
                MLTrainingConfig.from_yaml(config_a_path)

    def test_self_inheritance_detection(self, base_config_data):
        """Test that self-inheritance is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Config inherits from itself
            self_inherit_data = {
                "base": "self.yaml",
                "experiment": {"name": "self_inherit"},
                "training": base_config_data["training"],
            }
            self_path = Path(tmpdir) / "self.yaml"
            with open(self_path, "w") as f:
                yaml.dump(self_inherit_data, f)

            with pytest.raises(ValueError, match="Circular inheritance detected"):
                MLTrainingConfig.from_yaml(self_path)

    def test_relative_path_resolution(self, base_config_data):
        """Test that relative paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            base_dir = Path(tmpdir) / "configs" / "base"
            child_dir = Path(tmpdir) / "configs" / "experiments"
            base_dir.mkdir(parents=True)
            child_dir.mkdir(parents=True)

            # Base config in base directory
            base_path = base_dir / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Child config in experiments directory with relative path to base
            child_data = {
                "base": "../base/base.yaml",
                "experiment": {"name": "relative_test"},
                "training": {
                    "model": {"n_estimators": 300},
                },
            }
            child_path = child_dir / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            assert config.experiment.name == "relative_test"
            assert config.training.model.n_estimators == 300
            assert config.training.model.max_depth == 6  # From base

    def test_absolute_path_inheritance(self, base_config_data):
        """Test inheritance with absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Child with absolute path to base
            child_data = {
                "base": str(base_path.absolute()),
                "experiment": {"name": "absolute_test"},
                "training": {
                    "model": {"max_depth": 10},
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            assert config.experiment.name == "absolute_test"
            assert config.training.model.max_depth == 10
            assert config.training.model.n_estimators == 100  # From base

    def test_serialization_excludes_base_field(self, base_config_data):
        """Test that serialized configs don't include the base field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Child config
            child_data = {
                "base": "base.yaml",
                "experiment": {"name": "serialize_test"},
                "training": {
                    "model": {"n_estimators": 250},
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            # Load and serialize
            config = MLTrainingConfig.from_yaml(child_path)
            output_path = Path(tmpdir) / "resolved.yaml"
            config.to_yaml(output_path)

            # Verify serialized config has no base field
            with open(output_path) as f:
                resolved_data = yaml.safe_load(f)

            assert "base" not in resolved_data
            assert resolved_data["experiment"]["name"] == "serialize_test"
            assert resolved_data["training"]["model"]["n_estimators"] == 250
            # Inherited values should be present
            assert resolved_data["training"]["model"]["max_depth"] == 6

    def test_missing_base_file(self, base_config_data):
        """Test error when base file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_data = {
                "base": "nonexistent.yaml",
                "experiment": {"name": "missing_base"},
                "training": base_config_data["training"],
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                MLTrainingConfig.from_yaml(child_path)

    def test_json_inheritance(self, base_config_data):
        """Test inheritance with JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base JSON config
            base_path = Path(tmpdir) / "base.json"
            with open(base_path, "w") as f:
                json.dump(base_config_data, f)

            # Child JSON config
            child_data = {
                "base": "base.json",
                "experiment": {"name": "json_inherit"},
                "training": {
                    "model": {"n_estimators": 175},
                },
            }
            child_path = Path(tmpdir) / "child.json"
            with open(child_path, "w") as f:
                json.dump(child_data, f)

            config = MLTrainingConfig.from_json(child_path)

            assert config.experiment.name == "json_inherit"
            assert config.training.model.n_estimators == 175
            assert config.training.model.max_depth == 6  # From base

    def test_mixed_format_inheritance(self, base_config_data):
        """Test YAML inheriting from JSON and vice versa."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base JSON config
            base_path = Path(tmpdir) / "base.json"
            with open(base_path, "w") as f:
                json.dump(base_config_data, f)

            # Child YAML inheriting from JSON
            child_data = {
                "base": "base.json",
                "experiment": {"name": "mixed_format"},
                "training": {
                    "model": {"learning_rate": 0.2},
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            assert config.experiment.name == "mixed_format"
            assert config.training.model.learning_rate == 0.2
            assert config.training.model.n_estimators == 100  # From base

    def test_list_replacement_not_merge(self, base_config_data):
        """Test that lists are replaced, not merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config_data, f)

            # Child with different tags (should replace, not append)
            child_data = {
                "base": "base.yaml",
                "experiment": {
                    "name": "list_test",
                    "tags": ["child", "test"],
                },
                "training": {
                    "model": {},
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            # Tags should be replaced, not merged
            assert config.experiment.tags == ["child", "test"]
            assert "base" not in config.experiment.tags

    def test_no_base_field_works_normally(self):
        """Test that configs without base field work as before."""
        config_data = {
            "experiment": {"name": "no_base"},
            "training": {
                "strategy_type": "xgboost",
                "data": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "model": {"n_estimators": 100},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = MLTrainingConfig.from_yaml(temp_path)
            assert config.experiment.name == "no_base"
            assert config.training.model.n_estimators == 100
        finally:
            Path(temp_path).unlink()

    def test_unsupported_file_format(self, base_config_data):
        """Test error for unsupported file formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with unsupported extension
            unsupported_path = Path(tmpdir) / "config.txt"
            with open(unsupported_path, "w") as f:
                f.write("invalid")

            # Create child that references unsupported format
            child_data = {
                "base": "config.txt",
                "experiment": {"name": "unsupported"},
                "training": base_config_data["training"],
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            with pytest.raises(ValueError, match="Unsupported config file format"):
                MLTrainingConfig.from_yaml(child_path)

    def test_tuning_and_tracking_inheritance(self, base_config_data):
        """Test inheritance of optional sections (tuning, tracking)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Base with tuning and tracking
            base_with_options = base_config_data.copy()
            base_with_options["tuning"] = {
                "n_trials": 100,
                "direction": "minimize",
                "search_spaces": {
                    "n_estimators": {"type": "int", "low": 50, "high": 500},
                },
            }
            base_with_options["tracking"] = {
                "enabled": True,
                "tracking_uri": "mlruns",
            }

            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_with_options, f)

            # Child that overrides tuning but keeps tracking
            child_data = {
                "base": "base.yaml",
                "experiment": {"name": "options_test"},
                "training": {
                    "model": {},
                },
                "tuning": {
                    "n_trials": 50,
                },
            }
            child_path = Path(tmpdir) / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_data, f)

            config = MLTrainingConfig.from_yaml(child_path)

            # Tuning overridden
            assert config.tuning.n_trials == 50
            # Tuning search_spaces inherited
            assert "n_estimators" in config.tuning.search_spaces
            # Tracking inherited
            assert config.tracking.enabled is True
            assert config.tracking.tracking_uri == "mlruns"


# =============================================================================
# Deep Merge Utility Tests
# =============================================================================


class TestDeepMerge:
    """Tests for the deep_merge utility function."""

    def test_simple_merge(self):
        """Test simple dictionary merge."""
        from odds_analytics.training.config import deep_merge

        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        # Original dicts should be unchanged
        assert base == {"a": 1, "b": 2}
        assert override == {"b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        from odds_analytics.training.config import deep_merge

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        result = deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}

    def test_deep_nested_merge(self):
        """Test deeply nested dictionary merge."""
        from odds_analytics.training.config import deep_merge

        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        override = {"level1": {"level2": {"level3": {"b": 20, "c": 30}}}}
        result = deep_merge(base, override)

        assert result["level1"]["level2"]["level3"] == {"a": 1, "b": 20, "c": 30}

    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        from odds_analytics.training.config import deep_merge

        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)

        assert result == {"items": [4, 5]}

    def test_none_override(self):
        """Test that None values in override replace base values."""
        from odds_analytics.training.config import deep_merge

        base = {"a": 1, "b": 2}
        override = {"a": None}
        result = deep_merge(base, override)

        assert result == {"a": None, "b": 2}

    def test_empty_dicts(self):
        """Test merging with empty dictionaries."""
        from odds_analytics.training.config import deep_merge

        assert deep_merge({}, {"a": 1}) == {"a": 1}
        assert deep_merge({"a": 1}, {}) == {"a": 1}
        assert deep_merge({}, {}) == {}
