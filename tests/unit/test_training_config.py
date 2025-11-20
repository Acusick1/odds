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
        assert config.opening_hours_before == 48.0
        assert config.closing_hours_before == 0.5

    def test_custom_bookmakers(self):
        """Test custom bookmaker lists."""
        config = FeatureConfig(
            sharp_bookmakers=["pinnacle", "circasports"],
            retail_bookmakers=["fanduel"],
        )
        assert config.sharp_bookmakers == ["pinnacle", "circasports"]
        assert config.retail_bookmakers == ["fanduel"]

    def test_invalid_timing(self):
        """Test invalid timing validation (opening <= closing)."""
        with pytest.raises(ValueError, match="must be greater than"):
            FeatureConfig(
                opening_hours_before=1.0,
                closing_hours_before=2.0,
            )

    def test_same_timing_fails(self):
        """Test same opening/closing time fails."""
        with pytest.raises(ValueError, match="must be greater than"):
            FeatureConfig(
                opening_hours_before=24.0,
                closing_hours_before=24.0,
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
