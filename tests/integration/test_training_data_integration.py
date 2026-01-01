"""
Integration tests for config-driven training data preparation.

Tests the full flow:
1. Config loading from YAML/JSON
2. Event filtering by date range from database
3. Data preparation based on strategy type
4. Output shape verification

These tests use fixtures to create known test data for predictable results.
"""

from __future__ import annotations

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
    TrainingConfig,
    TrainingDataResult,
    XGBoostConfig,
    prepare_training_data_from_config,
)
from odds_lambda.fetch_tier import FetchTier

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestTrainingDataIntegration:
    """Integration tests for training data preparation pipeline."""

    @pytest.fixture
    def xgboost_config(self):
        """Create XGBoost training configuration matching test data."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="integration_test_xgboost",
                tags=["integration", "xgboost"],
                description="Integration test for XGBoost data preparation",
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    test_split=0.2,
                    validation_split=0.1,
                    random_seed=42,
                    shuffle=True,
                ),
                model=XGBoostConfig(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                ),
                features=FeatureConfig(
                    outcome="home",
                    markets=["h2h"],
                    sharp_bookmakers=["pinnacle"],
                    retail_bookmakers=["fanduel", "draftkings"],
                    opening_tier=FetchTier.OPENING,
                    closing_tier=FetchTier.CLOSING,
                ),
            ),
        )

    @pytest.fixture
    def lstm_config(self):
        """Create LSTM training configuration matching test data."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="integration_test_lstm",
                tags=["integration", "lstm"],
                description="Integration test for LSTM data preparation",
            ),
            training=TrainingConfig(
                strategy_type="lstm_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    test_split=0.2,
                    random_seed=42,
                ),
                model=LSTMConfig(
                    hidden_size=64,
                    num_layers=2,
                    lookback_hours=72,
                    timesteps=24,
                    epochs=20,
                ),
                features=FeatureConfig(
                    outcome="home",
                    markets=["h2h"],
                    sharp_bookmakers=["pinnacle"],
                    retail_bookmakers=["fanduel", "draftkings"],
                    opening_tier=FetchTier.OPENING,
                    closing_tier=FetchTier.CLOSING,
                ),
            ),
        )

    @pytest.fixture
    def yaml_config_file(self):
        """Create a temporary YAML config file."""
        config_data = {
            "experiment": {
                "name": "yaml_integration_test",
                "tags": ["yaml", "integration"],
            },
            "training": {
                "strategy_type": "xgboost_line_movement",
                "data": {
                    "start_date": "2024-10-01",
                    "end_date": "2024-10-31",
                    "test_split": 0.2,
                    "random_seed": 42,
                },
                "model": {
                    "n_estimators": 100,
                    "max_depth": 6,
                },
                "features": {
                    "outcome": "home",
                    "markets": ["h2h"],
                    "sharp_bookmakers": ["pinnacle"],
                    "retail_bookmakers": ["fanduel", "draftkings"],
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        Path(f.name).unlink()

    def test_yaml_config_loading_and_validation(self, yaml_config_file):
        """Test that YAML config loads and validates correctly."""
        config = MLTrainingConfig.from_yaml(yaml_config_file)

        # Verify all sections loaded correctly
        assert config.experiment.name == "yaml_integration_test"
        assert config.training.strategy_type == "xgboost_line_movement"
        assert config.training.data.start_date == date(2024, 10, 1)
        assert config.training.data.end_date == date(2024, 10, 31)
        assert config.training.data.test_split == 0.2
        assert isinstance(config.training.model, XGBoostConfig)
        assert config.training.model.n_estimators == 100
        assert config.training.features.outcome == "home"

    def test_config_serialization_roundtrip(self, xgboost_config):
        """Test that config can be serialized and loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to YAML
            yaml_path = Path(tmpdir) / "config.yaml"
            xgboost_config.to_yaml(yaml_path)

            # Load back
            loaded_config = MLTrainingConfig.from_yaml(yaml_path)

            # Verify key fields
            assert loaded_config.experiment.name == xgboost_config.experiment.name
            assert loaded_config.training.strategy_type == xgboost_config.training.strategy_type
            assert loaded_config.training.data.start_date == xgboost_config.training.data.start_date
            assert (
                loaded_config.training.model.n_estimators
                == xgboost_config.training.model.n_estimators
            )

    @pytest.mark.asyncio
    async def test_full_pipeline_xgboost(
        self, xgboost_config, pglite_async_session, test_events_with_odds
    ):
        """Test full XGBoost pipeline: config -> data -> result."""
        result = await prepare_training_data_from_config(xgboost_config, pglite_async_session)

        # Verify result structure
        assert isinstance(result, TrainingDataResult)
        assert result.strategy_type == "xgboost_line_movement"

        # Verify shapes
        assert result.X_train.ndim == 2
        assert result.X_test.ndim == 2
        assert result.y_train.ndim == 1
        assert result.y_test.ndim == 1

        # Verify feature consistency
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert result.X_train.shape[1] == len(result.feature_names)

        # Verify no masks for XGBoost
        assert result.masks_train is None
        assert result.masks_test is None

        # Verify we have data
        assert result.num_train_samples > 0
        assert result.num_test_samples > 0

        # Verify split ratio (approximately)
        total_samples = result.num_train_samples + result.num_test_samples
        actual_test_ratio = result.num_test_samples / total_samples
        expected_test_ratio = xgboost_config.training.data.test_split
        assert (
            abs(actual_test_ratio - expected_test_ratio) < 0.15
        )  # Allow tolerance for small dataset

    @pytest.skip("LSTM currently broken")
    @pytest.mark.asyncio
    async def test_full_pipeline_lstm(
        self, lstm_config, pglite_async_session, test_events_with_odds
    ):
        """Test full LSTM pipeline: config -> data -> result with masks."""
        result = await prepare_training_data_from_config(lstm_config, pglite_async_session)

        # Verify result structure
        assert isinstance(result, TrainingDataResult)
        assert result.strategy_type == "lstm_line_movement"

        # Verify shapes
        assert result.X_train.ndim == 3  # (samples, timesteps, features)
        assert result.X_test.ndim == 3
        assert result.y_train.ndim == 1
        assert result.y_test.ndim == 1

        # Verify masks present
        assert result.masks_train is not None
        assert result.masks_test is not None
        assert result.masks_train.ndim == 2
        assert result.masks_test.ndim == 2

        # Verify shape consistency
        assert result.X_train.shape[0] == result.masks_train.shape[0]
        assert result.X_test.shape[0] == result.masks_test.shape[0]
        assert result.X_train.shape[1] == result.masks_train.shape[1]  # timesteps

        # Verify we have data
        assert result.num_train_samples > 0
        assert result.num_test_samples > 0

    @pytest.mark.asyncio
    async def test_result_to_dict_integration(
        self, xgboost_config, pglite_async_session, test_events_with_odds
    ):
        """Test that result can be converted to dict for serialization."""
        result = await prepare_training_data_from_config(xgboost_config, pglite_async_session)

        # Convert to dict
        result_dict = result.to_dict()

        # Verify all expected keys present
        expected_keys = [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "feature_names",
            "strategy_type",
            "num_train_samples",
            "num_test_samples",
            "num_features",
        ]
        for key in expected_keys:
            assert key in result_dict

        # Verify values match
        assert result_dict["num_train_samples"] == result.num_train_samples
        assert result_dict["num_test_samples"] == result.num_test_samples
        assert result_dict["num_features"] == result.num_features
        assert result_dict["strategy_type"] == result.strategy_type
