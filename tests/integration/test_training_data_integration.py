"""
Integration tests for config-driven training data preparation.

Tests the full flow:
1. Config loading from YAML/JSON
2. Event filtering by date range from database
3. Data preparation based on strategy type
4. Output shape verification

These tests require a database connection and test data.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from odds_analytics.sequence_loader import prepare_lstm_training_data
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
from odds_analytics.xgboost_line_movement import prepare_tabular_training_data
from odds_core.models import EventStatus

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestTrainingDataIntegration:
    """Integration tests for training data preparation pipeline."""

    @pytest.fixture
    def xgboost_config(self):
        """Create XGBoost training configuration."""
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
                    end_date=date(2024, 12, 31),
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
                    retail_bookmakers=["fanduel", "draftkings", "betmgm"],
                    opening_hours_before=48.0,
                    closing_hours_before=0.5,
                ),
            ),
        )

    @pytest.fixture
    def lstm_config(self):
        """Create LSTM training configuration."""
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
                    end_date=date(2024, 12, 31),
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
                    "end_date": "2024-12-31",
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
        assert config.training.data.end_date == date(2024, 12, 31)
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
    async def test_prepare_tabular_with_feature_config_integration(self, async_session):
        """Test prepare_tabular_training_data with FeatureConfig parameter."""
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(async_session)

        # Get events from database
        start_date = datetime(2024, 10, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            status=EventStatus.FINAL,
        )

        if not events:
            pytest.skip("No events in test database for date range")

        # Create feature config
        feature_config = FeatureConfig(
            outcome="home",
            markets=["h2h"],
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
            opening_hours_before=48.0,
            closing_hours_before=0.5,
        )

        # Prepare data using config
        X, y, feature_names = await prepare_tabular_training_data(
            events=events,
            session=async_session,
            feature_config=feature_config,
        )

        # Verify output shapes
        if len(X) > 0:
            assert X.ndim == 2
            assert y.ndim == 1
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == len(feature_names)
            assert len(feature_names) > 0

    @pytest.mark.asyncio
    async def test_prepare_lstm_with_feature_config_integration(self, async_session):
        """Test prepare_lstm_training_data with FeatureConfig parameter."""
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(async_session)

        # Get events from database
        start_date = datetime(2024, 10, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            status=EventStatus.FINAL,
        )

        if not events:
            pytest.skip("No events in test database for date range")

        # Create feature config
        feature_config = FeatureConfig(
            outcome="home",
            markets=["h2h"],
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel"],
        )

        # Prepare data using config
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=async_session,
            lookback_hours=72,
            timesteps=24,
            feature_config=feature_config,
        )

        # Verify output shapes
        if len(X) > 0:
            assert X.ndim == 3  # (samples, timesteps, features)
            assert y.ndim == 1
            assert masks.ndim == 2
            assert X.shape[0] == y.shape[0] == masks.shape[0]
            assert X.shape[1] == masks.shape[1]  # timesteps match

    @pytest.mark.asyncio
    async def test_full_pipeline_xgboost(self, xgboost_config, async_session):
        """Test full XGBoost pipeline: config -> data -> result."""
        try:
            result = await prepare_training_data_from_config(xgboost_config, async_session)

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

            # Verify split ratio
            total_samples = result.num_train_samples + result.num_test_samples
            actual_test_ratio = result.num_test_samples / total_samples
            expected_test_ratio = xgboost_config.training.data.test_split
            assert abs(actual_test_ratio - expected_test_ratio) < 0.05  # Allow 5% tolerance

        except ValueError as e:
            if "No events found" in str(e):
                pytest.skip("No events in test database for date range")
            raise

    @pytest.mark.asyncio
    async def test_full_pipeline_lstm(self, lstm_config, async_session):
        """Test full LSTM pipeline: config -> data -> result with masks."""
        try:
            result = await prepare_training_data_from_config(lstm_config, async_session)

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

        except ValueError as e:
            if "No events found" in str(e):
                pytest.skip("No events in test database for date range")
            raise

    @pytest.mark.asyncio
    async def test_result_to_dict_integration(self, xgboost_config, async_session):
        """Test that result can be converted to dict for serialization."""
        try:
            result = await prepare_training_data_from_config(xgboost_config, async_session)

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

        except ValueError as e:
            if "No events found" in str(e):
                pytest.skip("No events in test database for date range")
            raise

    @pytest.mark.asyncio
    async def test_backward_compatibility_legacy_params(self, async_session):
        """Test that legacy parameter-based calls still work."""
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(async_session)

        start_date = datetime(2024, 10, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            status=EventStatus.FINAL,
        )

        if not events:
            pytest.skip("No events in test database for date range")

        # Call with legacy style (no FeatureConfig)
        X, y, feature_names = await prepare_tabular_training_data(
            events=events,
            session=async_session,
            outcome="home",
            market="h2h",
            opening_hours_before=48.0,
            closing_hours_before=0.5,
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
        )

        # Should work without error
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
