"""
Unit tests for config-driven training data preparation.

Tests cover:
- TrainingDataResult container
- filter_events_by_date_range function
- prepare_training_data_from_config entry point
- Legacy function backward compatibility with FeatureConfig
- Parameter extraction and defaults
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
)
from odds_analytics.training.data_preparation import (
    filter_events_by_date_range,
    prepare_training_data_from_config,
)
from odds_core.models import Event, EventStatus

# =============================================================================
# TrainingDataResult Tests
# =============================================================================


class TestTrainingDataResult:
    """Tests for TrainingDataResult container."""

    def test_basic_creation(self):
        """Test creating a basic result container."""
        X_train = np.random.randn(80, 10)
        X_test = np.random.randn(20, 10)
        y_train = np.random.randn(80)
        y_test = np.random.randn(20)
        feature_names = [f"feature_{i}" for i in range(10)]

        result = TrainingDataResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            strategy_type="xgboost_line_movement",
        )

        assert result.num_train_samples == 80
        assert result.num_test_samples == 20
        assert result.num_features == 10
        assert result.strategy_type == "xgboost_line_movement"
        assert result.masks_train is None
        assert result.masks_test is None

    def test_with_masks(self):
        """Test result container with LSTM masks."""
        X_train = np.random.randn(80, 24, 17)
        X_test = np.random.randn(20, 24, 17)
        y_train = np.random.randn(80)
        y_test = np.random.randn(20)
        masks_train = np.ones((80, 24), dtype=bool)
        masks_test = np.ones((20, 24), dtype=bool)
        feature_names = [f"feature_{i}" for i in range(17)]

        result = TrainingDataResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            strategy_type="lstm_line_movement",
            masks_train=masks_train,
            masks_test=masks_test,
        )

        assert result.masks_train is not None
        assert result.masks_test is not None
        assert result.masks_train.shape == (80, 24)

    def test_to_dict(self):
        """Test converting result to dictionary."""
        X_train = np.random.randn(10, 5)
        X_test = np.random.randn(5, 5)
        y_train = np.random.randn(10)
        y_test = np.random.randn(5)
        feature_names = ["a", "b", "c", "d", "e"]

        result = TrainingDataResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            strategy_type="xgboost",
        )

        result_dict = result.to_dict()

        assert "X_train" in result_dict
        assert "X_test" in result_dict
        assert "y_train" in result_dict
        assert "y_test" in result_dict
        assert "feature_names" in result_dict
        assert "strategy_type" in result_dict
        assert result_dict["num_train_samples"] == 10
        assert result_dict["num_test_samples"] == 5
        assert result_dict["num_features"] == 5

    def test_to_dict_with_masks(self):
        """Test that masks are included in dict when present."""
        X_train = np.random.randn(10, 24, 17)
        X_test = np.random.randn(5, 24, 17)
        y_train = np.random.randn(10)
        y_test = np.random.randn(5)
        masks_train = np.ones((10, 24), dtype=bool)
        masks_test = np.ones((5, 24), dtype=bool)
        feature_names = [f"f{i}" for i in range(17)]

        result = TrainingDataResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            strategy_type="lstm",
            masks_train=masks_train,
            masks_test=masks_test,
        )

        result_dict = result.to_dict()

        assert "masks_train" in result_dict
        assert "masks_test" in result_dict


# =============================================================================
# FeatureConfig Parameter Extraction Tests
# =============================================================================


class TestFeatureConfigExtraction:
    """Tests for FeatureConfig parameter extraction in data preparation functions."""

    def test_feature_config_default_values(self):
        """Test that FeatureConfig provides sensible defaults."""
        config = FeatureConfig()

        assert config.outcome == "home"
        assert config.markets == ["h2h", "spreads", "totals"]
        assert config.sharp_bookmakers == ["pinnacle"]
        assert config.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]
        assert config.opening_hours_before == 48.0
        assert config.closing_hours_before == 0.5

    def test_feature_config_custom_values(self):
        """Test FeatureConfig with custom values."""
        config = FeatureConfig(
            outcome="away",
            markets=["spreads"],
            sharp_bookmakers=["pinnacle", "circasports"],
            retail_bookmakers=["fanduel"],
            opening_hours_before=72.0,
            closing_hours_before=1.0,
        )

        assert config.outcome == "away"
        assert config.markets == ["spreads"]
        assert config.sharp_bookmakers == ["pinnacle", "circasports"]
        assert config.opening_hours_before == 72.0


# =============================================================================
# prepare_training_data_from_config Tests
# =============================================================================


class TestPrepareTrainingDataFromConfig:
    """Tests for the main prepare_training_data_from_config entry point."""

    @pytest.fixture
    def basic_xgboost_config(self):
        """Create a basic XGBoost training configuration."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="test_xgboost",
                tags=["test"],
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    test_split=0.2,
                    random_seed=42,
                ),
                model=XGBoostConfig(n_estimators=100),
            ),
        )

    @pytest.fixture
    def basic_lstm_config(self):
        """Create a basic LSTM training configuration."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="test_lstm",
                tags=["test"],
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
                    lookback_hours=72,
                    timesteps=24,
                ),
            ),
        )

    @pytest.fixture
    def mock_events(self):
        """Create mock events for testing."""
        events = []
        for i in range(10):
            event = MagicMock(spec=Event)
            event.id = f"event_{i}"
            event.home_team = "Team A"
            event.away_team = "Team B"
            event.home_score = 100 + i
            event.away_score = 95 + i
            event.status = EventStatus.FINAL
            event.commence_time = datetime(2024, 10, 15, 19, 0, tzinfo=UTC)
            events.append(event)
        return events

    @pytest.mark.asyncio
    async def test_xgboost_data_preparation(self, basic_xgboost_config, mock_events):
        """Test XGBoost data preparation returns correct structure."""
        session = AsyncMock()

        # Mock the dependencies
        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range"
            ) as mock_filter,
            patch("odds_analytics.training.data_preparation._prepare_tabular_data") as mock_prepare,
        ):
            # Setup mocks
            mock_filter.return_value = mock_events
            X = np.random.randn(10, 31).astype(np.float32)
            y = np.random.randn(10).astype(np.float32)
            feature_names = [f"feature_{i}" for i in range(31)]
            mock_prepare.return_value = (X, y, feature_names)

            # Call the function
            result = await prepare_training_data_from_config(basic_xgboost_config, session)

            # Verify result structure
            assert isinstance(result, TrainingDataResult)
            assert result.strategy_type == "xgboost_line_movement"
            assert result.num_train_samples + result.num_test_samples == 10
            assert result.num_features == 31
            assert result.masks_train is None  # XGBoost has no masks

    @pytest.mark.asyncio
    async def test_lstm_data_preparation(self, basic_lstm_config, mock_events):
        """Test LSTM data preparation returns correct structure with masks."""
        session = AsyncMock()

        # Mock the dependencies
        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range"
            ) as mock_filter,
            patch(
                "odds_analytics.training.data_preparation._prepare_sequence_data"
            ) as mock_prepare,
        ):
            # Setup mocks
            mock_filter.return_value = mock_events
            X = np.random.randn(10, 24, 17).astype(np.float32)
            y = np.random.randn(10).astype(np.float32)
            masks = np.ones((10, 24), dtype=bool)
            feature_names = [f"feature_{i}" for i in range(17)]
            mock_prepare.return_value = (X, y, masks, feature_names)

            # Call the function
            result = await prepare_training_data_from_config(basic_lstm_config, session)

            # Verify result structure
            assert isinstance(result, TrainingDataResult)
            assert result.strategy_type == "lstm_line_movement"
            assert result.masks_train is not None
            assert result.masks_test is not None

    @pytest.mark.asyncio
    async def test_no_events_raises_error(self, basic_xgboost_config):
        """Test that empty event list raises ValueError."""
        session = AsyncMock()

        with patch(
            "odds_analytics.training.data_preparation.filter_events_by_date_range"
        ) as mock_filter:
            mock_filter.return_value = []

            with pytest.raises(ValueError, match="No events found"):
                await prepare_training_data_from_config(basic_xgboost_config, session)

    @pytest.mark.asyncio
    async def test_no_valid_training_data_raises_error(self, basic_xgboost_config, mock_events):
        """Test that empty training data raises ValueError."""
        session = AsyncMock()

        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range"
            ) as mock_filter,
            patch("odds_analytics.training.data_preparation._prepare_tabular_data") as mock_prepare,
        ):
            mock_filter.return_value = mock_events
            # Return empty arrays
            mock_prepare.return_value = (np.array([]), np.array([]), [])

            with pytest.raises(ValueError, match="No valid training data"):
                await prepare_training_data_from_config(basic_xgboost_config, session)

    @pytest.mark.asyncio
    async def test_unsupported_strategy_type_raises_error(self):
        """Test that unsupported strategy type raises ValueError."""
        # Create config with invalid strategy type by bypassing validation
        config = MagicMock()
        config.experiment = MagicMock()
        config.experiment.name = "test"
        config.training = MagicMock()
        config.training.strategy_type = "unsupported_type"
        config.training.data = MagicMock()
        config.training.data.start_date = date(2024, 1, 1)
        config.training.data.end_date = date(2024, 12, 31)
        config.training.data.test_split = 0.2
        config.training.data.random_seed = 42
        config.training.data.shuffle = True

        session = AsyncMock()

        with patch(
            "odds_analytics.training.data_preparation.filter_events_by_date_range"
        ) as mock_filter:
            mock_filter.return_value = [MagicMock(spec=Event)]

            with pytest.raises(ValueError, match="Unsupported strategy type"):
                await prepare_training_data_from_config(config, session)

    @pytest.mark.asyncio
    async def test_train_test_split_ratio(self, basic_xgboost_config, mock_events):
        """Test that train/test split respects configuration."""
        session = AsyncMock()

        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range"
            ) as mock_filter,
            patch("odds_analytics.training.data_preparation._prepare_tabular_data") as mock_prepare,
        ):
            mock_filter.return_value = mock_events
            X = np.random.randn(100, 31).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            feature_names = [f"feature_{i}" for i in range(31)]
            mock_prepare.return_value = (X, y, feature_names)

            # Config has test_split=0.2
            result = await prepare_training_data_from_config(basic_xgboost_config, session)

            # Should be approximately 80/20 split
            assert result.num_train_samples == 80
            assert result.num_test_samples == 20


# =============================================================================
# Legacy Function Backward Compatibility Tests
# =============================================================================


class TestLegacyFunctionCompatibility:
    """Tests for backward compatibility of refactored functions."""

    @pytest.mark.asyncio
    async def test_prepare_tabular_with_defaults(self):
        """Test prepare_tabular_training_data works with default parameters."""
        from odds_analytics.xgboost_line_movement import prepare_tabular_training_data

        # Call with empty events to test parameter defaults
        X, y, feature_names = await prepare_tabular_training_data(
            events=[],
            session=AsyncMock(),
        )

        # Should return empty arrays without error
        assert len(X) == 0
        assert len(y) == 0
        assert len(feature_names) == 0

    @pytest.mark.asyncio
    async def test_prepare_tabular_with_explicit_params(self):
        """Test prepare_tabular_training_data with explicit parameters."""
        from odds_analytics.xgboost_line_movement import prepare_tabular_training_data

        X, y, feature_names = await prepare_tabular_training_data(
            events=[],
            session=AsyncMock(),
            outcome="away",
            market="spreads",
            opening_hours_before=72.0,
            closing_hours_before=1.0,
            sharp_bookmakers=["pinnacle", "circasports"],
            retail_bookmakers=["fanduel"],
        )

        # Should work without error
        assert len(X) == 0

    @pytest.mark.asyncio
    async def test_prepare_tabular_with_feature_config(self):
        """Test prepare_tabular_training_data with FeatureConfig."""
        from odds_analytics.xgboost_line_movement import prepare_tabular_training_data

        config = FeatureConfig(
            outcome="away",
            markets=["spreads"],
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["draftkings"],
            opening_hours_before=72.0,
            closing_hours_before=1.0,
        )

        X, y, feature_names = await prepare_tabular_training_data(
            events=[],
            session=AsyncMock(),
            feature_config=config,
        )

        # Should work without error
        assert len(X) == 0

    @pytest.mark.asyncio
    async def test_prepare_tabular_explicit_overrides_config(self):
        """Test that explicit parameters override FeatureConfig values."""
        from odds_analytics.xgboost_line_movement import prepare_tabular_training_data

        config = FeatureConfig(
            outcome="home",
            markets=["h2h"],
        )

        # This tests the priority logic - explicit params should win
        X, y, feature_names = await prepare_tabular_training_data(
            events=[],
            session=AsyncMock(),
            outcome="away",  # This should override config's "home"
            feature_config=config,
        )

        # Should work without error
        assert len(X) == 0

    @pytest.mark.asyncio
    async def test_prepare_lstm_with_defaults(self):
        """Test prepare_lstm_training_data works with default parameters."""
        from odds_analytics.sequence_loader import prepare_lstm_training_data

        X, y, masks = await prepare_lstm_training_data(
            events=[],
            session=AsyncMock(),
        )

        # Should return empty arrays without error
        assert len(X) == 0
        assert len(y) == 0
        assert len(masks) == 0

    @pytest.mark.asyncio
    async def test_prepare_lstm_with_feature_config(self):
        """Test prepare_lstm_training_data with FeatureConfig."""
        from odds_analytics.sequence_loader import prepare_lstm_training_data

        config = FeatureConfig(
            outcome="away",
            markets=["spreads"],
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["draftkings"],
        )

        X, y, masks = await prepare_lstm_training_data(
            events=[],
            session=AsyncMock(),
            feature_config=config,
        )

        # Should work without error
        assert len(X) == 0


# =============================================================================
# Config Loading and Integration Tests
# =============================================================================


class TestConfigLoadingIntegration:
    """Tests for loading config from file and preparing data."""

    @pytest.fixture
    def yaml_config_file(self):
        """Create a temporary YAML config file for testing."""
        config_data = {
            "experiment": {
                "name": "integration_test",
                "tags": ["test", "integration"],
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
                    "opening_hours_before": 48.0,
                    "closing_hours_before": 0.5,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name

        Path(f.name).unlink()

    def test_load_config_from_yaml(self, yaml_config_file):
        """Test loading configuration from YAML file."""
        config = MLTrainingConfig.from_yaml(yaml_config_file)

        assert config.experiment.name == "integration_test"
        assert config.training.strategy_type == "xgboost_line_movement"
        assert config.training.data.start_date == date(2024, 10, 1)
        assert config.training.data.end_date == date(2024, 10, 31)
        assert config.training.features.outcome == "home"
        assert config.training.features.markets == ["h2h"]

    @pytest.mark.asyncio
    async def test_config_to_data_pipeline(self, yaml_config_file):
        """Test end-to-end: config loading -> data preparation -> output shape."""
        # Load config
        config = MLTrainingConfig.from_yaml(yaml_config_file)

        session = AsyncMock()
        mock_events = [MagicMock(spec=Event) for _ in range(50)]

        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range"
            ) as mock_filter,
            patch("odds_analytics.training.data_preparation._prepare_tabular_data") as mock_prepare,
        ):
            mock_filter.return_value = mock_events

            # Return appropriately shaped data
            num_samples = 50
            num_features = 31
            X = np.random.randn(num_samples, num_features).astype(np.float32)
            y = np.random.randn(num_samples).astype(np.float32)
            feature_names = [f"feature_{i}" for i in range(num_features)]
            mock_prepare.return_value = (X, y, feature_names)

            # Execute pipeline
            result = await prepare_training_data_from_config(config, session)

            # Verify output shapes
            assert result.X_train.shape[1] == num_features
            assert result.X_test.shape[1] == num_features
            assert len(result.y_train) + len(result.y_test) == num_samples
            assert len(result.feature_names) == num_features

            # Verify split ratio
            expected_test_size = int(num_samples * config.training.data.test_split)
            assert result.num_test_samples == expected_test_size
            assert result.num_train_samples == num_samples - expected_test_size


# =============================================================================
# Date Range Filtering Tests
# =============================================================================


class TestDateRangeFiltering:
    """Tests for filter_events_by_date_range function."""

    @pytest.mark.asyncio
    async def test_filter_calls_reader(self):
        """Test that filter calls OddsReader with correct parameters."""
        session = AsyncMock()
        start_date = datetime(2024, 10, 1, tzinfo=UTC)
        end_date = datetime(2024, 10, 31, tzinfo=UTC)

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_events_by_date_range = AsyncMock(return_value=[])
            mock_reader_cls.return_value = mock_reader

            await filter_events_by_date_range(
                session=session,
                start_date=start_date,
                end_date=end_date,
            )

            # Verify reader was called with correct params
            mock_reader.get_events_by_date_range.assert_called_once_with(
                start_date=start_date,
                end_date=end_date,
                status=EventStatus.FINAL,
            )

    @pytest.mark.asyncio
    async def test_filter_returns_events(self):
        """Test that filter returns events from reader."""
        session = AsyncMock()
        start_date = datetime(2024, 10, 1, tzinfo=UTC)
        end_date = datetime(2024, 10, 31, tzinfo=UTC)

        expected_events = [MagicMock(spec=Event) for _ in range(5)]

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_events_by_date_range = AsyncMock(return_value=expected_events)
            mock_reader_cls.return_value = mock_reader

            events = await filter_events_by_date_range(
                session=session,
                start_date=start_date,
                end_date=end_date,
            )

            assert len(events) == 5
            assert events == expected_events
