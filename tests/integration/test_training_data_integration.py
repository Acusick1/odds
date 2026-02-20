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
from datetime import UTC, date, datetime, timedelta
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
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.injury_models import InjuryReport, InjuryStatus
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
                    adapter="lstm",
                    outcome="home",
                    markets=["h2h"],
                    sharp_bookmakers=["pinnacle"],
                    retail_bookmakers=["fanduel", "draftkings"],
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

    @pytest.mark.asyncio
    async def test_variance_filter_removes_constant_features(
        self, xgboost_config, pglite_async_session, test_events_with_odds
    ):
        """Constant features are absent from the pipeline output."""
        import numpy as np

        result = await prepare_training_data_from_config(xgboost_config, pglite_async_session)

        X_all = np.concatenate([result.X_train, result.X_test], axis=0)
        variances = np.var(X_all, axis=0)

        constant = [
            name for name, var in zip(result.feature_names, variances, strict=False) if var == 0.0
        ]
        assert constant == [], f"Constant features still present: {constant}"


class TestInjuryFeaturesIntegration:
    """Integration tests for injury features in the training pipeline."""

    @pytest.fixture
    def xgboost_with_injuries_config(self):
        """XGBoost config with tabular + injuries feature groups."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="integration_test_injuries",
                tags=["integration", "injuries"],
                description="Integration test for injury features",
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
                    closing_tier=FetchTier.CLOSING,
                    feature_groups=("tabular", "injuries"),
                ),
            ),
        )

    @pytest.fixture
    async def test_events_with_injury_reports(self, pglite_async_session, test_events_with_odds):
        """Add injury reports to a subset of test events."""
        events = test_events_with_odds
        base_time = datetime(2024, 10, 15, 19, 0, tzinfo=UTC)

        # Add injury reports for the first 3 events only
        for i in range(3):
            commence_time = base_time + timedelta(days=i)
            report_time = commence_time - timedelta(hours=10)

            # Two reports per event (one per team)
            report_home = InjuryReport(
                report_time=report_time,
                game_date=commence_time.date(),
                game_time_et="07:00 PM ET",
                matchup=f"Away Team {i}@Home Team {i}",
                team=f"Home Team {i}",
                player_name=f"Player, Home{i}",
                status=InjuryStatus.OUT,
                reason="Left Knee; Sprain",
                event_id=f"test_event_{i}",
            )
            report_away = InjuryReport(
                report_time=report_time,
                game_date=commence_time.date(),
                game_time_et="07:00 PM ET",
                matchup=f"Away Team {i}@Home Team {i}",
                team=f"Away Team {i}",
                player_name=f"Player, Away{i}",
                status=InjuryStatus.QUESTIONABLE,
                reason="Right Ankle; Soreness",
                event_id=f"test_event_{i}",
            )
            pglite_async_session.add(report_home)
            pglite_async_session.add(report_away)

        await pglite_async_session.commit()
        return events

    @pytest.mark.asyncio
    async def test_pipeline_with_injury_features(
        self,
        xgboost_with_injuries_config,
        pglite_async_session,
        test_events_with_injury_reports,
    ):
        """Full pipeline with injuries produces valid training data."""

        result = await prepare_training_data_from_config(
            xgboost_with_injuries_config, pglite_async_session
        )

        assert isinstance(result, TrainingDataResult)
        assert result.num_train_samples > 0
        assert result.num_test_samples > 0

        # Verify at least some injury features survive variance filter.
        # Impact features (impact_out_*, impact_gtd_*) are constant with fake team names,
        # so the variance filter drops them. Timing features vary per event.
        inj_features = [n for n in result.feature_names if n.startswith("inj_")]
        assert len(inj_features) > 0, "No injury features survived variance filter"
        assert "inj_report_hours_before_game" in result.feature_names
        assert "inj_injury_news_recency" in result.feature_names

        assert result.X_train.shape[1] == len(result.feature_names)

    @pytest.mark.asyncio
    async def test_events_without_injuries_not_dropped(
        self,
        xgboost_with_injuries_config,
        pglite_async_session,
        test_events_with_injury_reports,
    ):
        """Events without injury data still produce training rows (NaN-filled)."""
        # Run pipeline — events 3-7 have no injury reports
        result = await prepare_training_data_from_config(
            xgboost_with_injuries_config, pglite_async_session
        )

        # All 8 events should contribute rows (not just the 3 with injury data).
        # Total includes train + val + test (val split removes some from train+test).
        total_samples = result.num_train_samples + result.num_test_samples
        if result.num_val_samples is not None:
            total_samples += result.num_val_samples
        assert total_samples >= 8  # At least 1 row per event


class TestRestFeaturesIntegration:
    """Integration tests for rest/schedule features in the training pipeline."""

    @pytest.fixture
    def xgboost_with_rest_config(self):
        """XGBoost config with tabular + rest feature groups."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="integration_test_rest",
                tags=["integration", "rest"],
                description="Integration test for rest/schedule features",
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
                    closing_tier=FetchTier.CLOSING,
                    feature_groups=("tabular", "rest"),
                ),
            ),
        )

    @pytest.fixture
    async def test_events_with_game_logs(self, pglite_async_session, test_events_with_odds):
        """Add game log records for a subset of test events."""
        events = test_events_with_odds
        base_time = datetime(2024, 10, 15, 19, 0, tzinfo=UTC)

        # Add game logs for the first 5 events with prior games
        for i in range(5):
            commence_time = base_time + timedelta(days=i)
            game_date = commence_time.date()

            # Game logs for the event itself (2 rows: home + away)
            home_log = NbaTeamGameLog(
                nba_game_id=f"002240010{i}",
                team_id=1610612740 + i,
                team_abbreviation=f"HM{i}",
                game_date=game_date,
                matchup=f"HM{i} vs. AW{i}",
                season="2024-25",
                event_id=f"test_event_{i}",
            )
            away_log = NbaTeamGameLog(
                nba_game_id=f"002240010{i}",
                team_id=1610612750 + i,
                team_abbreviation=f"AW{i}",
                game_date=game_date,
                matchup=f"AW{i} @ HM{i}",
                season="2024-25",
                event_id=f"test_event_{i}",
            )
            pglite_async_session.add(home_log)
            pglite_async_session.add(away_log)

            # Prior game for home team (2 days before → 2 days rest)
            prev_home = NbaTeamGameLog(
                nba_game_id=f"002240009{i}",
                team_id=1610612740 + i,
                team_abbreviation=f"HM{i}",
                game_date=game_date - timedelta(days=2),
                matchup=f"HM{i} vs. OPP",
                season="2024-25",
                event_id=None,
            )
            pglite_async_session.add(prev_home)

            # Prior game for away team (varies: 1 day for even i, 3 days for odd i)
            away_rest_days = 1 if i % 2 == 0 else 3
            prev_away = NbaTeamGameLog(
                nba_game_id=f"002240008{i}",
                team_id=1610612750 + i,
                team_abbreviation=f"AW{i}",
                game_date=game_date - timedelta(days=away_rest_days),
                matchup=f"AW{i} @ OPP",
                season="2024-25",
                event_id=None,
            )
            pglite_async_session.add(prev_away)

        await pglite_async_session.commit()
        return events

    @pytest.mark.asyncio
    async def test_pipeline_with_rest_features(
        self,
        xgboost_with_rest_config,
        pglite_async_session,
        test_events_with_game_logs,
    ):
        """Full pipeline with rest features produces valid training data."""
        result = await prepare_training_data_from_config(
            xgboost_with_rest_config, pglite_async_session
        )

        assert isinstance(result, TrainingDataResult)
        assert result.num_train_samples > 0
        assert result.num_test_samples > 0

        # Verify rest features survive variance filter.
        # home_days_rest is constant (2 for all events with logs), but
        # away_days_rest varies (1 or 3), so rest_advantage and away features vary.
        rest_features = [n for n in result.feature_names if n.startswith("rest_")]
        assert len(rest_features) > 0, "No rest features survived variance filter"

        assert result.X_train.shape[1] == len(result.feature_names)

    @pytest.mark.asyncio
    async def test_events_without_game_logs_not_dropped(
        self,
        xgboost_with_rest_config,
        pglite_async_session,
        test_events_with_game_logs,
    ):
        """Events without game log data still produce training rows (NaN-filled)."""
        result = await prepare_training_data_from_config(
            xgboost_with_rest_config, pglite_async_session
        )

        # All 8 events should contribute rows (not just the 5 with game logs).
        total_samples = result.num_train_samples + result.num_test_samples
        if result.num_val_samples is not None:
            total_samples += result.num_val_samples
        assert total_samples >= 8  # At least 1 row per event
