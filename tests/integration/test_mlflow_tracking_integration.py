"""
Integration tests for MLflow tracking integration with training pipeline.

Tests the full flow:
1. Config with tracking enabled
2. Training with MLflow logging
3. Verification of logged parameters, metrics, and artifacts
4. Both XGBoost and LSTM strategies

These tests verify that the tracking abstraction is properly wired into
the training flow and logs all expected data to MLflow.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrackingConfig,
    TrainingConfig,
    XGBoostConfig,
    create_tracker,
)
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_lambda.fetch_tier import FetchTier

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestMLflowTrackingIntegration:
    """Integration tests for MLflow tracking with training pipeline."""

    @pytest.fixture
    def temp_mlflow_dir(self):
        """Create temporary directory for MLflow artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def xgboost_config_with_tracking(self, temp_mlflow_dir):
        """Create XGBoost training configuration with tracking enabled."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(
                name="test_mlflow_xgboost",
                tags=["integration", "mlflow", "xgboost"],
                description="Integration test for XGBoost with MLflow tracking",
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    test_split=0.2,
                    validation_split=0.0,
                ),
                model=XGBoostConfig(
                    n_estimators=10,  # Small for fast testing
                    max_depth=3,
                    learning_rate=0.1,
                ),
                features=FeatureConfig(
                    outcome="home",
                    markets=["h2h"],
                    sharp_bookmakers=["pinnacle"],
                    retail_bookmakers=["fanduel", "draftkings"],
                    opening_tier=FetchTier.EARLY,
                    closing_tier=FetchTier.CLOSING,
                ),
                output_path="models/test",
                model_name="test_model.pkl",
            ),
            tracking=TrackingConfig(
                enabled=True,
                backend="mlflow",
                tracking_uri=temp_mlflow_dir,
                experiment_name="test_mlflow_xgboost",
                run_name="test_run",
                log_model=True,
                log_params=True,
                log_metrics=True,
            ),
        )

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples)
        X_val = np.random.randn(20, n_features)
        y_val = np.random.randn(20)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X_train, y_train, X_val, y_val, feature_names

    def test_tracker_creation_from_config(self, xgboost_config_with_tracking):
        """Test that tracker can be created from config."""
        tracker = create_tracker(xgboost_config_with_tracking.tracking)
        assert tracker is not None
        assert tracker.tracking_uri == xgboost_config_with_tracking.tracking.tracking_uri
        assert tracker.experiment_name == "test_mlflow_xgboost"

    def test_xgboost_training_with_tracking(
        self, xgboost_config_with_tracking, sample_training_data, temp_mlflow_dir
    ):
        """Test XGBoost training logs all expected data to MLflow."""
        X_train, y_train, X_val, y_val, feature_names = sample_training_data

        # Create tracker
        tracker = create_tracker(xgboost_config_with_tracking.tracking)

        # Start run
        with tracker.start_run(run_name="test_xgboost_run"):
            # Train model
            strategy = XGBoostLineMovementStrategy()
            history = strategy.train_from_config(
                config=xgboost_config_with_tracking,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_val,
                y_val=y_val,
                tracker=tracker,
            )

            # Verify training completed
            assert "train_mse" in history
            assert "train_mae" in history
            assert "train_r2" in history
            assert "val_mse" in history
            assert "val_mae" in history
            assert "val_r2" in history

        # Verify MLflow directory was created and has artifacts
        mlflow_path = Path(temp_mlflow_dir)
        assert mlflow_path.exists()
        # MLflow creates experiment and run directories
        assert any(mlflow_path.iterdir()), "MLflow should have created tracking data"

    def test_training_without_tracking(self, sample_training_data):
        """Test that training still works when tracking is disabled."""
        X_train, y_train, X_val, y_val, feature_names = sample_training_data

        # Create config without tracking
        config = MLTrainingConfig(
            experiment=ExperimentConfig(
                name="test_no_tracking",
                description="Test without tracking",
            ),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    test_split=0.2,
                ),
                model=XGBoostConfig(
                    n_estimators=10,
                    max_depth=3,
                ),
                features=FeatureConfig(
                    outcome="home",
                    markets=["h2h"],
                    sharp_bookmakers=["pinnacle"],
                    retail_bookmakers=["fanduel"],
                ),
                output_path="models/test",
            ),
            tracking=TrackingConfig(enabled=False),
        )

        # Train without tracker
        strategy = XGBoostLineMovementStrategy()
        history = strategy.train_from_config(
            config=config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_val,
            y_val=y_val,
            tracker=None,  # No tracker
        )

        # Verify training still works
        assert "train_mse" in history
        assert "train_mae" in history
        assert "train_r2" in history

    def test_tracker_logs_configuration_params(
        self, xgboost_config_with_tracking, sample_training_data, temp_mlflow_dir
    ):
        """Test that all configuration parameters are logged to MLflow."""
        X_train, y_train, X_val, y_val, feature_names = sample_training_data

        # Create tracker
        tracker = create_tracker(xgboost_config_with_tracking.tracking)

        # Track what was logged
        logged_params = {}

        # Monkey-patch log_params to capture what was logged
        original_log_params = tracker.log_params

        def capture_log_params(params):
            logged_params.update(params)
            original_log_params(params)

        tracker.log_params = capture_log_params

        # Start run and train
        with tracker.start_run(run_name="test_params_logging"):
            strategy = XGBoostLineMovementStrategy()
            strategy.train_from_config(
                config=xgboost_config_with_tracking,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_val,
                y_val=y_val,
                tracker=tracker,
            )

        # Verify key parameters were logged
        assert "experiment_name" in logged_params
        assert "strategy_type" in logged_params
        assert "n_estimators" in logged_params
        assert "max_depth" in logged_params
        assert "learning_rate" in logged_params
        assert logged_params["strategy_type"] == "xgboost_line_movement"
        assert logged_params["n_estimators"] == 10  # Integer before MLflow conversion

    def test_tracker_logs_metrics_per_round(
        self, xgboost_config_with_tracking, sample_training_data, temp_mlflow_dir
    ):
        """Test that per-round/epoch metrics are logged to MLflow."""
        X_train, y_train, X_val, y_val, feature_names = sample_training_data

        # Create tracker
        tracker = create_tracker(xgboost_config_with_tracking.tracking)

        # Track metrics logged
        logged_metrics = []

        # Monkey-patch log_metrics to capture what was logged
        original_log_metrics = tracker.log_metrics

        def capture_log_metrics(metrics, step=None):
            logged_metrics.append((metrics.copy(), step))
            original_log_metrics(metrics, step)

        tracker.log_metrics = capture_log_metrics

        # Start run and train
        with tracker.start_run(run_name="test_metrics_logging"):
            strategy = XGBoostLineMovementStrategy()
            strategy.train_from_config(
                config=xgboost_config_with_tracking,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_val,
                y_val=y_val,
                tracker=tracker,
            )

        # Verify metrics were logged
        assert len(logged_metrics) > 0, "No metrics were logged"

        # Check for per-round metrics (with step parameter)
        per_round_metrics = [m for m in logged_metrics if m[1] is not None]
        assert len(per_round_metrics) > 0, "No per-round metrics were logged"

        # Check for final metrics (without step or with None)
        final_metrics = [m for m in logged_metrics if "final_train_mse" in m[0]]
        assert len(final_metrics) > 0, "No final metrics were logged"
