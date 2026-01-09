"""
Integration test for hyperparameter tuning with MLflow tracking.

Tests the complete workflow of Optuna hyperparameter tuning with MLflow
experiment tracking, including parent-child run creation, parameter logging,
and model persistence.
"""

import tempfile
from pathlib import Path

import mlflow
import pytest
from odds_analytics.training import MLTrainingConfig, TrackingConfig
from odds_analytics.training.tuner import OptunaTuner, create_objective


class TestTuningTrackingIntegration:
    """Integration tests for hyperparameter tuning with MLflow tracking."""

    @pytest.fixture
    def minimal_tuning_config_path(self, tmp_path):
        """Create a minimal tuning configuration file for testing."""
        config_content = """
experiment:
  name: "test_tuning_tracking"
  description: "Test hyperparameter tuning with MLflow tracking"
  tags: ["test", "tuning", "mlflow"]

training:
  strategy_type: "xgboost_line_movement"

  data:
    start_date: "2024-11-01"
    end_date: "2024-11-15"
    test_split: 0.2
    random_seed: 42

  model:
    n_estimators: 50
    max_depth: 3
    learning_rate: 0.1
    random_state: 42

  features:
    sharp_bookmakers: ["pinnacle"]
    retail_bookmakers: ["fanduel"]
    markets: ["h2h"]
    outcome: "home"

  output_path: "models"

tuning:
  n_trials: 3
  direction: "minimize"
  metric: "val_mse"
  sampler: "tpe"
  pruner: "median"
  search_spaces:
    max_depth:
      type: int
      low: 2
      high: 5
      step: 1
    learning_rate:
      type: float
      low: 0.05
      high: 0.2
      log: false

tracking:
  enabled: true
  tracking_uri: "mlruns"
  experiment_name: "test_tuning_tracking"
"""
        config_path = tmp_path / "test_tuning_config.yaml"
        config_path.write_text(config_content)
        return config_path

    @pytest.mark.asyncio
    async def test_tuning_with_tracking_creates_parent_and_child_runs(
        self, minimal_tuning_config_path, sample_training_data
    ):
        """Test that tuning with tracking creates parent run and nested child runs."""
        # Load configuration
        config = MLTrainingConfig.from_yaml(minimal_tuning_config_path)

        # Create temporary tracking directory
        with tempfile.TemporaryDirectory() as tmp_tracking_dir:
            # Update tracking URI to temporary directory
            config.tracking.tracking_uri = tmp_tracking_dir
            mlflow.set_tracking_uri(tmp_tracking_dir)

            # Create tuner with tracking
            tuner = OptunaTuner(
                study_name="test_parent_child_runs",
                direction=config.tuning.direction,
                sampler=config.tuning.sampler,
                pruner=config.tuning.pruner,
                tracking_config=config.tracking,
            )

            # Create objective function with sample data
            X_train, y_train, feature_names = sample_training_data
            objective = create_objective(
                config=config,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_train[:20],  # Small validation set
                y_val=y_train[:20],
            )

            # Run optimization
            study = tuner.optimize(objective, n_trials=3)

            # Verify study completed
            assert study is not None
            assert len(study.trials) == 3

            # Verify parent run was created
            assert tuner._parent_run_id is not None

            # Get experiment and verify runs
            experiment = mlflow.get_experiment_by_name(config.tracking.experiment_name)
            assert experiment is not None

            # Get all runs for the experiment
            all_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                output_format="list",
            )

            # Should have 1 parent run + 3 child runs = 4 total
            assert len(all_runs) >= 4, f"Expected at least 4 runs, got {len(all_runs)}"

            # Find parent run
            parent_runs = [
                r for r in all_runs if r.data.tags.get("type") == "hyperparameter_tuning"
            ]
            assert len(parent_runs) == 1, "Should have exactly one parent run"
            parent_run = parent_runs[0]

            # Verify parent run has study metadata
            assert "study_name" in parent_run.data.params
            assert "direction" in parent_run.data.params
            assert "sampler" in parent_run.data.params
            assert "pruner" in parent_run.data.params

            # Verify parent run has best trial metrics
            assert "best_value" in parent_run.data.metrics
            assert "n_completed_trials" in parent_run.data.metrics

            # Verify parent run has best parameters
            assert any(k.startswith("best_") for k in parent_run.data.params.keys())

            # Find child runs (nested under parent)
            child_runs = [
                r
                for r in all_runs
                if r.data.tags.get("mlflow.parentRunId") == parent_run.info.run_id
            ]
            assert len(child_runs) == 3, f"Expected 3 child runs, got {len(child_runs)}"

    @pytest.mark.asyncio
    async def test_tuning_without_tracking_works(
        self, minimal_tuning_config_path, sample_training_data
    ):
        """Test that tuning works without tracking enabled."""
        # Load configuration
        config = MLTrainingConfig.from_yaml(minimal_tuning_config_path)

        # Disable tracking
        config.tracking.enabled = False

        # Create tuner without tracking
        tuner = OptunaTuner(
            study_name="test_no_tracking",
            direction=config.tuning.direction,
            sampler=config.tuning.sampler,
            pruner=config.tuning.pruner,
            tracking_config=None,  # No tracking
        )

        # Create objective function with sample data
        X_train, y_train, feature_names = sample_training_data
        objective = create_objective(
            config=config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_train[:20],  # Small validation set
            y_val=y_train[:20],
        )

        # Run optimization
        study = tuner.optimize(objective, n_trials=2)

        # Verify study completed without errors
        assert study is not None
        assert len(study.trials) == 2

        # Verify no tracking artifacts created
        assert tuner._tracker is None
        assert tuner._parent_run_id is None

    @pytest.mark.asyncio
    async def test_log_best_model_to_parent_run(
        self, minimal_tuning_config_path, sample_training_data
    ):
        """Test that the best model can be logged to the parent run."""
        # Load configuration
        config = MLTrainingConfig.from_yaml(minimal_tuning_config_path)

        # Create temporary tracking directory
        with tempfile.TemporaryDirectory() as tmp_tracking_dir:
            # Update tracking URI to temporary directory
            config.tracking.tracking_uri = tmp_tracking_dir
            mlflow.set_tracking_uri(tmp_tracking_dir)

            # Create tuner with tracking
            tuner = OptunaTuner(
                study_name="test_model_logging",
                direction=config.tuning.direction,
                sampler=config.tuning.sampler,
                pruner=config.tuning.pruner,
                tracking_config=config.tracking,
            )

            # Create objective function with sample data
            X_train, y_train, feature_names = sample_training_data
            objective = create_objective(
                config=config,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_train[:20],
                y_val=y_train[:20],
            )

            # Run optimization
            study = tuner.optimize(objective, n_trials=2)

            # Train a simple model with best parameters
            from xgboost import XGBRegressor

            best_params = study.best_params
            model = XGBRegressor(
                **{k: v for k, v in best_params.items() if k in ["max_depth", "learning_rate"]},
                n_estimators=10,
            )
            model.fit(X_train, y_train)

            # Log best model to parent run (should not raise an exception)
            try:
                tuner.log_best_model(model, artifact_path="best_model")
            except Exception as e:
                pytest.fail(f"log_best_model raised an exception: {e}")

            # Verify that the method completed successfully
            # (The model is logged to a reactivated parent run)
            assert tuner._parent_run_id is not None

    @pytest.mark.asyncio
    async def test_tracking_with_disabled_config(
        self, minimal_tuning_config_path, sample_training_data
    ):
        """Test that tracking_config=None works correctly."""
        # Load configuration
        config = MLTrainingConfig.from_yaml(minimal_tuning_config_path)

        # Create tuner with tracking_config=None (explicitly disabled)
        tuner = OptunaTuner(
            study_name="test_tracking_disabled",
            direction=config.tuning.direction,
            sampler=config.tuning.sampler,
            pruner=config.tuning.pruner,
            tracking_config=None,  # Explicitly disabled
        )

        # Create objective function
        X_train, y_train, feature_names = sample_training_data
        objective = create_objective(
            config=config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_train[:20],
            y_val=y_train[:20],
        )

        # Run optimization (should succeed without tracking)
        study = tuner.optimize(objective, n_trials=2)

        # Verify study completed
        assert study is not None
        assert len(study.trials) == 2

        # Verify tracking was not initialized
        assert tuner._tracker is None
        assert tuner._parent_run_id is None

    @pytest.mark.asyncio
    async def test_log_best_model_raises_without_tracking(
        self, minimal_tuning_config_path, sample_training_data
    ):
        """Test that log_best_model raises ValueError when tracking is not enabled."""
        # Load configuration
        config = MLTrainingConfig.from_yaml(minimal_tuning_config_path)

        # Create tuner without tracking
        tuner = OptunaTuner(
            study_name="test_no_tracking_model_log",
            direction=config.tuning.direction,
            sampler=config.tuning.sampler,
            pruner=config.tuning.pruner,
            tracking_config=None,  # No tracking
        )

        # Create objective and run optimization
        X_train, y_train, feature_names = sample_training_data
        objective = create_objective(
            config=config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_train[:20],
            y_val=y_train[:20],
        )
        study = tuner.optimize(objective, n_trials=2)

        # Create a simple model
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=10)
        model.fit(X_train, y_train)

        # Attempt to log model should raise ValueError
        with pytest.raises(
            ValueError,
            match="Cannot log model: tracking is not enabled or parent run doesn't exist",
        ):
            tuner.log_best_model(model)

    @pytest.fixture
    def sample_training_data(self):
        """Create minimal sample training data for testing."""
        import numpy as np

        # Create small synthetic dataset
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randn(n_samples).astype(np.float32)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X_train, y_train, feature_names
