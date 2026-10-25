"""Unit tests for XGBoost Line Movement Predictor strategy."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy


class TestXGBoostLineMovementStrategy:
    """Test XGBoost Line Movement strategy class."""

    def test_strategy_initialization(self):
        strategy = XGBoostLineMovementStrategy()

        assert strategy.model is None
        assert strategy.feature_names == []

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
        movement = float(strategy.predict(test_features.reshape(1, -1))[0])

        assert isinstance(movement, float)
        # Movement should be in reasonable range for probability deltas
        assert -1 <= movement <= 1

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_save_and_load_model(self):
        """Test that model can be saved and loaded."""
        strategy = XGBoostLineMovementStrategy()

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

            # Predictions should be identical
            test_features = np.random.rand(5)
            original_movement = float(strategy.predict(test_features.reshape(1, -1))[0])
            loaded_movement = float(loaded_strategy.predict(test_features.reshape(1, -1))[0])

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


class TestModelPersistence:
    """Test model persistence workflow."""

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_complete_save_load_workflow(self):
        """Test complete save/load workflow with all parameters."""
        # Create and train model
        strategy = XGBoostLineMovementStrategy()

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

            # Verify model works
            test_vec = np.random.rand(15)
            movement1 = float(strategy.predict(test_vec.reshape(1, -1))[0])
            movement2 = float(new_strategy.predict(test_vec.reshape(1, -1))[0])

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

        strategy = XGBoostLineMovementStrategy()

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
            assert config_data["feature_names"] == feature_names
            assert config_data["n_features"] == 5

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_load_model_with_config(self):
        """Test that load_model logs config information when available."""
        strategy = XGBoostLineMovementStrategy()

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

            assert loaded_strategy.feature_names == feature_names

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="xgboost not installed"),
        reason="requires xgboost",
    )
    def test_load_model_backward_compatibility_pickle(self):
        """Test that load_model can still load old pickle format models."""
        import pickle

        strategy = XGBoostLineMovementStrategy()

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
            }
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            # Load into new strategy
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            # Verify model loaded correctly
            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names

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
            }
            joblib.dump(model_data, model_path)

            # Load - should work without config file
            loaded_strategy = XGBoostLineMovementStrategy()
            loaded_strategy.load_model(str(model_path))

            assert loaded_strategy.model is not None
            assert loaded_strategy.feature_names == feature_names
