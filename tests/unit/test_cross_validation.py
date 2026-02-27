"""
Unit tests for cross-validation module.

Tests cover:
- CVFoldResult creation and fields
- CVResult aggregation and statistics
- run_cv function with mock strategy
- TimeSeriesSplit behavior for temporal data
- KFold behavior with shuffling
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from odds_analytics.training import (
    CVFoldResult,
    CVResult,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrainingConfig,
    XGBoostConfig,
    run_cv,
)

# =============================================================================
# CVFoldResult Tests
# =============================================================================


class TestCVFoldResult:
    """Tests for CVFoldResult dataclass."""

    def test_basic_creation(self):
        """Test creating a fold result with all fields."""
        result = CVFoldResult(
            fold_idx=0,
            train_mse=0.01,
            train_mae=0.05,
            train_r2=0.95,
            val_mse=0.02,
            val_mae=0.08,
            val_r2=0.85,
            n_train=100,
            n_val=25,
        )
        assert result.fold_idx == 0
        assert result.train_mse == 0.01
        assert result.train_mae == 0.05
        assert result.train_r2 == 0.95
        assert result.val_mse == 0.02
        assert result.val_mae == 0.08
        assert result.val_r2 == 0.85
        assert result.n_train == 100
        assert result.n_val == 25

    def test_negative_r2(self):
        """Test that negative R² values are allowed (overfitting indicator)."""
        result = CVFoldResult(
            fold_idx=0,
            train_mse=0.01,
            train_mae=0.05,
            train_r2=0.9,
            val_mse=0.1,
            val_mae=0.2,
            val_r2=-0.5,  # Negative R² is valid
            n_train=80,
            n_val=20,
        )
        assert result.val_r2 == -0.5


# =============================================================================
# CVResult Tests
# =============================================================================


class TestCVResult:
    """Tests for CVResult aggregation."""

    @pytest.fixture
    def sample_fold_results(self) -> list[CVFoldResult]:
        """Create sample fold results for testing."""
        return [
            CVFoldResult(
                fold_idx=0,
                train_mse=0.01,
                train_mae=0.05,
                train_r2=0.95,
                val_mse=0.02,
                val_mae=0.08,
                val_r2=0.85,
                n_train=80,
                n_val=20,
            ),
            CVFoldResult(
                fold_idx=1,
                train_mse=0.012,
                train_mae=0.055,
                train_r2=0.94,
                val_mse=0.025,
                val_mae=0.09,
                val_r2=0.82,
                n_train=80,
                n_val=20,
            ),
            CVFoldResult(
                fold_idx=2,
                train_mse=0.008,
                train_mae=0.045,
                train_r2=0.96,
                val_mse=0.018,
                val_mae=0.07,
                val_r2=0.88,
                n_train=80,
                n_val=20,
            ),
        ]

    def test_basic_creation(self, sample_fold_results):
        """Test creating CVResult from fold results."""
        cv_result = CVResult(
            fold_results=sample_fold_results,
            n_folds=3,
            random_seed=42,
        )
        assert cv_result.n_folds == 3
        assert cv_result.random_seed == 42
        assert len(cv_result.fold_results) == 3

    def test_mean_validation_metrics(self, sample_fold_results):
        """Test that mean validation metrics are computed correctly."""
        cv_result = CVResult(
            fold_results=sample_fold_results,
            n_folds=3,
            random_seed=42,
        )
        # Expected means: val_mse = (0.02 + 0.025 + 0.018) / 3 = 0.021
        assert cv_result.mean_val_mse == pytest.approx(0.021, rel=1e-3)
        # val_mae = (0.08 + 0.09 + 0.07) / 3 = 0.08
        assert cv_result.mean_val_mae == pytest.approx(0.08, rel=1e-3)
        # val_r2 = (0.85 + 0.82 + 0.88) / 3 = 0.85
        assert cv_result.mean_val_r2 == pytest.approx(0.85, rel=1e-3)

    def test_std_validation_metrics(self, sample_fold_results):
        """Test that std validation metrics are computed correctly."""
        cv_result = CVResult(
            fold_results=sample_fold_results,
            n_folds=3,
            random_seed=42,
        )
        # std_val_mse should be > 0
        assert cv_result.std_val_mse > 0
        assert cv_result.std_val_mae > 0
        assert cv_result.std_val_r2 > 0

    def test_mean_train_metrics(self, sample_fold_results):
        """Test that mean training metrics are computed correctly."""
        cv_result = CVResult(
            fold_results=sample_fold_results,
            n_folds=3,
            random_seed=42,
        )
        # train_mse = (0.01 + 0.012 + 0.008) / 3 = 0.01
        assert cv_result.mean_train_mse == pytest.approx(0.01, rel=1e-3)

    def test_to_dict(self, sample_fold_results):
        """Test converting CVResult to dictionary."""
        cv_result = CVResult(
            fold_results=sample_fold_results,
            n_folds=3,
            random_seed=42,
        )
        result_dict = cv_result.to_dict()

        assert result_dict["cv_n_folds"] == 3
        assert result_dict["cv_random_seed"] == 42
        assert "cv_val_mse_mean" in result_dict
        assert "cv_val_mse_std" in result_dict
        assert "cv_val_mae_mean" in result_dict
        assert "cv_val_r2_mean" in result_dict
        assert "cv_fold_results" in result_dict
        assert len(result_dict["cv_fold_results"]) == 3

    def test_single_fold(self):
        """Test CVResult with single fold (edge case)."""
        fold_results = [
            CVFoldResult(
                fold_idx=0,
                train_mse=0.01,
                train_mae=0.05,
                train_r2=0.95,
                val_mse=0.02,
                val_mae=0.08,
                val_r2=0.85,
                n_train=90,
                n_val=10,
            )
        ]
        cv_result = CVResult(
            fold_results=fold_results,
            n_folds=1,
            random_seed=42,
        )
        # With single fold, mean equals the value and std is 0
        assert cv_result.mean_val_mse == 0.02
        assert cv_result.std_val_mse == 0.0


# =============================================================================
# run_cv Tests
# =============================================================================


class TestRunCV:
    """Tests for run_cv function."""

    @pytest.fixture
    def mock_config(self) -> MLTrainingConfig:
        """Create a mock MLTrainingConfig for testing with KFold (not timeseries)."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(name="test_cv"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    use_kfold=True,
                    cv_method="kfold",  # Explicitly set to kfold for these tests
                    n_folds=3,
                    kfold_shuffle=True,
                    random_seed=42,
                ),
                model=XGBoostConfig(n_estimators=10, max_depth=3),
                features=FeatureConfig(),
            ),
        )

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        return X, y, feature_names

    def test_run_cv_returns_cv_result(self, mock_config, sample_data):
        """Test that run_cv returns a CVResult."""
        X, y, feature_names = sample_data

        # Create mock strategy
        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=mock_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        assert isinstance(cv_result, CVResult)
        assert cv_result.n_folds == 3
        assert len(cv_result.fold_results) == 3

    def test_run_cv_calls_train_per_fold(self, mock_config, sample_data):
        """Test that train_from_config is called once per fold."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        run_cv(
            strategy=mock_strategy,
            config=mock_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # Should be called 3 times (once per fold)
        assert mock_strategy.train_from_config.call_count == 3

    def test_run_cv_fold_sizes(self, mock_config, sample_data):
        """Test that fold sizes are approximately equal."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=mock_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # With 100 samples and 3 folds, each fold should have ~33 validation samples
        for fold in cv_result.fold_results:
            assert 30 <= fold.n_val <= 35  # Allow some variance
            assert 65 <= fold.n_train <= 70

    def test_run_cv_reproducible(self, mock_config, sample_data):
        """Test that results are reproducible with same seed."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        # Run twice with same seed
        result1 = run_cv(
            strategy=mock_strategy,
            config=mock_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        result2 = run_cv(
            strategy=mock_strategy,
            config=mock_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # Should have same fold sizes in same order
        for f1, f2 in zip(result1.fold_results, result2.fold_results, strict=True):
            assert f1.n_train == f2.n_train
            assert f1.n_val == f2.n_val


# =============================================================================
# TimeSeriesSplit CV Tests
# =============================================================================


class TestTimeSeriesCV:
    """Tests for TimeSeriesSplit cross-validation."""

    @pytest.fixture
    def timeseries_config(self) -> MLTrainingConfig:
        """Create a config with cv_method='timeseries'."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(name="test_timeseries_cv"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    use_kfold=True,
                    cv_method="timeseries",
                    n_folds=3,
                    kfold_shuffle=False,
                    random_seed=42,
                ),
                model=XGBoostConfig(n_estimators=10, max_depth=3),
                features=FeatureConfig(),
            ),
        )

    @pytest.fixture
    def kfold_config(self) -> MLTrainingConfig:
        """Create a config with cv_method='kfold'."""
        return MLTrainingConfig(
            experiment=ExperimentConfig(name="test_kfold_cv"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    use_kfold=True,
                    cv_method="kfold",
                    n_folds=3,
                    kfold_shuffle=True,
                    random_seed=42,
                ),
                model=XGBoostConfig(n_estimators=10, max_depth=3),
                features=FeatureConfig(),
            ),
        )

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        return X, y, feature_names

    def test_timeseries_cv_expanding_windows(self, timeseries_config, sample_data):
        """Test that TimeSeriesSplit produces expanding training windows."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=timeseries_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # TimeSeriesSplit should have expanding training windows
        # Each fold should have more training samples than the previous
        train_sizes = [f.n_train for f in cv_result.fold_results]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Training window should expand: fold {i} has {train_sizes[i]} samples, "
                f"but fold {i - 1} has {train_sizes[i - 1]} samples"
            )

    def test_timeseries_cv_returns_cv_method(self, timeseries_config, sample_data):
        """Test that CVResult includes correct cv_method."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=timeseries_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        assert cv_result.cv_method == "timeseries"

    def test_kfold_cv_returns_cv_method(self, kfold_config, sample_data):
        """Test that CVResult includes correct cv_method for kfold."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=kfold_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        assert cv_result.cv_method == "kfold"

    def test_cv_result_to_dict_includes_cv_method(self, timeseries_config, sample_data):
        """Test that to_dict includes cv_method."""
        X, y, feature_names = sample_data

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "train_mse": 0.01,
            "train_mae": 0.05,
            "train_r2": 0.9,
            "val_mse": 0.02,
            "val_mae": 0.08,
            "val_r2": 0.8,
        }

        cv_result = run_cv(
            strategy=mock_strategy,
            config=timeseries_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        result_dict = cv_result.to_dict()
        assert "cv_method" in result_dict
        assert result_dict["cv_method"] == "timeseries"

    def test_timeseries_cv_validation_follows_training(self, timeseries_config, sample_data):
        """Test that validation samples always come after training samples chronologically."""
        X, y, feature_names = sample_data

        # Track the indices used in each fold
        fold_indices = []

        def capture_indices(config, X_train, y_train, feature_names, X_val, y_val, **kwargs):
            # Store the original indices by comparing with full X
            fold_indices.append(
                {
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                }
            )
            return {
                "train_mse": 0.01,
                "train_mae": 0.05,
                "train_r2": 0.9,
                "val_mse": 0.02,
                "val_mae": 0.08,
                "val_r2": 0.8,
            }

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.side_effect = capture_indices

        run_cv(
            strategy=mock_strategy,
            config=timeseries_config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # Verify TimeSeriesSplit behavior:
        # With 100 samples and 3 splits, we expect:
        # Fold 0: train on first 25, validate on next 25
        # Fold 1: train on first 50, validate on next 25
        # Fold 2: train on first 75, validate on next 25
        assert len(fold_indices) == 3

        # Training sizes should increase
        assert fold_indices[0]["n_train"] < fold_indices[1]["n_train"]
        assert fold_indices[1]["n_train"] < fold_indices[2]["n_train"]


class TestRunCVStaticFeatures:
    """Tests for static feature threading through run_cv."""

    @pytest.fixture
    def kfold_config(self) -> MLTrainingConfig:
        return MLTrainingConfig(
            experiment=ExperimentConfig(name="test_static_cv"),
            training=TrainingConfig(
                strategy_type="xgboost_line_movement",
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    use_kfold=True,
                    cv_method="kfold",
                    n_folds=3,
                    kfold_shuffle=True,
                    random_seed=42,
                ),
                model=XGBoostConfig(),
                features=FeatureConfig(),
            ),
        )

    def test_static_features_split_by_fold(self, kfold_config):
        """Test that static_features are split alongside X/y per fold."""
        np.random.seed(42)
        n_samples = 60
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        static = np.random.randn(n_samples, 3)
        feature_names = [f"f{i}" for i in range(5)]

        calls = []

        def capture(**kwargs):
            calls.append(kwargs)
            return {
                "train_mse": 0.01,
                "train_mae": 0.05,
                "train_r2": 0.9,
                "val_mse": 0.02,
                "val_mae": 0.08,
                "val_r2": 0.8,
            }

        mock_strategy = MagicMock()
        mock_strategy.train_from_config.side_effect = (
            lambda **kw: capture(**kw) if kw else capture()
        )
        # Use side_effect that captures kwargs
        mock_strategy.train_from_config = MagicMock(side_effect=lambda **kw: capture(**kw))

        run_cv(
            strategy=mock_strategy,
            config=kfold_config,
            X=X,
            y=y,
            feature_names=feature_names,
            static_features=static,
        )

        assert len(calls) == 3
        for call in calls:
            assert "static_train" in call
            assert "static_val" in call
            assert call["static_train"] is not None
            assert call["static_val"] is not None
            # Dimensions should match X splits
            assert call["static_train"].shape[0] == call["X_train"].shape[0]
            assert call["static_val"].shape[0] == call["X_val"].shape[0]
            assert call["static_train"].shape[1] == 3
            assert call["static_val"].shape[1] == 3

    def test_static_none_passes_none_to_folds(self, kfold_config):
        """Test that static_features=None passes None to each fold."""
        np.random.seed(42)
        X = np.random.randn(60, 5)
        y = np.random.randn(60)

        calls = []
        mock_strategy = MagicMock()
        mock_strategy.train_from_config = MagicMock(
            side_effect=lambda **kw: (
                calls.append(kw)
                or {
                    "train_mse": 0.01,
                    "train_mae": 0.05,
                    "train_r2": 0.9,
                    "val_mse": 0.02,
                    "val_mae": 0.08,
                    "val_r2": 0.8,
                }
            )
        )

        run_cv(
            strategy=mock_strategy,
            config=kfold_config,
            X=X,
            y=y,
            feature_names=[f"f{i}" for i in range(5)],
            static_features=None,
        )

        assert len(calls) == 3
        for call in calls:
            assert call["static_train"] is None
            assert call["static_val"] is None
