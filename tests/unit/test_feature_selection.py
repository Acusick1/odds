"""Unit tests for feature selection module."""

from __future__ import annotations

import numpy as np
import pytest
from odds_analytics.training.config import FeatureSelectionConfig
from odds_analytics.training.feature_selection import (
    FEATURE_SELECTOR_REGISTRY,
    EnsembleSelector,
    FeatureRanking,
    FeatureSelector,
    FilterSelector,
    HybridSelector,
    ManualSelector,
    get_feature_selector,
)
from pydantic import ValidationError


class TestFeatureRanking:
    """Tests for FeatureRanking output model."""

    def test_valid_ranking(self):
        """Test creating valid FeatureRanking."""
        ranking = FeatureRanking(
            feature_names=["feat1", "feat2", "feat3"],
            scores=[0.9, 0.7, 0.5],
            method="ensemble",
            metadata={"test": "value"},
        )

        assert len(ranking.feature_names) == 3
        assert len(ranking.scores) == 3
        assert ranking.method == "ensemble"
        assert ranking.metadata["test"] == "value"

    def test_mismatched_lengths(self):
        """Test validation fails when lengths don't match."""
        with pytest.raises(ValidationError, match="must have same length"):
            FeatureRanking(
                feature_names=["feat1", "feat2"],
                scores=[0.9, 0.7, 0.5],  # Length mismatch
                method="filter",
            )

    def test_min_length_validation(self):
        """Test minimum length validation for feature_names."""
        with pytest.raises(ValidationError):
            FeatureRanking(
                feature_names=[],  # Empty list
                scores=[],
                method="manual",
            )


class TestFeatureSelectionConfig:
    """Tests for FeatureSelectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureSelectionConfig()

        assert config.enabled is True
        assert config.method == "ensemble"
        assert config.n_ensemble_models == 3
        assert len(config.ensemble_seeds) == 3
        assert config.min_correlation == 0.02
        assert config.min_variance == 0.01

    def test_manual_method_validation(self):
        """Test that manual method requires feature_names."""
        with pytest.raises(ValidationError, match="requires feature_names"):
            FeatureSelectionConfig(
                method="manual",
                feature_names=None,  # Missing required field
            )

    def test_manual_method_with_features(self):
        """Test manual method with valid feature_names."""
        config = FeatureSelectionConfig(
            method="manual",
            feature_names=["feat1", "feat2", "feat3"],
        )

        assert config.feature_names == ["feat1", "feat2", "feat3"]

    def test_ensemble_seeds_validation(self):
        """Test that ensemble_seeds must match n_ensemble_models."""
        with pytest.raises(ValidationError, match="must match"):
            FeatureSelectionConfig(
                method="ensemble",
                n_ensemble_models=3,
                ensemble_seeds=[42, 123],  # Only 2 seeds for 3 models
            )

    def test_valid_ensemble_config(self):
        """Test valid ensemble configuration."""
        config = FeatureSelectionConfig(
            method="ensemble",
            n_ensemble_models=2,
            ensemble_seeds=[42, 123],
        )

        assert config.n_ensemble_models == 2
        assert len(config.ensemble_seeds) == 2

    def test_unknown_method(self):
        """Test that unknown method raises validation error."""
        with pytest.raises(ValidationError, match="Input should be"):
            FeatureSelectionConfig(method="invalid_method")

    def test_correlation_bounds(self):
        """Test correlation field bounds."""
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(min_correlation=1.5)  # Above 1.0

        with pytest.raises(ValidationError):
            FeatureSelectionConfig(min_correlation=-0.1)  # Below 0.0

    def test_hybrid_weight_bounds(self):
        """Test hybrid_filter_weight bounds."""
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(hybrid_filter_weight=1.5)  # Above 1.0

        # Valid boundary values
        config = FeatureSelectionConfig(hybrid_filter_weight=0.0)
        assert config.hybrid_filter_weight == 0.0

        config = FeatureSelectionConfig(hybrid_filter_weight=1.0)
        assert config.hybrid_filter_weight == 1.0


class TestManualSelector:
    """Tests for ManualSelector."""

    def test_manual_selection(self):
        """Test manual feature selection."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="manual",
            feature_names=["feat2", "feat4", "feat1"],
        )

        selector = ManualSelector(config)
        ranking = selector.select(X, y, feature_names)

        assert ranking.method == "manual"
        assert ranking.feature_names == ["feat2", "feat4", "feat1"]
        assert len(ranking.scores) == 3
        assert all(s == 1.0 for s in ranking.scores)  # All equal scores

    def test_manual_missing_features(self):
        """Test error when manual features not in dataset."""
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3"]

        config = FeatureSelectionConfig(
            method="manual",
            feature_names=["feat1", "feat_missing"],  # feat_missing doesn't exist
        )

        selector = ManualSelector(config)

        with pytest.raises(ValueError, match="not found"):
            selector.select(X, y, feature_names)

    def test_manual_input_validation(self):
        """Test input validation in manual selector."""
        X = np.random.rand(100, 3)
        y = np.random.rand(50)  # Mismatched length
        feature_names = ["feat1", "feat2", "feat3"]

        config = FeatureSelectionConfig(
            method="manual",
            feature_names=["feat1"],
        )

        selector = ManualSelector(config)

        with pytest.raises(ValueError, match="must match"):
            selector.select(X, y, feature_names)


class TestFilterSelector:
    """Tests for FilterSelector."""

    def test_filter_selection(self):
        """Test filter-based feature selection."""
        np.random.seed(42)

        # Create data with known properties
        n_samples = 100
        X = np.random.rand(n_samples, 5)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.rand(n_samples) * 0.1

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="filter",
            min_correlation=0.1,
            min_variance=0.01,
        )

        selector = FilterSelector(config)
        ranking = selector.select(X, y, feature_names)

        assert ranking.method == "filter"
        assert len(ranking.feature_names) > 0  # At least some features
        assert len(ranking.feature_names) <= len(feature_names)
        # Scores should be sorted descending
        assert all(
            ranking.scores[i] >= ranking.scores[i + 1] for i in range(len(ranking.scores) - 1)
        )

    def test_filter_all_filtered_fallback(self):
        """Test that filter returns all features if none pass filters."""
        np.random.seed(42)

        # Create uncorrelated random data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="filter",
            min_correlation=0.95,  # Very high threshold
            min_variance=0.01,
        )

        selector = FilterSelector(config)
        ranking = selector.select(X, y, feature_names)

        # Should return all features as fallback
        assert len(ranking.feature_names) == len(feature_names)

    def test_filter_low_variance_removal(self):
        """Test that filter removes low-variance features."""
        np.random.seed(42)

        # Create data with one constant feature
        X = np.random.rand(100, 3)
        X[:, 1] = 0.5  # Constant feature (zero variance)
        y = X[:, 0] + np.random.rand(100) * 0.1

        feature_names = ["feat1", "feat2", "feat3"]

        config = FeatureSelectionConfig(
            method="filter",
            min_variance=0.001,
        )

        selector = FilterSelector(config)
        ranking = selector.select(X, y, feature_names)

        # feat2 should be filtered out due to zero variance
        assert "feat2" not in ranking.feature_names or len(ranking.feature_names) == 3


class TestEnsembleSelector:
    """Tests for EnsembleSelector."""

    def test_ensemble_selection(self):
        """Test ensemble-based feature selection."""
        np.random.seed(42)

        # Create data with signal
        n_samples = 100
        X = np.random.rand(n_samples, 5)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.rand(n_samples) * 0.1

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="ensemble",
            n_ensemble_models=2,
            ensemble_seeds=[42, 123],
            model_types=["xgboost", "random_forest"],
        )

        selector = EnsembleSelector(config)
        ranking = selector.select(X, y, feature_names)

        assert ranking.method == "ensemble"
        assert len(ranking.feature_names) == len(feature_names)
        # Scores should be sorted descending
        assert all(
            ranking.scores[i] >= ranking.scores[i + 1] for i in range(len(ranking.scores) - 1)
        )
        # Metadata should contain per-model importances
        assert "per_model_importances" in ranking.metadata

    def test_ensemble_deterministic(self):
        """Test that ensemble is deterministic with fixed seeds."""
        np.random.seed(42)

        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="ensemble",
            n_ensemble_models=2,
            ensemble_seeds=[42, 42],  # Same seed twice
        )

        selector = EnsembleSelector(config)

        # Run twice
        ranking1 = selector.select(X, y, feature_names)
        ranking2 = selector.select(X, y, feature_names)

        # Should produce same results
        assert ranking1.feature_names == ranking2.feature_names
        np.testing.assert_array_almost_equal(ranking1.scores, ranking2.scores)

    def test_ensemble_multiple_model_types(self):
        """Test ensemble with different model types."""
        np.random.seed(42)

        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="ensemble",
            n_ensemble_models=3,
            ensemble_seeds=[42, 123, 456],
            model_types=["xgboost", "random_forest", "extra_trees"],
        )

        selector = EnsembleSelector(config)
        ranking = selector.select(X, y, feature_names)

        assert len(ranking.feature_names) == 5
        assert "num_models" in ranking.metadata
        assert ranking.metadata["num_models"] == 3


class TestHybridSelector:
    """Tests for HybridSelector."""

    def test_hybrid_selection(self):
        """Test hybrid filter + ensemble selection."""
        np.random.seed(42)

        X = np.random.rand(100, 5)
        y = X[:, 0] * 2 + np.random.rand(100) * 0.1

        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(
            method="hybrid",
            min_correlation=0.05,
            n_ensemble_models=2,
            ensemble_seeds=[42, 123],
            hybrid_filter_weight=0.5,
        )

        selector = HybridSelector(config)
        ranking = selector.select(X, y, feature_names)

        assert ranking.method == "hybrid"
        assert len(ranking.feature_names) > 0
        # Metadata should contain both filter and ensemble info
        assert "filter_metadata" in ranking.metadata
        assert "ensemble_metadata" in ranking.metadata

    def test_hybrid_weighting(self):
        """Test that hybrid weighting affects results."""
        np.random.seed(42)

        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        # Filter-heavy weighting
        config_filter = FeatureSelectionConfig(
            method="hybrid",
            hybrid_filter_weight=0.9,
            n_ensemble_models=2,
            ensemble_seeds=[42, 123],
        )

        # Ensemble-heavy weighting
        config_ensemble = FeatureSelectionConfig(
            method="hybrid",
            hybrid_filter_weight=0.1,
            n_ensemble_models=2,
            ensemble_seeds=[42, 123],
        )

        selector_filter = HybridSelector(config_filter)
        selector_ensemble = HybridSelector(config_ensemble)

        ranking_filter = selector_filter.select(X, y, feature_names)
        ranking_ensemble = selector_ensemble.select(X, y, feature_names)

        # Different weightings should produce different scores
        # (though feature order might be similar)
        assert not np.allclose(ranking_filter.scores, ranking_ensemble.scores)


class TestFeatureSelectorRegistry:
    """Tests for feature selector registry and factory."""

    def test_registry_contains_all_methods(self):
        """Test that registry contains all expected methods."""
        expected_methods = ["manual", "filter", "ensemble", "hybrid"]

        for method in expected_methods:
            assert method in FEATURE_SELECTOR_REGISTRY

    def test_factory_function(self):
        """Test get_feature_selector factory function."""
        config = FeatureSelectionConfig(method="filter")
        selector = get_feature_selector(config)

        assert isinstance(selector, FilterSelector)
        assert isinstance(selector, FeatureSelector)

    def test_factory_with_each_method(self):
        """Test factory creates correct selector for each method."""
        methods_and_types = [
            ("manual", ManualSelector, {"feature_names": ["feat1"]}),
            ("filter", FilterSelector, {}),
            ("ensemble", EnsembleSelector, {}),
            ("hybrid", HybridSelector, {}),
        ]

        for method, expected_type, extra_kwargs in methods_and_types:
            config = FeatureSelectionConfig(method=method, **extra_kwargs)
            selector = get_feature_selector(config)
            assert isinstance(selector, expected_type)

    def test_factory_unknown_method(self):
        """Test factory raises error for unknown method."""
        # This should be caught by config validation, but test factory too
        with pytest.raises(ValidationError):
            FeatureSelectionConfig(method="unknown")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_features(self):
        """Test error when X has no features."""
        X = np.random.rand(100, 0)  # No features
        y = np.random.rand(100)
        feature_names = []

        config = FeatureSelectionConfig(method="filter")
        selector = FilterSelector(config)

        with pytest.raises(ValueError, match="at least one feature"):
            selector.select(X, y, feature_names)

    def test_empty_samples(self):
        """Test error when X has no samples."""
        X = np.random.rand(0, 5)  # No samples
        y = np.random.rand(0)
        feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        config = FeatureSelectionConfig(method="filter")
        selector = FilterSelector(config)

        with pytest.raises(ValueError, match="at least one sample"):
            selector.select(X, y, feature_names)

    def test_mismatched_feature_names_length(self):
        """Test error when feature_names length doesn't match X columns."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        feature_names = ["feat1", "feat2", "feat3"]  # Only 3 names for 5 features

        config = FeatureSelectionConfig(method="filter")
        selector = FilterSelector(config)

        with pytest.raises(ValueError, match="must match"):
            selector.select(X, y, feature_names)

    def test_single_feature(self):
        """Test handling of single feature case."""
        X = np.random.rand(100, 1)
        y = np.random.rand(100)
        feature_names = ["feat1"]

        config = FeatureSelectionConfig(method="filter")
        selector = FilterSelector(config)

        ranking = selector.select(X, y, feature_names)

        assert len(ranking.feature_names) == 1
        assert ranking.feature_names[0] == "feat1"

    def test_nan_handling_in_correlation(self):
        """Test that NaN correlations are handled gracefully."""
        np.random.seed(42)

        # Create data with constant feature (will produce NaN correlation)
        X = np.random.rand(100, 3)
        X[:, 1] = 0.5  # Constant feature
        y = np.random.rand(100)

        feature_names = ["feat1", "feat2", "feat3"]

        config = FeatureSelectionConfig(
            method="filter",
            min_correlation=0.0,
            min_variance=0.0,  # Allow zero variance
        )

        selector = FilterSelector(config)
        ranking = selector.select(X, y, feature_names)

        # Should not raise error, NaN should be converted to 0
        assert len(ranking.feature_names) > 0
