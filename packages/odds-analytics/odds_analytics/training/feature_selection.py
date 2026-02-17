"""
Feature selection module for pre-HPO feature reduction.

This module provides an extensible feature selection system that reduces the feature
space before hyperparameter optimization, improving model generalization on small datasets.

Key Features:
- Multiple selection methods via registry pattern (manual, filter, ensemble, hybrid)
- Ensemble approach using multiple models to avoid single-model bias
- Filter methods for statistical feature selection
- Manual selection for domain knowledge integration
- Configurable and extensible via Pydantic configuration

Example usage:
    ```python
    from odds_analytics.training.feature_selection import (
        get_feature_selector,
        FeatureSelectionConfig,
    )

    config = FeatureSelectionConfig(
        enabled=True,
        method="ensemble",
        n_ensemble_models=3,
    )

    selector = get_feature_selector(config)
    ranking = selector.select(X_train, y_train, feature_names)

    # Use selected features
    selected_indices = [feature_names.index(f) for f in ranking.feature_names]
    X_train_selected = X_train[:, selected_indices]
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.feature_selection import VarianceThreshold

from odds_analytics.training.config import FeatureSelectionConfig

logger = structlog.get_logger()

__all__ = [
    "apply_variance_filter",
    "FeatureRanking",
    "FeatureSelector",
    "ManualSelector",
    "FilterSelector",
    "EnsembleSelector",
    "HybridSelector",
    "FEATURE_SELECTOR_REGISTRY",
    "get_feature_selector",
]


# =============================================================================
# Variance Filter
# =============================================================================


def apply_variance_filter(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Drop features whose variance is below threshold using sklearn VarianceThreshold.

    Handles both 2D (samples, features) and 3D (samples, timesteps, features) arrays.
    For 3D arrays, variance is computed over the flattened (samples × timesteps) axis.
    Skips filtering when fewer than 2 samples are present.

    Args:
        X: Feature array, shape (samples, features) or (samples, timesteps, features).
        feature_names: Names corresponding to the last axis of X.
        threshold: Variance threshold; features strictly below this are dropped.

    Returns:
        Filtered (X, feature_names) with low-variance columns removed.
    """
    if len(X) < 2:
        return X, feature_names

    X_2d = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_2d)
    mask = selector.get_support()

    dropped = [name for name, keep in zip(feature_names, mask, strict=False) if not keep]
    if dropped:
        logger.info("dropped_low_variance_features", features=dropped, count=len(dropped))

    return X[..., mask], [name for name, keep in zip(feature_names, mask, strict=False) if keep]


# =============================================================================
# Output Models
# =============================================================================


class FeatureRanking(BaseModel):
    """
    Output model for feature selection results.

    Contains ranked features with importance scores and method metadata.

    Attributes:
        feature_names: Features ordered by importance (highest first)
        scores: Importance scores corresponding to feature_names
        method: Selection method used ("manual", "filter", "ensemble", "hybrid")
        metadata: Method-specific info (e.g., {"per_model_scores": {...}})
    """

    model_config = ConfigDict(extra="forbid")

    feature_names: list[str] = Field(..., min_length=1, description="Ranked feature names")
    scores: list[float] = Field(..., description="Importance scores for features")
    method: str = Field(..., description="Selection method used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Method-specific metadata")

    @model_validator(mode="after")
    def validate_lengths_match(self) -> FeatureRanking:
        """Ensure feature_names and scores have matching lengths."""
        if len(self.feature_names) != len(self.scores):
            raise ValueError(
                f"feature_names ({len(self.feature_names)}) and scores "
                f"({len(self.scores)}) must have same length"
            )
        return self


# =============================================================================
# Abstract Base Class
# =============================================================================


class FeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.

    All selection methods must implement select() and return FeatureRanking.
    Provides common initialization and utility methods.
    """

    def __init__(self, config: FeatureSelectionConfig):
        """
        Initialize feature selector.

        Args:
            config: Feature selection configuration
        """
        self.config = config

    @abstractmethod
    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FeatureRanking:
        """
        Select and rank features.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            FeatureRanking with ordered features and scores

        Raises:
            ValueError: If input dimensions don't match or invalid data
        """
        pass

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """
        Validate input data dimensions and consistency.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature name list

        Raises:
            ValueError: If dimensions don't match or data is invalid
        """
        if X.shape[0] != len(y):
            raise ValueError(f"X rows ({X.shape[0]}) must match y length ({len(y)})")

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X columns ({X.shape[1]}) must match feature_names length "
                f"({len(feature_names)})"
            )

        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")

        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")


# =============================================================================
# Selection Method Implementations
# =============================================================================


class ManualSelector(FeatureSelector):
    """
    Manual feature selection using explicit feature names from config.

    Use case: Domain knowledge, hypothesis testing, or specific feature subsets.
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FeatureRanking:
        """
        Select features using explicit list from config.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            FeatureRanking with manually specified features

        Raises:
            ValueError: If manual features not found in feature_names
        """
        self._validate_inputs(X, y, feature_names)

        if not self.config.feature_names:
            raise ValueError("Manual selector requires feature_names in config")

        # Validate all manual features exist
        missing = set(self.config.feature_names) - set(feature_names)
        if missing:
            raise ValueError(f"Manual features not found in feature_names: {sorted(missing)}")

        # Return features in order specified, with equal scores
        selected_features = self.config.feature_names
        scores = [1.0] * len(selected_features)

        logger.info(
            "manual_feature_selection",
            num_selected=len(selected_features),
            num_total=len(feature_names),
            features=selected_features,
        )

        return FeatureRanking(
            feature_names=selected_features,
            scores=scores,
            method="manual",
            metadata={"total_features": len(feature_names)},
        )


class FilterSelector(FeatureSelector):
    """
    Statistical filter-based feature selection.

    Uses correlation with target and variance thresholds to filter features.
    Fast and model-agnostic.
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FeatureRanking:
        """
        Select features using statistical filters.

        Filters features by:
        1. Variance threshold (removes near-constant features)
        2. Correlation with target (removes low-correlation features)

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            FeatureRanking with filtered features ranked by correlation
        """
        self._validate_inputs(X, y, feature_names)

        # Calculate variance for each feature
        variances = np.var(X, axis=0)

        # Calculate correlation with target for each feature
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])

        # Handle NaN correlations (constant features)
        correlations = np.nan_to_num(correlations, nan=0.0)

        # Apply filters
        variance_mask = variances >= self.config.min_variance
        correlation_mask = np.abs(correlations) >= self.config.min_correlation

        # Combine filters
        combined_mask = variance_mask & correlation_mask

        if not np.any(combined_mask):
            logger.warning(
                "filter_selection_no_features",
                min_variance=self.config.min_variance,
                min_correlation=self.config.min_correlation,
                message="No features passed filters, returning all features",
            )
            # Return all features if none pass filters
            combined_mask = np.ones(len(feature_names), dtype=bool)

        # Get surviving features and their correlations
        selected_indices = np.where(combined_mask)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        selected_correlations = np.abs(correlations[selected_indices])

        # Sort by correlation (highest first)
        sort_indices = np.argsort(selected_correlations)[::-1]
        selected_features = [selected_features[i] for i in sort_indices]
        scores = selected_correlations[sort_indices].tolist()

        logger.info(
            "filter_feature_selection",
            num_selected=len(selected_features),
            num_total=len(feature_names),
            num_variance_filtered=np.sum(~variance_mask),
            num_correlation_filtered=np.sum(~correlation_mask),
            top_5_features=selected_features[:5],
        )

        return FeatureRanking(
            feature_names=selected_features,
            scores=scores,
            method="filter",
            metadata={
                "total_features": len(feature_names),
                "variance_threshold": self.config.min_variance,
                "correlation_threshold": self.config.min_correlation,
                "num_variance_filtered": int(np.sum(~variance_mask)),
                "num_correlation_filtered": int(np.sum(~correlation_mask)),
            },
        )


class EnsembleSelector(FeatureSelector):
    """
    Ensemble-based feature selection using multiple models.

    Trains multiple models (XGBoost, RandomForest, ExtraTrees) with different
    seeds and averages their feature importance scores. This avoids over-reliance
    on any single model's importance ranking.
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FeatureRanking:
        """
        Select features using ensemble of models.

        Trains n_ensemble_models models of different types with different seeds,
        then averages their feature importance scores.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            FeatureRanking with features ranked by average importance
        """
        self._validate_inputs(X, y, feature_names)

        # Import ML libraries here to avoid overhead when not using ensemble
        try:
            import xgboost as xgb
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        except ImportError as e:
            raise ImportError(
                "Ensemble selector requires xgboost and scikit-learn. "
                "Install with: uv add xgboost scikit-learn"
            ) from e

        # Collect importance scores from all models
        all_importances = []
        model_metadata = {}

        for i, seed in enumerate(self.config.ensemble_seeds):
            model_type = self.config.model_types[i % len(self.config.model_types)]

            # Create model based on type
            if model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=seed,
                    n_jobs=1,
                )
            elif model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=seed,
                    n_jobs=1,
                )
            elif model_type == "extra_trees":
                model = ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=seed,
                    n_jobs=1,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train model
            model.fit(X, y)

            # Get feature importances
            importances = model.feature_importances_

            all_importances.append(importances)
            model_metadata[f"{model_type}_seed_{seed}"] = importances.tolist()

            logger.debug(
                "ensemble_model_trained",
                model_type=model_type,
                seed=seed,
                top_3_features=[feature_names[i] for i in np.argsort(importances)[::-1][:3]],
            )

        # Average importances across all models
        avg_importances = np.mean(all_importances, axis=0)

        # Sort features by average importance (highest first)
        sort_indices = np.argsort(avg_importances)[::-1]
        selected_features = [feature_names[i] for i in sort_indices]
        scores = avg_importances[sort_indices].tolist()

        logger.info(
            "ensemble_feature_selection",
            num_features=len(selected_features),
            num_models=len(self.config.ensemble_seeds),
            model_types=self.config.model_types,
            top_5_features=selected_features[:5],
            top_5_scores=scores[:5],
        )

        return FeatureRanking(
            feature_names=selected_features,
            scores=scores,
            method="ensemble",
            metadata={
                "total_features": len(feature_names),
                "num_models": len(self.config.ensemble_seeds),
                "model_types": self.config.model_types,
                "seeds": self.config.ensemble_seeds,
                "per_model_importances": model_metadata,
            },
        )


class HybridSelector(FeatureSelector):
    """
    Hybrid feature selection combining filter and ensemble methods.

    First applies filter methods to remove obviously poor features,
    then uses ensemble importance on remaining features.
    Combines both approaches with configurable weighting.
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FeatureRanking:
        """
        Select features using hybrid filter + ensemble approach.

        Process:
        1. Apply filter selection to get initial subset
        2. Apply ensemble selection on filtered features
        3. Combine rankings with weighted average

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            FeatureRanking with hybrid-ranked features
        """
        self._validate_inputs(X, y, feature_names)

        # Step 1: Apply filter selection
        filter_selector = FilterSelector(self.config)
        filter_ranking = filter_selector.select(X, y, feature_names)

        # Get indices of filtered features
        filtered_indices = [feature_names.index(f) for f in filter_ranking.feature_names]
        X_filtered = X[:, filtered_indices]

        # Step 2: Apply ensemble selection on filtered features
        ensemble_selector = EnsembleSelector(self.config)
        ensemble_ranking = ensemble_selector.select(X_filtered, y, filter_ranking.feature_names)

        # Step 3: Combine rankings with weighted average
        # Normalize scores to [0, 1] for fair weighting
        filter_scores_array = np.array(filter_ranking.scores)
        if filter_scores_array.max() > 0:
            filter_scores_array = filter_scores_array / filter_scores_array.max()

        # Normalize ensemble scores
        ensemble_scores_array = np.array(ensemble_ranking.scores)
        if ensemble_scores_array.max() > 0:
            ensemble_scores_array = ensemble_scores_array / ensemble_scores_array.max()

        # Create normalized score dicts
        filter_scores_norm = dict(
            zip(filter_ranking.feature_names, filter_scores_array, strict=True)
        )
        ensemble_scores_norm = dict(
            zip(ensemble_ranking.feature_names, ensemble_scores_array, strict=True)
        )

        # Combine scores with weighting
        filter_weight = self.config.hybrid_filter_weight
        ensemble_weight = 1.0 - filter_weight

        combined_scores = {}
        for feature in ensemble_ranking.feature_names:
            combined_scores[feature] = (
                filter_weight * filter_scores_norm[feature]
                + ensemble_weight * ensemble_scores_norm[feature]
            )

        # Sort by combined score (highest first)
        sorted_features = sorted(
            combined_scores.keys(), key=lambda f: combined_scores[f], reverse=True
        )
        sorted_scores = [combined_scores[f] for f in sorted_features]

        logger.info(
            "hybrid_feature_selection",
            num_selected=len(sorted_features),
            num_total=len(feature_names),
            filter_weight=filter_weight,
            ensemble_weight=ensemble_weight,
            top_5_features=sorted_features[:5],
        )

        return FeatureRanking(
            feature_names=sorted_features,
            scores=sorted_scores,
            method="hybrid",
            metadata={
                "total_features": len(feature_names),
                "filter_weight": filter_weight,
                "ensemble_weight": ensemble_weight,
                "num_filtered": len(filter_ranking.feature_names),
                "filter_metadata": filter_ranking.metadata,
                "ensemble_metadata": ensemble_ranking.metadata,
            },
        )


# =============================================================================
# Registry and Factory
# =============================================================================


FEATURE_SELECTOR_REGISTRY: dict[str, type[FeatureSelector]] = {
    "manual": ManualSelector,
    "filter": FilterSelector,
    "ensemble": EnsembleSelector,
    "hybrid": HybridSelector,
}


def get_feature_selector(config: FeatureSelectionConfig) -> FeatureSelector:
    """
    Factory function to get selector instance from config.

    Args:
        config: Feature selection configuration

    Returns:
        Instantiated FeatureSelector

    Raises:
        ValueError: If method not in registry

    Example:
        >>> config = FeatureSelectionConfig(method="ensemble")
        >>> selector = get_feature_selector(config)
        >>> ranking = selector.select(X_train, y_train, feature_names)
    """
    if config.method not in FEATURE_SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selection method '{config.method}'. "
            f"Available: {list(FEATURE_SELECTOR_REGISTRY.keys())}"
        )

    selector_cls = FEATURE_SELECTOR_REGISTRY[config.method]
    return selector_cls(config)
