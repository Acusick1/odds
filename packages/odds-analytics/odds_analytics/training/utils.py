"""
Utility functions for ML training workflows.

This module provides helper functions used across different training strategies
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from odds_analytics.training.config import MLTrainingConfig


def flatten_config_for_tracking(
    config: MLTrainingConfig,
    X_train: np.ndarray,
    feature_names: list[str],
    X_val: np.ndarray | None = None,
) -> dict[str, str | int | float]:
    """
    Flatten training configuration into a dictionary suitable for experiment tracking.

    Extracts all relevant configuration parameters including experiment metadata,
    data config, model hyperparameters, and feature settings into a flat dictionary
    that can be logged to MLflow or other tracking backends.

    Args:
        config: ML training configuration object
        X_train: Training features array (used for n_samples)
        feature_names: List of feature names (used for n_features)
        X_val: Optional validation features (used for has_validation flag)

    Returns:
        Flat dictionary of configuration parameters with string/numeric values

    Example:
        >>> config = MLTrainingConfig.from_yaml("experiments/xgb.yaml")
        >>> params = flatten_config_for_tracking(config, X_train, feature_names, X_val)
        >>> tracker.log_params(params)
    """
    # Basic experiment metadata
    config_params = {
        "experiment_name": config.experiment.name,
        "experiment_description": config.experiment.description or "",
        "strategy_type": config.training.strategy_type,
        "n_samples": len(X_train),
        "n_features": len(feature_names),
        "has_validation": X_val is not None,
    }

    # Data configuration
    config_params.update(
        {
            "data_start_date": str(config.training.data.start_date),
            "data_end_date": str(config.training.data.end_date),
            "test_split": config.training.data.test_split,
            "validation_split": config.training.data.validation_split,
        }
    )

    # Model hyperparameters (already flat from model config)
    model_params = config.training.model.model_dump()
    config_params.update(model_params)

    # Feature configuration
    features = config.training.features
    config_params.update(
        {
            "sharp_bookmakers": ",".join(features.sharp_bookmakers),
            "retail_bookmakers": ",".join(features.retail_bookmakers),
            "markets": ",".join(features.markets),
            "outcome": features.outcome,
            "closing_tier": features.closing_tier.value,
            "sampling_strategy": features.sampling.strategy,
        }
    )

    # Add LSTM-specific feature config if present
    if hasattr(features, "lookback_hours") and features.lookback_hours is not None:
        config_params["lookback_hours"] = features.lookback_hours
    if hasattr(features, "timesteps") and features.timesteps is not None:
        config_params["timesteps"] = features.timesteps

    # Experiment tags (comma-separated)
    if config.experiment.tags:
        config_params["tags"] = ",".join(config.experiment.tags)

    return config_params
