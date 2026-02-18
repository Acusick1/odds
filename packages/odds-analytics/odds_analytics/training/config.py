"""
ML Training Configuration Schema.

This module provides a Pydantic-based configuration system for ML model training
that supports YAML/JSON loading and future Optuna/MLflow integration.

Key Features:
- Hierarchical configuration structure for extensibility
- YAML/JSON file loading and serialization
- Discriminated unions for strategy-specific configs
- Validation with clear error messages
- Search space definitions for hyperparameter tuning (Optuna)
- Tracking configuration for experiment logging (MLflow)

Example usage:
    ```python
    from odds_analytics.training import MLTrainingConfig

    # Load from YAML
    config = MLTrainingConfig.from_yaml("experiments/xgboost_v1.yaml")

    # Load from JSON
    config = MLTrainingConfig.from_json("experiments/lstm_v1.json")

    # Access nested configuration
    print(config.training.strategy_type)  # "xgboost"
    print(config.training.model.n_estimators)  # 100

    # Serialize back to YAML
    config.to_yaml("experiments/xgboost_v1_updated.yaml")
    ```
"""

from __future__ import annotations

import copy
import json
import math
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml
from odds_lambda.fetch_tier import FetchTier
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = structlog.get_logger()

# Maximum inheritance depth to prevent excessive nesting
_MAX_INHERITANCE_DEPTH = 10


# =============================================================================
# Deep Merge Utility
# =============================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Recursively merges nested dictionaries while override values replace
    base values at leaf nodes. Lists are not merged - override replaces base.

    Args:
        base: Base dictionary to merge into
        override: Dictionary with override values

    Returns:
        New merged dictionary

    Example:
        >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> override = {"a": {"b": 10}, "e": 5}
        >>> deep_merge(base, override)
        {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "LSTMConfig",
    "SamplingConfig",
    "FeatureConfig",
    "FeatureSelectionConfig",
    "SearchSpace",
    "TrainingConfig",
    "TuningConfig",
    "TrackingConfig",
    "MLTrainingConfig",
    "resolve_search_spaces",
    "deep_merge",
]


# =============================================================================
# Experiment Metadata
# =============================================================================


class ExperimentConfig(BaseModel):
    """
    Experiment metadata for tracking and organization.

    Used for experiment identification, documentation, and MLflow tracking.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Unique experiment name for identification",
        min_length=1,
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering experiments",
    )
    description: str = Field(
        default="",
        description="Detailed description of the experiment purpose and approach",
    )


# =============================================================================
# Data Configuration
# =============================================================================


class DataConfig(BaseModel):
    """
    Data loading and splitting configuration.

    Controls the date range for training data and train/test split settings.
    Supports optional cross-validation for more robust model evaluation.

    Cross-Validation Methods:
        - kfold: Standard K-Fold cross-validation.
        - timeseries: Walk-forward validation using TimeSeriesSplit.
    """

    model_config = ConfigDict(extra="forbid")

    start_date: date = Field(
        ...,
        description="Start date for training data (inclusive)",
    )
    end_date: date = Field(
        ...,
        description="End date for training data (inclusive)",
    )
    test_split: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of data to use for testing (0.0 to 1.0)",
    )
    validation_split: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of training data to use for validation (0.0 to 1.0). "
        "Ignored when use_kfold=True.",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible splits",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle data before splitting",
    )

    # Cross-Validation settings
    use_kfold: bool = Field(
        default=True,
        description="Enable cross-validation for model evaluation",
    )
    cv_method: Literal["kfold", "timeseries", "group_timeseries"] = Field(
        default="timeseries",
        description="Cross-validation method. 'timeseries' uses walk-forward validation "
        "(recommended for temporal betting data), 'kfold' uses standard K-Fold, "
        "'group_timeseries' uses walk-forward with event-level grouping "
        "(required for multi-horizon data where events have multiple rows).",
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of folds for cross-validation (only used when use_kfold=True)",
    )
    kfold_shuffle: bool = Field(
        default=True,
        description="Whether to shuffle data before splitting into folds. "
        "Only applies when cv_method='kfold'; ignored for 'timeseries'.",
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> DataConfig:
        """Ensure start_date is before end_date."""
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )
        return self

    @model_validator(mode="after")
    def validate_splits(self) -> DataConfig:
        """Ensure splits don't exceed 1.0 in total."""
        # When using kfold, validation_split is ignored
        if self.use_kfold:
            return self

        total = self.test_split + self.validation_split
        if total >= 1.0:
            raise ValueError(
                f"test_split ({self.test_split}) + validation_split ({self.validation_split}) "
                f"= {total} must be less than 1.0"
            )
        return self


# =============================================================================
# Model-Specific Configurations
# =============================================================================


class XGBoostConfig(BaseModel):
    """
    XGBoost model hyperparameters.

    Contains all configurable XGBoost parameters for regression tasks.
    See XGBoost documentation for detailed parameter descriptions.
    """

    model_config = ConfigDict(extra="forbid")

    # Tree parameters
    n_estimators: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of boosting rounds",
    )
    max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum tree depth for base learners",
    )
    min_child_weight: int = Field(
        default=1,
        ge=0,
        description="Minimum sum of instance weight needed in a child",
    )

    # Learning parameters
    learning_rate: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Boosting learning rate (eta)",
    )
    gamma: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum loss reduction required for split",
    )

    # Sampling parameters
    subsample: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of training instances",
    )
    colsample_bytree: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns when constructing each tree",
    )
    colsample_bylevel: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns for each level",
    )
    colsample_bynode: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Subsample ratio of columns for each split",
    )

    # Regularization
    reg_alpha: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 regularization term on weights (alpha)",
    )
    reg_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="L2 regularization term on weights (lambda)",
    )

    # Other parameters
    objective: str = Field(
        default="reg:squarederror",
        description="Learning objective (e.g., reg:squarederror, reg:absoluteerror)",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel threads (-1 for all available)",
    )

    # Early stopping
    early_stopping_rounds: int | None = Field(
        default=None,
        ge=1,
        description="Validation metric needs to improve at least once in every N rounds",
    )


class LSTMConfig(BaseModel):
    """
    LSTM model architecture and training hyperparameters.

    Contains all configurable parameters for LSTM sequence models.
    """

    model_config = ConfigDict(extra="forbid")

    # Architecture parameters
    hidden_size: int = Field(
        default=64,
        ge=8,
        le=1024,
        description="Number of features in LSTM hidden state",
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of recurrent LSTM layers",
    )
    dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.9,
        description="Dropout probability between LSTM layers",
    )
    bidirectional: bool = Field(
        default=False,
        description="Use bidirectional LSTM",
    )

    # Sequence parameters
    lookback_hours: int = Field(
        default=72,
        ge=1,
        le=168,
        description="Hours of historical data to use",
    )
    timesteps: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of sequence timesteps",
    )

    # Training parameters
    epochs: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Training batch size",
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        le=1.0,
        description="Learning rate for optimizer",
    )

    # Loss function
    loss_function: Literal["mse", "mae", "huber"] = Field(
        default="mse",
        description="Loss function for training",
    )

    # Optimizer settings
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization weight decay",
    )
    clip_grad_norm: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum gradient norm for clipping",
    )

    # Early stopping
    patience: int | None = Field(
        default=None,
        ge=1,
        description="Number of epochs without improvement before stopping",
    )
    min_delta: float = Field(
        default=0.0001,
        ge=0.0,
        description="Minimum change to qualify as improvement",
    )


# =============================================================================
# Sampling Configuration
# =============================================================================


class SamplingConfig(BaseModel):
    """
    Snapshot sampling strategy for training data preparation.

    Controls how decision-time snapshots are selected per event.
    'tier' returns the last snapshot in the specified tier (single row per event).
    'time_range' uses stratified sampling across a time window (multiple rows per event).
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["tier", "time_range"] = Field(
        default="time_range",
        description="Sampling strategy: 'tier' for single snapshot, 'time_range' for multi-horizon",
    )
    decision_tier: FetchTier = Field(
        default=FetchTier.PREGAME,
        description="Tier to sample from when strategy='tier'",
    )
    min_hours: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum hours before game for sampling window (strategy='time_range')",
    )
    max_hours: float = Field(
        default=12.0,
        gt=0.0,
        description="Maximum hours before game for sampling window (strategy='time_range')",
    )
    max_samples_per_event: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum snapshots to sample per event (strategy='time_range')",
    )

    @model_validator(mode="after")
    def validate_hours_range(self) -> SamplingConfig:
        if self.min_hours >= self.max_hours:
            raise ValueError(
                f"min_hours ({self.min_hours}) must be less than max_hours ({self.max_hours})"
            )
        return self


# =============================================================================
# Feature Configuration
# =============================================================================


class FeatureConfig(BaseModel):
    """
    Feature extraction configuration.

    Controls how features are extracted from raw odds data for model training.
    """

    model_config = ConfigDict(extra="forbid")

    # Adapter selection
    adapter: Literal["xgboost", "lstm"] = Field(
        default="xgboost",
        description="Model adapter type for feature formatting",
    )

    # Bookmaker configuration
    sharp_bookmakers: list[str] = Field(
        default=["pinnacle"],
        min_length=1,
        description="List of sharp bookmakers for feature calculation",
    )
    retail_bookmakers: list[str] = Field(
        default=["fanduel", "draftkings", "betmgm"],
        min_length=1,
        description="List of retail bookmakers for feature calculation",
    )

    # Market configuration
    markets: list[str] = Field(
        default=["h2h", "spreads", "totals"],
        min_length=1,
        description="Markets to extract features from",
    )

    # Target configuration
    outcome: Literal["home", "away"] = Field(
        default="home",
        description="Outcome to predict (home or away team)",
    )

    # Tier-based closing line configuration
    closing_tier: FetchTier = Field(
        default=FetchTier.CLOSING,
        description="Tier for closing line (last snapshot in this tier)",
    )

    # Sampling configuration
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Snapshot sampling strategy",
    )

    # Sequence model configuration (LSTM only — auto-populated from LSTMConfig
    # by TrainingConfig.sync_lstm_sequence_params when using config-driven training)
    lookback_hours: int | None = Field(
        default=None,
        ge=1,
        le=168,
        description="Hours of historical data for LSTM sequence extraction",
    )
    timesteps: int | None = Field(
        default=None,
        ge=1,
        le=168,
        description="Number of LSTM sequence timesteps",
    )

    # Feature processing
    normalize: bool = Field(
        default=False,
        description="Whether to normalize features before model input",
    )

    # Feature groups to compose
    feature_groups: tuple[str, ...] = Field(
        default=("tabular",),
        min_length=1,
        description="Feature groups to compose. Available: tabular, trajectory, polymarket",
    )

    # Trajectory feature configuration
    movement_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=0.1,
        description="Probability change threshold for counting significant movements (0.5% default)",
    )

    # Polymarket feature configuration
    pm_velocity_window_hours: float = Field(
        default=2.0,
        ge=0.5,
        le=12.0,
        description="Hours of recent PM price history used for velocity/acceleration features",
    )
    pm_price_tolerance_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Tolerance in minutes for matching PM snapshots to a target timestamp",
    )

    # Variance filter
    variance_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Drop features whose variance is below this threshold. "
        "0.0 (default) drops only strictly constant features.",
    )

    # Target formulation
    target_type: Literal["raw", "devigged_pinnacle"] = Field(
        default="raw",
        description="Target formulation. 'raw': avg implied prob delta at snapshot vs closing. "
        "'devigged_pinnacle': Pinnacle close vs Pinnacle at snapshot, both devigged.",
    )


# =============================================================================
# Feature Selection Configuration
# =============================================================================


class FeatureSelectionConfig(BaseModel):
    """
    Feature selection configuration for pre-HPO feature reduction.

    Supports multiple selection methods with configurable parameters.
    Designed to be nested in TrainingConfig hierarchy.

    Example:
        ```python
        config = FeatureSelectionConfig(
            enabled=True,
            method="ensemble",
            n_ensemble_models=3,
            ensemble_seeds=[42, 123, 456],
        )
        ```
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable feature selection before HPO",
    )
    method: Literal["manual", "filter", "ensemble", "hybrid"] = Field(
        default="ensemble",
        description="Selection method: manual, filter, ensemble, hybrid",
    )

    # Manual selection
    feature_names: list[str] | None = Field(
        default=None,
        description="Explicit feature list (only used when method='manual')",
    )

    # Filter method settings
    min_correlation: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum absolute correlation with target",
    )
    min_variance: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum feature variance threshold",
    )

    # Ensemble settings
    n_ensemble_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of models in ensemble",
    )
    ensemble_seeds: list[int] = Field(
        default=[42, 123, 456],
        description="Random seeds for ensemble models",
    )
    model_types: list[str] = Field(
        default=["xgboost", "random_forest", "extra_trees"],
        description="Model types for ensemble importance",
    )

    # Hybrid settings
    hybrid_filter_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for filter method in hybrid (1-weight goes to ensemble)",
    )

    @model_validator(mode="after")
    def validate_method_specific_fields(self) -> FeatureSelectionConfig:
        """Validate required fields for each method."""
        if self.method == "manual" and not self.feature_names:
            raise ValueError("method='manual' requires feature_names to be set")

        if self.method == "ensemble" and len(self.ensemble_seeds) != self.n_ensemble_models:
            raise ValueError(
                f"ensemble_seeds length ({len(self.ensemble_seeds)}) must match "
                f"n_ensemble_models ({self.n_ensemble_models})"
            )

        return self


# =============================================================================
# Search Space for Hyperparameter Tuning (Optuna)
# =============================================================================


class SearchSpace(BaseModel):
    """
    Search space definition for hyperparameter tuning.

    Used with Optuna for defining parameter search ranges.
    Supports int, float, and categorical distributions.

    Example:
        ```yaml
        n_estimators:
          type: int
          low: 50
          high: 500
          step: 50

        learning_rate:
          type: float
          low: 0.001
          high: 0.3
          log: true

        objective:
          type: categorical
          choices: ["reg:squarederror", "reg:absoluteerror"]
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["int", "float", "categorical"] = Field(
        ...,
        description="Parameter type for search space",
    )
    low: float | None = Field(
        default=None,
        description="Lower bound for int/float search (inclusive)",
    )
    high: float | None = Field(
        default=None,
        description="Upper bound for int/float search (inclusive)",
    )
    step: float | None = Field(
        default=None,
        gt=0.0,
        description="Step size for discrete search (int/float)",
    )
    log: bool = Field(
        default=False,
        description="Use logarithmic scale for float search",
    )
    choices: list[Any] | None = Field(
        default=None,
        description="List of choices for categorical search",
    )

    @model_validator(mode="after")
    def validate_search_space(self) -> SearchSpace:
        """Validate search space parameters based on type."""
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Search space type '{self.type}' requires 'low' and 'high' bounds"
                )
            if self.low >= self.high:
                raise ValueError(f"'low' ({self.low}) must be less than 'high' ({self.high})")
            if self.log and self.low <= 0:
                raise ValueError(f"Log scale requires 'low' > 0, got {self.low}")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    "Search space type 'categorical' requires non-empty 'choices' list"
                )
            if self.low is not None or self.high is not None:
                raise ValueError("Search space type 'categorical' should not have 'low' or 'high'")
        return self


# =============================================================================
# Tuning Configuration (Optuna)
# =============================================================================


class TuningConfig(BaseModel):
    """
    Hyperparameter tuning configuration for Optuna.

    Controls the optimization process for finding best hyperparameters.
    """

    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of optimization trials",
    )
    timeout: int | None = Field(
        default=None,
        gt=0,
        description="Maximum optimization time in seconds",
    )
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction for the objective",
    )
    metric: str = Field(
        default="val_mse",
        description="Metric to optimize (e.g., val_mse, val_mae, val_r2)",
    )
    pruner: Literal["median", "hyperband", "none"] = Field(
        default="median",
        description="Pruning strategy for early trial termination",
    )
    sampler: Literal["tpe", "random", "cmaes"] = Field(
        default="tpe",
        description="Sampling strategy for hyperparameter selection",
    )

    # Search spaces for each parameter (parameter_name -> SearchSpace)
    search_spaces: dict[str, SearchSpace] = Field(
        default_factory=dict,
        description="Search space definitions for each tunable parameter",
    )


# =============================================================================
# Tracking Configuration (MLflow)
# =============================================================================


class TrackingConfig(BaseModel):
    """
    Experiment tracking configuration.

    Controls logging and artifact storage for experiment reproducibility.
    Supports MLflow with extensibility for future backends (W&B, Neptune).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable experiment tracking",
    )
    backend: str = Field(
        default="mlflow",
        description="Tracking backend to use (mlflow, wandb, neptune)",
    )
    tracking_uri: str = Field(
        default="mlruns",
        description="MLflow tracking server URI or local directory",
    )
    experiment_name: str | None = Field(
        default=None,
        description="MLflow experiment name (uses experiment.name if not set)",
    )
    run_name: str | None = Field(
        default=None,
        description="MLflow run name",
    )
    log_model: bool = Field(
        default=True,
        description="Log trained model as MLflow artifact",
    )
    log_params: bool = Field(
        default=True,
        description="Log all configuration parameters",
    )
    log_metrics: bool = Field(
        default=True,
        description="Log training metrics at each epoch",
    )
    artifact_path: str | None = Field(
        default=None,
        description="Custom artifact storage path",
    )


# =============================================================================
# Training Configuration (Main Training Settings)
# =============================================================================


class TrainingConfig(BaseModel):
    """
    Main training configuration with model-specific settings.

    Uses discriminated unions to select the appropriate model configuration
    based on strategy_type.
    """

    model_config = ConfigDict(extra="forbid")

    strategy_type: Literal["xgboost", "xgboost_line_movement", "lstm", "lstm_line_movement"] = (
        Field(
            ...,
            description="Type of ML strategy to train",
        )
    )

    # Data configuration
    data: DataConfig = Field(
        ...,
        description="Data loading and splitting settings",
    )

    # Model-specific configuration (discriminated union)
    model: XGBoostConfig | LSTMConfig = Field(
        ...,
        description="Model hyperparameters (type depends on strategy_type)",
    )

    # Feature configuration
    features: FeatureConfig = Field(
        default_factory=FeatureConfig,
        description="Feature extraction settings",
    )

    # Feature selection configuration
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig,
        description="Feature selection settings for pre-HPO reduction",
    )

    # Strategy-specific parameters
    min_predicted_movement: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum predicted movement to trigger bet",
    )
    movement_confidence_scale: float = Field(
        default=5.0,
        gt=0.0,
        description="Scale factor for converting movement to confidence",
    )
    base_confidence: float = Field(
        default=0.52,
        ge=0.5,
        le=1.0,
        description="Base confidence at minimum movement threshold",
    )

    # Output settings
    output_path: str = Field(
        default="models",
        description="Directory path for saving trained models",
    )
    model_name: str | None = Field(
        default=None,
        description="Custom model filename (auto-generated if not set)",
    )

    @model_validator(mode="after")
    def validate_model_type(self) -> TrainingConfig:
        """Ensure model config matches strategy type."""
        if self.strategy_type in ("xgboost", "xgboost_line_movement"):
            if not isinstance(self.model, XGBoostConfig):
                raise ValueError(
                    f"Strategy '{self.strategy_type}' requires XGBoostConfig, "
                    f"got {type(self.model).__name__}"
                )
        elif self.strategy_type in ("lstm", "lstm_line_movement"):
            if not isinstance(self.model, LSTMConfig):
                raise ValueError(
                    f"Strategy '{self.strategy_type}' requires LSTMConfig, "
                    f"got {type(self.model).__name__}"
                )
        return self

    @model_validator(mode="after")
    def sync_lstm_sequence_params(self) -> TrainingConfig:
        """Auto-populate LSTM sequence params from LSTMConfig into FeatureConfig.

        LSTMConfig is the canonical source for lookback_hours and timesteps.
        When FeatureConfig leaves them as None, this validator copies them from
        the model config. If both are set explicitly, they must agree.
        """
        if not isinstance(self.model, LSTMConfig):
            return self

        for field_name in ("lookback_hours", "timesteps"):
            model_val = getattr(self.model, field_name)
            feat_val = getattr(self.features, field_name)

            if feat_val is None:
                object.__setattr__(self.features, field_name, model_val)
            elif feat_val != model_val:
                raise ValueError(
                    f"{field_name} mismatch: features={feat_val}, model={model_val}. "
                    f"Remove from features section (auto-populated from model config)."
                )

        if self.features.adapter != "lstm":
            raise ValueError(
                f"LSTM strategy requires features.adapter='lstm', got '{self.features.adapter}'"
            )

        return self


# =============================================================================
# Top-Level Configuration Container
# =============================================================================


class MLTrainingConfig(BaseModel):
    """
    Top-level ML training configuration container.

    Combines all configuration sections and provides methods for
    loading from and saving to YAML/JSON files.

    Supports configuration inheritance via the `base` field. When specified,
    the base config is loaded first and the child config values are deep-merged
    on top, with child values taking precedence.

    Example YAML with inheritance:
        ```yaml
        # configs/training/xgboost_h2h_v2.yaml
        base: configs/training/xgboost_base.yaml

        experiment:
          name: "xgboost_h2h_v2"

        training:
          data:
            start_date: "2024-11-01"
          model:
            n_estimators: 200
        ```

    Example YAML without inheritance:
        ```yaml
        experiment:
          name: "xgboost_line_movement_v1"
          tags: ["xgboost", "line_movement", "h2h"]
          description: "XGBoost model for predicting line movements"

        training:
          strategy_type: "xgboost_line_movement"
          data:
            start_date: "2024-10-01"
            end_date: "2024-12-31"
            test_split: 0.2
            random_seed: 42
          model:
            n_estimators: 100
            max_depth: 6
            learning_rate: 0.1
          features:
            sharp_bookmakers: ["pinnacle"]
            retail_bookmakers: ["fanduel", "draftkings", "betmgm"]
          output_path: "models"

        tuning:
          n_trials: 100
          direction: "minimize"
          metric: "val_mse"
          search_spaces:
            n_estimators:
              type: int
              low: 50
              high: 500
            learning_rate:
              type: float
              low: 0.001
              high: 0.3
              log: true

        tracking:
          enabled: true
          tracking_uri: "mlruns"
        ```
    """

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(
        ...,
        description="Experiment metadata",
    )
    training: TrainingConfig = Field(
        ...,
        description="Training configuration",
    )
    tuning: TuningConfig | None = Field(
        default=None,
        description="Hyperparameter tuning configuration (optional)",
    )
    tracking: TrackingConfig | None = Field(
        default=None,
        description="Experiment tracking configuration (optional)",
    )

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> MLTrainingConfig:
        """
        Load configuration from a YAML file.

        Supports configuration inheritance via the `base` field. When a base
        config is specified, it is loaded recursively and the child config
        values are deep-merged on top.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            MLTrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid, fails validation, or has circular inheritance

        Example:
            >>> config = MLTrainingConfig.from_yaml("experiments/config.yaml")

        Example with inheritance:
            >>> # configs/child.yaml inherits from configs/base.yaml
            >>> config = MLTrainingConfig.from_yaml("configs/child.yaml")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Load with inheritance support
        data = cls._load_with_inheritance(filepath, visited=set(), depth=0)

        # Handle model config discriminated union
        data = cls._preprocess_model_config(data)

        return cls.model_validate(data)

    @classmethod
    def from_json(cls, filepath: str | Path) -> MLTrainingConfig:
        """
        Load configuration from a JSON file.

        Supports configuration inheritance via the `base` field. When a base
        config is specified, it is loaded recursively and the child config
        values are deep-merged on top.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            MLTrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid, fails validation, or has circular inheritance

        Example:
            >>> config = MLTrainingConfig.from_json("experiments/config.json")

        Example with inheritance:
            >>> # configs/child.json inherits from configs/base.json
            >>> config = MLTrainingConfig.from_json("configs/child.json")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Load with inheritance support
        data = cls._load_with_inheritance(filepath, visited=set(), depth=0)

        # Handle model config discriminated union
        data = cls._preprocess_model_config(data)

        return cls.model_validate(data)

    @classmethod
    def _load_with_inheritance(cls, filepath: Path, visited: set[str], depth: int) -> dict:
        """
        Load configuration file with inheritance support.

        Recursively loads base configs and deep-merges child values on top.
        Detects circular inheritance and enforces maximum depth limits.

        Args:
            filepath: Path to configuration file
            visited: Set of already-visited file paths (for circular detection)
            depth: Current inheritance depth

        Returns:
            Merged configuration dictionary

        Raises:
            ValueError: If circular inheritance detected or max depth exceeded
            FileNotFoundError: If file doesn't exist
        """
        # Check for circular inheritance
        resolved_path = str(filepath.resolve())
        if resolved_path in visited:
            raise ValueError(
                f"Circular inheritance detected: {filepath} was already loaded in the chain"
            )

        # Check max depth
        if depth > _MAX_INHERITANCE_DEPTH:
            raise ValueError(
                f"Maximum inheritance depth ({_MAX_INHERITANCE_DEPTH}) exceeded. "
                f"Check for excessive nesting in your config files."
            )

        # Load the file
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Determine file type and load
        suffix = filepath.suffix.lower()
        with open(filepath) as f:
            if suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json"
                )

        if data is None:
            raise ValueError(f"Empty configuration file: {filepath}")

        # Mark this file as visited
        visited.add(resolved_path)

        # Check for base config
        base_path = data.pop("base", None)
        if base_path is not None:
            # Resolve relative paths relative to the current config file
            base_filepath = Path(base_path)
            if not base_filepath.is_absolute():
                base_filepath = filepath.parent / base_filepath

            logger.debug(
                "loading_base_config",
                child_config=str(filepath),
                base_config=str(base_filepath),
                depth=depth,
            )

            # Recursively load base config
            base_data = cls._load_with_inheritance(base_filepath, visited.copy(), depth + 1)

            # Deep merge: child overrides base
            data = deep_merge(base_data, data)

        return data

    @classmethod
    def _preprocess_model_config(cls, data: dict) -> dict:
        """
        Preprocess configuration data to handle model type discrimination.

        Converts raw model config dict to appropriate typed config based on strategy_type.
        """
        if "training" not in data:
            return data

        training = data["training"]
        strategy_type = training.get("strategy_type", "")
        model_data = training.get("model", {})

        # Convert dict to appropriate config type based on strategy
        if isinstance(model_data, dict):
            if strategy_type in ("xgboost", "xgboost_line_movement"):
                training["model"] = XGBoostConfig.model_validate(model_data)
            elif strategy_type in ("lstm", "lstm_line_movement"):
                training["model"] = LSTMConfig.model_validate(model_data)

        return data

    def to_yaml(self, filepath: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            filepath: Path to save YAML configuration

        Example:
            >>> config.to_yaml("experiments/config_updated.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with proper serialization
        data = self._to_serializable_dict()

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def to_json(self, filepath: str | Path, indent: int = 2) -> None:
        """
        Save configuration to a JSON file.

        Args:
            filepath: Path to save JSON configuration
            indent: JSON indentation level

        Example:
            >>> config.to_json("experiments/config_updated.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with proper serialization
        data = self._to_serializable_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=str)

    def _to_serializable_dict(self) -> dict:
        """Convert configuration to a JSON/YAML serializable dictionary."""

        def serialize_value(value: Any) -> Any:
            if isinstance(value, date):
                return value.isoformat()
            elif isinstance(value, Path):
                return str(value)
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, BaseModel):
                return {k: serialize_value(v) for k, v in value.model_dump().items()}
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list | tuple):
                return [serialize_value(v) for v in value]
            return value

        return serialize_value(self.model_dump())

    def to_dict(self) -> dict:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return self._to_serializable_dict()

    @classmethod
    def from_dict(cls, data: dict) -> MLTrainingConfig:
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            MLTrainingConfig instance
        """
        data = cls._preprocess_model_config(data)
        return cls.model_validate(data)


# =============================================================================
# Utility Functions
# =============================================================================


def resolve_search_spaces(
    params: dict[str, Any],
    search_spaces: dict[str, SearchSpace],
) -> dict[str, Any]:
    """
    Resolve search spaces to concrete values using midpoints.

    When Optuna is not available, this function converts search space
    definitions to concrete values using the midpoint of the range
    (for int/float types) or the first choice (for categorical types).

    Args:
        params: Current parameter dictionary
        search_spaces: Search space definitions from tuning config

    Returns:
        Updated parameters with resolved values

    Example:
        >>> params = {"n_estimators": 100, "learning_rate": 0.1}
        >>> search_spaces = {
        ...     "n_estimators": SearchSpace(type="int", low=50, high=150),
        ...     "learning_rate": SearchSpace(type="float", low=0.01, high=0.3, log=True),
        ... }
        >>> resolved = resolve_search_spaces(params, search_spaces)
        >>> resolved["n_estimators"]  # 100 (midpoint of 50-150)
        >>> resolved["learning_rate"]  # ~0.055 (geometric mean)
    """
    resolved = params.copy()

    for param_name, space in search_spaces.items():
        if param_name not in resolved:
            logger.warning(
                "unknown_search_space_param",
                param_name=param_name,
                message=f"Search space defined for unknown parameter '{param_name}'",
            )
            continue

        if space.type == "int":
            # Use midpoint for integer parameters
            midpoint = int((space.low + space.high) / 2)
            if space.step:
                # Round to nearest step
                midpoint = int(round(midpoint / space.step) * space.step)
            resolved[param_name] = midpoint
            logger.debug(
                "resolved_search_space",
                param_name=param_name,
                value=midpoint,
                space_type="int",
                low=space.low,
                high=space.high,
            )

        elif space.type == "float":
            if space.log:
                # Use geometric mean for log-scale parameters
                midpoint = math.exp((math.log(space.low) + math.log(space.high)) / 2)
            else:
                # Use arithmetic mean for linear-scale parameters
                midpoint = (space.low + space.high) / 2
            resolved[param_name] = midpoint
            logger.debug(
                "resolved_search_space",
                param_name=param_name,
                value=midpoint,
                space_type="float",
                log_scale=space.log,
            )

        elif space.type == "categorical":
            # Use first choice for categorical parameters
            if space.choices:
                resolved[param_name] = space.choices[0]
                logger.debug(
                    "resolved_search_space",
                    param_name=param_name,
                    value=space.choices[0],
                    space_type="categorical",
                    choices=space.choices,
                )

    return resolved
