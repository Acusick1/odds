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

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "LSTMConfig",
    "FeatureConfig",
    "SearchSpace",
    "TrainingConfig",
    "TuningConfig",
    "TrackingConfig",
    "MLTrainingConfig",
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
        description="Fraction of training data to use for validation (0.0 to 1.0)",
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
# Feature Configuration
# =============================================================================


class FeatureConfig(BaseModel):
    """
    Feature extraction configuration.

    Controls how features are extracted from raw odds data for model training.
    """

    model_config = ConfigDict(extra="forbid")

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

    # Timing configuration
    opening_hours_before: float = Field(
        default=48.0,
        gt=0.0,
        le=168.0,
        description="Hours before game for opening line snapshot",
    )
    closing_hours_before: float = Field(
        default=0.5,
        ge=0.0,
        le=24.0,
        description="Hours before game for closing line snapshot",
    )

    @model_validator(mode="after")
    def validate_timing(self) -> FeatureConfig:
        """Ensure opening is before closing."""
        if self.opening_hours_before <= self.closing_hours_before:
            raise ValueError(
                f"opening_hours_before ({self.opening_hours_before}) must be greater than "
                f"closing_hours_before ({self.closing_hours_before})"
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
    Experiment tracking configuration for MLflow.

    Controls logging and artifact storage for experiment reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable MLflow tracking",
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


# =============================================================================
# Top-Level Configuration Container
# =============================================================================


class MLTrainingConfig(BaseModel):
    """
    Top-level ML training configuration container.

    Combines all configuration sections and provides methods for
    loading from and saving to YAML/JSON files.

    Example YAML:
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

        Args:
            filepath: Path to YAML configuration file

        Returns:
            MLTrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or fails validation

        Example:
            >>> config = MLTrainingConfig.from_yaml("experiments/config.yaml")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {filepath}")

        # Handle model config discriminated union
        data = cls._preprocess_model_config(data)

        return cls.model_validate(data)

    @classmethod
    def from_json(cls, filepath: str | Path) -> MLTrainingConfig:
        """
        Load configuration from a JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            MLTrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or fails validation

        Example:
            >>> config = MLTrainingConfig.from_json("experiments/config.json")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        # Handle model config discriminated union
        data = cls._preprocess_model_config(data)

        return cls.model_validate(data)

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
            elif isinstance(value, BaseModel):
                return {k: serialize_value(v) for k, v in value.model_dump().items()}
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
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
