"""
ML Training Configuration Module.

This module provides a Pydantic-based configuration system for ML model training
that supports YAML/JSON loading and future Optuna/MLflow integration.

Example usage:
    ```python
    from odds_analytics.training import MLTrainingConfig

    # Load from YAML
    config = MLTrainingConfig.from_yaml("experiments/xgboost_v1.yaml")

    # Access nested configuration
    print(config.training.strategy_type)
    print(config.training.model.n_estimators)
    ```
"""

from odds_analytics.training.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    LSTMConfig,
    MLTrainingConfig,
    SamplingConfig,
    SearchSpace,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)
from odds_analytics.training.cross_validation import (
    CVFoldResult,
    CVResult,
    TrainableStrategy,
    run_cv,
    train_with_cv,
)
from odds_analytics.training.data_preparation import (
    TrainingDataResult,
    filter_events_by_date_range,
    prepare_training_data_from_config,
)
from odds_analytics.training.tracking import (
    ExperimentTracker,
    MLflowTracker,
    create_tracker,
)
from odds_analytics.training.tuner import (
    HyperparameterTuner,
    OptunaTuner,
    create_objective,
)
from odds_analytics.training.utils import compute_regression_metrics, flatten_config_for_tracking

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "LSTMConfig",
    "SamplingConfig",
    "FeatureConfig",
    "SearchSpace",
    "TrainingConfig",
    "TuningConfig",
    "TrackingConfig",
    "MLTrainingConfig",
    "prepare_training_data_from_config",
    "filter_events_by_date_range",
    "TrainingDataResult",
    "CVFoldResult",
    "CVResult",
    "TrainableStrategy",
    "run_cv",
    "train_with_cv",
    "ExperimentTracker",
    "MLflowTracker",
    "create_tracker",
    "compute_regression_metrics",
    "flatten_config_for_tracking",
    "HyperparameterTuner",
    "OptunaTuner",
    "create_objective",
]
