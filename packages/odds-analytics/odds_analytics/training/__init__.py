"""
ML Training Configuration Module.

Configuration classes (FeatureConfig, MLTrainingConfig, etc.) are always available.
Training utilities (cross-validation, tuning, tracking) require the ``[training]``
optional extra and are lazy-loaded on first access.
"""

from odds_analytics.training.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    SamplingConfig,
    SearchSpace,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)

# Names loaded from modules that need the training extra, on first access, by static imports
# the rule hash can follow.
_CROSS_VALIDATION: frozenset[str] = frozenset(
    {"CVFoldResult", "CVResult", "TrainableStrategy", "run_cv", "train_with_cv"}
)
_DATA_PREPARATION: frozenset[str] = frozenset(
    {
        "TrainingDataResult",
        "filter_events_by_date_range",
        "prepare_training_data_from_config",
        "save_verification_artifacts",
    }
)
_TRACKING: frozenset[str] = frozenset({"ExperimentTracker", "MLflowTracker", "create_tracker"})
_TUNER: frozenset[str] = frozenset({"HyperparameterTuner", "OptunaTuner", "create_objective"})
_UTILS: frozenset[str] = frozenset({"compute_regression_metrics", "flatten_config_for_tracking"})


def __getattr__(name: str) -> object:
    if name in _CROSS_VALIDATION:
        from odds_analytics.training import cross_validation as module
    elif name in _DATA_PREPARATION:
        from odds_analytics.training import data_preparation as module
    elif name in _TRACKING:
        from odds_analytics.training import tracking as module
    elif name in _TUNER:
        from odds_analytics.training import tuner as module
    elif name in _UTILS:
        from odds_analytics.training import utils as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)


__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "SamplingConfig",
    "FeatureConfig",
    "SearchSpace",
    "TrainingConfig",
    "TuningConfig",
    "TrackingConfig",
    "MLTrainingConfig",
    "prepare_training_data_from_config",
    "save_verification_artifacts",
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
