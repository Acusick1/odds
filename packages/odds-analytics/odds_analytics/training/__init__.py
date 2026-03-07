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
    LSTMConfig,
    MLTrainingConfig,
    SamplingConfig,
    SearchSpace,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CVFoldResult": ("odds_analytics.training.cross_validation", "CVFoldResult"),
    "CVResult": ("odds_analytics.training.cross_validation", "CVResult"),
    "TrainableStrategy": ("odds_analytics.training.cross_validation", "TrainableStrategy"),
    "run_cv": ("odds_analytics.training.cross_validation", "run_cv"),
    "train_with_cv": ("odds_analytics.training.cross_validation", "train_with_cv"),
    "TrainingDataResult": ("odds_analytics.training.data_preparation", "TrainingDataResult"),
    "filter_events_by_date_range": (
        "odds_analytics.training.data_preparation",
        "filter_events_by_date_range",
    ),
    "prepare_training_data_from_config": (
        "odds_analytics.training.data_preparation",
        "prepare_training_data_from_config",
    ),
    "ExperimentTracker": ("odds_analytics.training.tracking", "ExperimentTracker"),
    "MLflowTracker": ("odds_analytics.training.tracking", "MLflowTracker"),
    "create_tracker": ("odds_analytics.training.tracking", "create_tracker"),
    "HyperparameterTuner": ("odds_analytics.training.tuner", "HyperparameterTuner"),
    "OptunaTuner": ("odds_analytics.training.tuner", "OptunaTuner"),
    "create_objective": ("odds_analytics.training.tuner", "create_objective"),
    "compute_regression_metrics": ("odds_analytics.training.utils", "compute_regression_metrics"),
    "flatten_config_for_tracking": ("odds_analytics.training.utils", "flatten_config_for_tracking"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
