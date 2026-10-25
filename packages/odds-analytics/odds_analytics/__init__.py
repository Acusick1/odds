"""
Analytics and ML functionality for betting odds pipeline.

Provides candidate scoring and replay, feature extraction, model training, and game selection.

Heavy dependencies (mlflow, scikit-learn, etc.) are optional —
install with ``pip install odds-analytics[training]`` for full functionality.
Modules that require training extras raise ImportError at import time if missing.
"""

from odds_analytics.utils import calculate_ev


def __getattr__(name: str) -> object:
    """Lazy-load modules that require optional training dependencies."""
    _training_imports: dict[str, tuple[str, str]] = {
        "XGBoostLineMovementStrategy": (
            "odds_analytics.xgboost_line_movement",
            "XGBoostLineMovementStrategy",
        ),
        "BackfillExecutor": ("odds_analytics.backfill_executor", "BackfillExecutor"),
        "GameSelector": ("odds_analytics.game_selector", "GameSelector"),
        "MLTrainingConfig": ("odds_analytics.training", "MLTrainingConfig"),
        "TrainingConfig": ("odds_analytics.training", "TrainingConfig"),
        "ExperimentConfig": ("odds_analytics.training", "ExperimentConfig"),
        "DataConfig": ("odds_analytics.training", "DataConfig"),
        "XGBoostConfig": ("odds_analytics.training", "XGBoostConfig"),
        "FeatureConfig": ("odds_analytics.training", "FeatureConfig"),
        "SearchSpace": ("odds_analytics.training", "SearchSpace"),
        "TuningConfig": ("odds_analytics.training", "TuningConfig"),
        "TrackingConfig": ("odds_analytics.training", "TrackingConfig"),
    }
    if name in _training_imports:
        import importlib

        module_path, attr = _training_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'odds_analytics' has no attribute {name!r}")


__all__ = [
    # Models
    "XGBoostLineMovementStrategy",
    # Backfill
    "BackfillExecutor",
    "GameSelector",
    # Utils
    "calculate_ev",
    # Training Configuration
    "MLTrainingConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "FeatureConfig",
    "SearchSpace",
    "TuningConfig",
    "TrackingConfig",
]
