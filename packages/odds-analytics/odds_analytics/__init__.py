"""
Analytics and ML functionality for betting odds pipeline.

Provides strategies, backtesting, feature extraction, and game selection.

Heavy dependencies (torch, mlflow, scikit-learn, etc.) are optional —
install with ``pip install odds-analytics[training]`` for full functionality.
Modules that require training extras raise ImportError at import time if missing.
"""

from odds_core.odds_math import (
    american_to_decimal,
    calculate_implied_probability,
    calculate_profit_from_odds,
    decimal_to_american,
    determine_h2h_winner,
)

from odds_analytics.sequence_loader import load_sequences_for_event
from odds_analytics.utils import calculate_ev


def __getattr__(name: str) -> object:
    """Lazy-load modules that require optional training dependencies."""
    _training_imports: dict[str, tuple[str, str]] = {
        "LSTMLineMovementStrategy": (
            "odds_analytics.lstm_line_movement",
            "LSTMLineMovementStrategy",
        ),
        "LSTMModel": ("odds_analytics.lstm_line_movement", "LSTMModel"),
        "XGBoostLineMovementStrategy": (
            "odds_analytics.xgboost_line_movement",
            "XGBoostLineMovementStrategy",
        ),
        "BackfillExecutor": ("odds_analytics.backfill_executor", "BackfillExecutor"),
        "GameSelector": ("odds_analytics.game_selector", "GameSelector"),
        "ArbitrageStrategy": ("odds_analytics.strategies", "ArbitrageStrategy"),
        "BasicEVStrategy": ("odds_analytics.strategies", "BasicEVStrategy"),
        "FlatBettingStrategy": ("odds_analytics.strategies", "FlatBettingStrategy"),
        "MLTrainingConfig": ("odds_analytics.training", "MLTrainingConfig"),
        "TrainingConfig": ("odds_analytics.training", "TrainingConfig"),
        "ExperimentConfig": ("odds_analytics.training", "ExperimentConfig"),
        "DataConfig": ("odds_analytics.training", "DataConfig"),
        "XGBoostConfig": ("odds_analytics.training", "XGBoostConfig"),
        "LSTMConfig": ("odds_analytics.training", "LSTMConfig"),
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
    # Strategies
    "FlatBettingStrategy",
    "BasicEVStrategy",
    "ArbitrageStrategy",
    "LSTMLineMovementStrategy",
    "XGBoostLineMovementStrategy",
    # Models
    "LSTMModel",
    # Backfill
    "BackfillExecutor",
    "GameSelector",
    # Sequence Loading
    "load_sequences_for_event",
    # Utils
    "american_to_decimal",
    "decimal_to_american",
    "calculate_implied_probability",
    "calculate_ev",
    "calculate_profit_from_odds",
    "determine_h2h_winner",
    # Training Configuration
    "MLTrainingConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "DataConfig",
    "XGBoostConfig",
    "LSTMConfig",
    "FeatureConfig",
    "SearchSpace",
    "TuningConfig",
    "TrackingConfig",
]
