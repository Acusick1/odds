"""
Analytics and ML functionality for betting odds pipeline.

Provides strategies, backtesting, feature extraction, and game selection.
"""

from odds_analytics.backfill_executor import BackfillExecutor
from odds_analytics.game_selector import GameSelector
from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy, LSTMModel
from odds_analytics.sequence_loader import load_sequences_for_event
from odds_analytics.strategies import (
    ArbitrageStrategy,
    BasicEVStrategy,
    FlatBettingStrategy,
)
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    LSTMConfig,
    MLTrainingConfig,
    SearchSpace,
    TrackingConfig,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)
from odds_analytics.utils import (
    american_to_decimal,
    calculate_ev,
    calculate_implied_probability,
    decimal_to_american,
)
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy

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
