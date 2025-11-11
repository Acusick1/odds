"""
Analytics and ML functionality for betting odds pipeline.

Provides strategies, backtesting, feature extraction, and game selection.
"""

from odds_analytics.backfill_executor import BackfillExecutor
from odds_analytics.game_selector import GameSelector
from odds_analytics.strategies import (
    ArbitrageStrategy,
    BasicEVStrategy,
    FlatBettingStrategy,
)
from odds_analytics.utils import (
    american_to_decimal,
    american_to_implied_probability,
    calculate_expected_value,
    decimal_to_american,
)

__all__ = [
    # Strategies
    "FlatBettingStrategy",
    "BasicEVStrategy",
    "ArbitrageStrategy",
    # Backfill
    "BackfillExecutor",
    "GameSelector",
    # Utils
    "american_to_decimal",
    "decimal_to_american",
    "american_to_implied_probability",
    "calculate_expected_value",
]
