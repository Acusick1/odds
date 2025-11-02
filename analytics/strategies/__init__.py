"""Betting strategies for backtesting.

This module exports all available betting strategies:
- Rule-based strategies (logic and statistical calculations)
- ML-based strategies (machine learning models with feature extraction)

All strategies work with clean domain objects (BettingEvent, OddsObservation)
and are automatically compatible with BacktestEngine.
"""

from analytics.strategies.ml_betting_strategy import MLBettingStrategy
from analytics.strategies.rule_based import (
    ArbitrageStrategy,
    BasicEVStrategy,
    FlatBettingStrategy,
)

__all__ = [
    # Rule-based strategies
    "FlatBettingStrategy",
    "BasicEVStrategy",
    "ArbitrageStrategy",
    # ML strategies
    "MLBettingStrategy",
]
