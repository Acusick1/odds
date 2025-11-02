"""Betting domain implementations.

This module provides betting-specific implementations of the core time series abstractions.
Works for any betting market: sports, elections, prediction markets, etc.
"""

from .observations import OddsObservation
from .problems import BettingEvent
from .strategies import BettingStrategy, RuleBasedStrategy

__all__ = [
    "BettingEvent",
    "OddsObservation",
    "BettingStrategy",
    "RuleBasedStrategy",
]
