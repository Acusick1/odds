"""Feature engineering modules.

Provides feature extractors for different model types and domain-specific
feature computation functions.
"""

from .betting_features import (
    compute_consensus_odds,
    compute_line_movement,
    compute_market_hold,
    compute_sharp_retail_diff,
    compute_timing_features,
)
from .sequential import SequentialFeatureExtractor
from .tabular import TabularFeatureExtractor

__all__ = [
    # Extractors
    "TabularFeatureExtractor",
    "SequentialFeatureExtractor",
    # Betting feature computers
    "compute_sharp_retail_diff",
    "compute_market_hold",
    "compute_line_movement",
    "compute_consensus_odds",
    "compute_timing_features",
]
