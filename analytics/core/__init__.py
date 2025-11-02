"""Universal time series prediction abstractions.

This module provides domain-agnostic abstractions for time series prediction problems.
It supports both discrete event predictions (sports, elections) and continuous predictions
(stock returns, price movements).
"""

from .features import FeatureExtractor, FeatureSet, SequentialFeatureSet, TabularFeatureSet
from .models import ModelPredictor, Prediction
from .observations import Observation
from .problems import ContinuousPredictionProblem, DiscreteEventProblem, PredictionProblem

__all__ = [
    # Problems
    "PredictionProblem",
    "DiscreteEventProblem",
    "ContinuousPredictionProblem",
    # Observations
    "Observation",
    # Features
    "FeatureSet",
    "TabularFeatureSet",
    "SequentialFeatureSet",
    "FeatureExtractor",
    # Models
    "Prediction",
    "ModelPredictor",
]
