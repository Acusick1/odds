"""
Feature extraction abstractions for sports betting ML models.

This module provides extensible feature extraction patterns that decouple
feature engineering from betting strategy implementations, enabling support
for different model architectures.

Architecture:
- FeatureExtractor: Abstract base class defining the interface
- TabularFeatureExtractor: Snapshot-based features for tabular models (XGBoost, Random Forest)

Example:
    ```python
    # Tabular model (XGBoost)
    extractor = TabularFeatureExtractor()
    features = extractor.extract_features(event, quoted, outcome=PriceOutcome.HOME)
    vector = extractor.create_feature_vector(features)
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from odds_core.models import EventStatus
from odds_core.quotes import Quote
from odds_core.time import to_eastern
from odds_core.vocabulary import PriceOutcome
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

__all__ = [
    "FeatureEvent",
    "FeatureExtractor",
    "TabularFeatureExtractor",
    "TabularFeatures",
    "DEFAULT_SHARP_BOOKMAKERS",
    "DEFAULT_RETAIL_BOOKMAKERS",
    "select_sharp_quotes",
]


class FeatureEvent(BaseModel):
    """The event fields feature extraction reads.

    Scores are ``None`` for events not yet played, scored at inference time.
    """

    id: str
    commence_time: datetime
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    status: EventStatus

    @field_validator("commence_time")
    @classmethod
    def _must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("commence_time must be timezone-aware")
        return v


# =============================================================================
# Quote Selection
# =============================================================================

DEFAULT_SHARP_BOOKMAKERS = ["pinnacle"]
DEFAULT_RETAIL_BOOKMAKERS = ["fanduel", "draftkings", "betmgm_us"]


def select_sharp_quotes(
    quoted: Sequence[Quote],
    sharp_bookmakers: list[str],
    outcome: PriceOutcome,
) -> list[Quote]:
    """``outcome`` quotes of the first sharp bookmaker, in priority order, that quotes it."""
    outcome_quotes = [q for q in quoted if q.outcome is outcome]
    for bm in sharp_bookmakers:
        sharp = [q for q in outcome_quotes if q.bookmaker == bm]
        if sharp:
            return sharp
    return []


# =============================================================================
# Feature Dataclasses
# =============================================================================


@dataclass
class TabularFeatures:
    """
    Type-safe feature container for tabular ML models.

    All features are outcome-relative (computed for the specific outcome being
    analyzed), eliminating structural duplicates between home/away sides.

    None values are converted to np.nan in array representation to distinguish
    "feature unavailable" from "calculated value equals zero".
    """

    # Consensus probability (optional - requires h2h market data)
    consensus_prob: float | None = None

    # Sharp bookmaker features (optional - require sharp books)
    sharp_prob: float | None = None

    # Retail vs sharp features (optional - require both sharp and retail)
    retail_sharp_diff: float | None = None

    # Market maturity (optional - require bookmaker data)
    num_bookmakers: float | None = None

    # Calendar features (derived from commence_time)
    is_weekend: float | None = None
    day_of_week: float | None = None

    def to_array(self) -> np.ndarray:
        """
        Convert features to numpy array for model input.

        None values are converted to np.nan to distinguish "feature unavailable"
        from "calculated value equals zero".

        Returns:
            Numpy array with shape (num_features,) where None → np.nan
        """
        return np.array(
            [
                getattr(self, field.name) if getattr(self, field.name) is not None else np.nan
                for field in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """
        Return ordered list of feature names.

        Returns:
            List of feature names in the order they appear in to_array()
        """
        return [field.name for field in fields(cls)]


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    Feature extractors transform raw odds data into model-ready features.
    Tabular models (XGBoost) take a single snapshot as a flat feature vector.

    Subclasses must implement:
    - extract_features(): Extract features from event and odds data
    - get_feature_names(): Return ordered list of feature names (for tabular models)
    """

    @abstractmethod
    def extract_features(
        self,
        event: FeatureEvent,
        quoted: Sequence[Quote],
        outcome: PriceOutcome | None = None,
        **kwargs,
    ) -> TabularFeatures | dict[str, Any]:
        """
        Extract features from event and one snapshot's quotes.

        Args:
            event: Event with final scores
            quoted: One snapshot's quotes of one market
            outcome: Specific outcome to analyze
            **kwargs: Additional extractor-specific parameters

        Returns:
            TabularFeatures instance
        """

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return ordered list of feature names.

        For tabular models: List of scalar feature names

        Returns:
            Ordered list of feature names
        """

    def create_feature_vector(
        self, features: TabularFeatures | dict[str, float], feature_names: list[str] | None = None
    ) -> np.ndarray:
        """
        Convert features to numpy array for model input.

        Args:
            features: TabularFeatures instance or feature dictionary from extract_features()
            feature_names: Ordered list of feature names (uses get_feature_names() if None)

        Returns:
            Numpy array of feature values (fills missing with 0.0 for dicts, np.nan for TabularFeatures)
        """
        # If features is a TabularFeatures instance, use its to_array() method
        if isinstance(features, TabularFeatures):
            return features.to_array()

        # Otherwise, treat as dictionary (legacy support)
        if feature_names is None:
            feature_names = self.get_feature_names()

        return np.array([features.get(name, 0.0) for name in feature_names])


class TabularFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for tabular ML models (XGBoost, Random Forest, etc.).

    Extracts features from a single odds snapshot focused on line movement
    prediction signals:
    - Market consensus probability
    - Sharp vs retail bookmaker divergence (strongest publicly observable signal)
    - Market maturity (bookmaker count)
    """

    def __init__(
        self,
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
    ):
        """
        Initialize tabular feature extractor.

        Args:
            sharp_bookmakers: Sharp bookmakers for "true" odds (default: DEFAULT_SHARP_BOOKMAKERS)
            retail_bookmakers: Retail bookmakers for comparison (default: DEFAULT_RETAIL_BOOKMAKERS)
        """
        self.sharp_bookmakers = sharp_bookmakers or DEFAULT_SHARP_BOOKMAKERS
        self.retail_bookmakers = retail_bookmakers or DEFAULT_RETAIL_BOOKMAKERS

    @classmethod
    def from_config(cls, config: FeatureConfig) -> TabularFeatureExtractor:
        """
        Create a TabularFeatureExtractor from a FeatureConfig object.

        This factory method enables configuration-driven extractor instantiation,
        mapping all relevant FeatureConfig fields to extractor parameters.

        Args:
            config: FeatureConfig object with extraction parameters

        Returns:
            Configured TabularFeatureExtractor instance

        Example:
            ```python
            from odds_analytics.training.config import FeatureConfig

            config = FeatureConfig(
                target_type="raw",
                sharp_bookmakers=["pinnacle", "circa"],
                retail_bookmakers=["fanduel", "draftkings"],
            )
            extractor = TabularFeatureExtractor.from_config(config)
            ```
        """
        return cls(
            sharp_bookmakers=config.sharp_bookmakers,
            retail_bookmakers=config.retail_bookmakers,
        )

    def extract_features(
        self,
        event: FeatureEvent,
        quoted: Sequence[Quote],
        outcome: PriceOutcome | None = None,
        **kwargs,
    ) -> TabularFeatures:
        """
        Extract ML features from event and one snapshot's quotes.

        Args:
            event: Event with final scores
            quoted: Quotes at decision time (one snapshot of one market)
            outcome: Specific outcome to analyze (if None, only calendar and maturity features)

        Returns:
            TabularFeatures instance with type-safe feature access
        """
        # Initialize feature values dictionary for building
        feature_values: dict[str, float] = {}

        # Calendar features — convert to US Eastern so midnight UTC rollover
        # doesn't shift US evening games to the next calendar day.
        weekday = to_eastern(event.commence_time).weekday()
        feature_values["day_of_week"] = float(weekday)
        feature_values["is_weekend"] = 1.0 if weekday >= 5 else 0.0

        if not quoted:
            return TabularFeatures(**feature_values)

        retail_quotes = [q for q in quoted if q.bookmaker in self.retail_bookmakers]

        # 1. Consensus probability
        if outcome is not None:
            outcome_quotes = [q for q in quoted if q.outcome is outcome]
            if outcome_quotes:
                feature_values["consensus_prob"] = float(
                    np.mean([1 / q.price for q in outcome_quotes])
                )

        # 2. Sharp vs Retail features
        if retail_quotes and outcome is not None:
            sharp_quotes = select_sharp_quotes(quoted, self.sharp_bookmakers, outcome)
            if sharp_quotes:
                outcome_sharp_prob = 1 / sharp_quotes[0].price
                feature_values["sharp_prob"] = outcome_sharp_prob

                # Retail-sharp diff for outcome side only
                retail_outcome = [q for q in retail_quotes if q.outcome is outcome]
                if retail_outcome:
                    avg_retail_prob = float(np.mean([1 / q.price for q in retail_outcome]))
                    feature_values["retail_sharp_diff"] = avg_retail_prob - outcome_sharp_prob

        # 3. Market maturity
        feature_values["num_bookmakers"] = float(len({q.bookmaker for q in quoted}))

        return TabularFeatures(**feature_values)

    def get_feature_names(self) -> list[str]:
        """
        Return ordered list of feature names.

        Note: Not all features may be present for every event (depends on
        available bookmakers and market type). Missing features are filled
        with np.nan in create_feature_vector() for TabularFeatures.

        Returns:
            Ordered list of all possible feature names
        """
        return TabularFeatures.get_feature_names()
