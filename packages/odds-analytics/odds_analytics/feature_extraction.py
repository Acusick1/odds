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
    features = extractor.extract_features(event, odds_snapshot, outcome="Lakers")
    vector = extractor.create_feature_vector(features)
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from odds_core.models import EventStatus, Odds
from odds_core.time import to_eastern
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

__all__ = [
    "FeatureEvent",
    "FeatureExtractor",
    "TabularFeatureExtractor",
    "TabularFeatures",
    # Utility functions
    "DEFAULT_SHARP_BOOKMAKERS",
    "DEFAULT_RETAIL_BOOKMAKERS",
    "filter_odds_by_market_outcome",
    "filter_odds_by_bookmakers",
    "calculate_avg_price",
    "calculate_avg_probability",
    "calculate_sharp_retail_metrics",
    "select_sharp_odds",
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
# Odds Filtering and Aggregation Utilities
# =============================================================================

DEFAULT_SHARP_BOOKMAKERS = ["pinnacle"]
DEFAULT_RETAIL_BOOKMAKERS = ["fanduel", "draftkings", "betmgm"]


def filter_odds_by_market_outcome(
    odds: list[Odds],
    market: str,
    outcome: str | None = None,
) -> list[Odds]:
    """Filter odds by market and optionally by specific outcome."""
    filtered = [o for o in odds if o.market_key == market]
    if outcome:
        filtered = [o for o in filtered if o.outcome_name == outcome]
    return filtered


def filter_odds_by_bookmakers(
    odds: list[Odds],
    bookmakers: list[str],
) -> list[Odds]:
    """Filter odds to only those from specified bookmakers."""
    return [o for o in odds if o.bookmaker_key in bookmakers]


def select_sharp_odds(
    odds: list[Odds],
    sharp_bookmakers: list[str],
    outcome: str,
) -> list[Odds]:
    """Return ``outcome`` rows of the first sharp bookmaker, in priority order, that quotes it."""
    outcome_odds = [o for o in odds if o.outcome_name == outcome]
    for bm in sharp_bookmakers:
        sharp_odds = filter_odds_by_bookmakers(outcome_odds, [bm])
        if sharp_odds:
            return sharp_odds
    return []


def calculate_avg_price(odds: list[Odds]) -> float:
    """Calculate the mean American quote across list."""
    if not odds:
        return 0.0
    return float(np.mean([o.american_price for o in odds]))


def calculate_avg_probability(odds: list[Odds]) -> float:
    """Calculate average implied probability across odds list."""
    if not odds:
        return 0.0
    return float(np.mean([o.implied_probability for o in odds]))


def calculate_sharp_retail_metrics(
    odds: list[Odds],
    sharp_bookmakers: list[str],
    retail_bookmakers: list[str],
    outcome: str,
) -> dict[str, float | None]:
    """
    Calculate sharp vs retail probability metrics.

    Returns dict with:
    - sharp_price: Average sharp bookmaker price
    - sharp_prob: Average sharp bookmaker probability
    - retail_price: Average retail bookmaker price
    - retail_prob: Average retail bookmaker probability
    - diff: Differential (retail_prob - sharp_prob)
    """
    sharp_odds = select_sharp_odds(odds, sharp_bookmakers, outcome)
    retail_odds = filter_odds_by_bookmakers(
        [o for o in odds if o.outcome_name == outcome], retail_bookmakers
    )

    result: dict[str, float | None] = {
        "sharp_price": None,
        "sharp_prob": None,
        "retail_price": None,
        "retail_prob": None,
        "diff": None,
    }

    if sharp_odds:
        result["sharp_price"] = calculate_avg_price(sharp_odds)
        result["sharp_prob"] = calculate_avg_probability(sharp_odds)

    if retail_odds:
        result["retail_price"] = calculate_avg_price(retail_odds)
        result["retail_prob"] = calculate_avg_probability(retail_odds)

    if result["sharp_prob"] is not None and result["retail_prob"] is not None:
        result["diff"] = result["retail_prob"] - result["sharp_prob"]

    return result


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
        odds_data: list[Odds] | list[list[Odds]],
        outcome: str | None = None,
        **kwargs,
    ) -> TabularFeatures | dict[str, Any]:
        """
        Extract features from event and odds data.

        Args:
            event: Event with final scores
            odds_data: Odds snapshot(s)
            outcome: Specific outcome to analyze (team name or "Over"/"Under")
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
        odds_data: list[Odds],
        outcome: str | None = None,
        market: str = "h2h",
        **kwargs,
    ) -> TabularFeatures:
        """
        Extract ML features from event and odds snapshot.

        Args:
            event: Event with final scores
            odds_data: Odds at decision time (single snapshot)
            outcome: Specific outcome to analyze (if None, analyzes both sides)
            market: Market to analyze (h2h, spreads, totals)

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

        # Filter for target market
        market_odds = filter_odds_by_market_outcome(odds_data, market)

        if not market_odds:
            return TabularFeatures(**feature_values)

        retail_odds = filter_odds_by_bookmakers(market_odds, self.retail_bookmakers)

        # Resolve outcome (team name for h2h, "Over"/"Under" for totals)
        if outcome in ("Over", "Under"):
            outcome_team = outcome
        elif outcome == event.home_team:
            outcome_team = event.home_team
        elif outcome == event.away_team:
            outcome_team = event.away_team
        else:
            outcome_team = None

        # 1. Consensus probability
        if outcome_team:
            outcome_odds_list = filter_odds_by_market_outcome(market_odds, market, outcome_team)
            if outcome_odds_list:
                avg_outcome_prob = float(
                    np.mean([o.implied_probability for o in outcome_odds_list])
                )
                feature_values["consensus_prob"] = avg_outcome_prob

        # 2. Sharp vs Retail features
        if retail_odds and outcome_team:
            sharp_outcome_odds = select_sharp_odds(market_odds, self.sharp_bookmakers, outcome_team)
            sharp_outcome = sharp_outcome_odds[0] if sharp_outcome_odds else None
            if sharp_outcome:
                outcome_sharp_prob = sharp_outcome.implied_probability
                feature_values["sharp_prob"] = outcome_sharp_prob

                # Retail-sharp diff for outcome side only
                retail_outcome_odds = filter_odds_by_market_outcome(
                    retail_odds, market, outcome_team
                )
                if retail_outcome_odds:
                    avg_retail_prob = float(
                        np.mean([o.implied_probability for o in retail_outcome_odds])
                    )
                    feature_values["retail_sharp_diff"] = avg_retail_prob - outcome_sharp_prob

        # 3. Market maturity
        all_books = {o.bookmaker_key for o in market_odds}
        feature_values["num_bookmakers"] = float(len(all_books))

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
