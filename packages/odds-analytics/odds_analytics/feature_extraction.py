"""
Feature extraction abstractions for sports betting ML models.

This module provides extensible feature extraction patterns that decouple
feature engineering from betting strategy implementations, enabling support
for different model architectures (XGBoost, LSTM, Transformers, ensembles).

Architecture:
- FeatureExtractor: Abstract base class defining the interface
- TabularFeatureExtractor: Snapshot-based features for tabular models (XGBoost, Random Forest)
- SequenceFeatureExtractor: Time-series features for sequence models (LSTM, Transformers)

Example:
    ```python
    # Tabular model (XGBoost)
    extractor = TabularFeatureExtractor()
    features = extractor.extract_features(event, odds_snapshot, outcome="Lakers")
    vector = extractor.create_feature_vector(features)

    # Sequence model (LSTM) - future implementation
    extractor = SequenceFeatureExtractor(lookback_hours=72, timesteps=24)
    sequence = extractor.extract_features(event, odds_history, outcome="Lakers")
    tensor = extractor.create_feature_tensor(sequence)  # Shape: (timesteps, features)
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from odds_core.models import Odds

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.utils import (
    american_to_decimal,
    calculate_implied_probability,
    calculate_market_hold,
)

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

__all__ = [
    "FeatureExtractor",
    "TabularFeatureExtractor",
    "SequenceFeatureExtractor",
    "TrajectoryFeatureExtractor",
    "HybridFeatureExtractor",
    "TabularFeatures",
    "SequenceFeatures",
    "TrajectoryFeatures",
    "HybridFeatures",
    # Utility functions
    "DEFAULT_SHARP_BOOKMAKERS",
    "DEFAULT_RETAIL_BOOKMAKERS",
    "filter_odds_by_market_outcome",
    "filter_odds_by_bookmakers",
    "calculate_avg_price",
    "calculate_avg_probability",
    "calculate_sharp_retail_metrics",
]


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


def calculate_avg_price(odds: list[Odds]) -> float:
    """Calculate average American odds price across list."""
    if not odds:
        return 0.0
    return float(np.mean([o.price for o in odds]))


def calculate_avg_probability(odds: list[Odds]) -> float:
    """Calculate average implied probability across odds list."""
    if not odds:
        return 0.0
    avg_price = calculate_avg_price(odds)
    return calculate_implied_probability(int(avg_price))


def calculate_sharp_retail_metrics(
    odds: list[Odds],
    sharp_bookmakers: list[str],
    retail_bookmakers: list[str],
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
    sharp_odds = filter_odds_by_bookmakers(odds, sharp_bookmakers)
    retail_odds = filter_odds_by_bookmakers(odds, retail_bookmakers)

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

    Replaces string-keyed dictionaries with typed attributes to eliminate
    runtime errors from typos and provide compile-time type checking.

    Fields marked as Optional (with None default) indicate features that may
    be unavailable depending on data completeness. None values are converted
    to np.nan in array representation to distinguish from calculated zeros.

    Required fields (always present):
    - is_home_team, is_away_team: Team indicators

    Optional fields (depend on data availability):
    - Consensus features: Require h2h market with multiple bookmakers
    - Sharp features: Require sharp bookmaker data (e.g., Pinnacle)
    - Retail features: Require retail bookmaker data
    - Best odds features: Require multiple bookmaker quotes
    """

    # Required fields - always present
    is_home_team: float  # 1.0 or 0.0
    is_away_team: float  # 1.0 or 0.0

    # Consensus features (optional - require h2h market data)
    avg_home_odds: float | None = None
    avg_away_odds: float | None = None
    std_home_odds: float | None = None
    std_away_odds: float | None = None
    home_consensus_prob: float | None = None
    away_consensus_prob: float | None = None
    consensus_prob: float | None = None
    opponent_consensus_prob: float | None = None

    # Sharp bookmaker features (optional - require sharp books)
    sharp_home_prob: float | None = None
    sharp_away_prob: float | None = None
    sharp_market_hold: float | None = None
    sharp_prob: float | None = None
    opponent_sharp_prob: float | None = None

    # Retail vs sharp features (optional - require both sharp and retail)
    retail_sharp_diff_home: float | None = None
    retail_sharp_diff_away: float | None = None

    # Market efficiency features (optional - require bookmaker data)
    num_bookmakers: float | None = None
    avg_market_hold: float | None = None
    std_market_hold: float | None = None

    # Best odds features (optional - require line shopping data)
    best_home_odds: float | None = None
    worst_home_odds: float | None = None
    home_odds_range: float | None = None
    best_away_odds: float | None = None
    worst_away_odds: float | None = None
    away_odds_range: float | None = None
    best_available_odds: float | None = None
    odds_range: float | None = None
    best_available_decimal: float | None = None

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


@dataclass
class SequenceFeatures:
    """
    Type-safe feature container for sequence models (LSTM, Transformers).

    Represents features for a single timestep in a sequence. Multiple
    SequenceFeatures instances are stacked to form time series.

    Required fields (always present when odds exist):
    - american_odds, decimal_odds, implied_prob: Basic odds representation
    - num_bookmakers: Count of bookmakers in snapshot
    - hours_to_game: Time until game start
    - time_of_day_sin, time_of_day_cos: Cyclical time encoding

    Optional fields (depend on data availability):
    - Line movement features: Require previous snapshots
    - Sharp features: Require sharp bookmaker data
    - Retail features: Require retail bookmaker data
    """

    # Required fields - always present when odds exist
    american_odds: float
    decimal_odds: float
    implied_prob: float
    num_bookmakers: float
    hours_to_game: float
    time_of_day_sin: float
    time_of_day_cos: float

    # Optional line movement features (require previous snapshots)
    odds_change_from_prev: float | None = None
    odds_change_from_opening: float | None = None
    implied_prob_change_from_prev: float | None = None
    implied_prob_change_from_opening: float | None = None

    # Optional market features
    odds_std: float | None = None

    # Optional sharp vs retail features (require sharp/retail bookmakers)
    sharp_odds: float | None = None
    sharp_prob: float | None = None
    retail_sharp_diff: float | None = None

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


@dataclass
class TrajectoryFeatures:
    """
    Aggregate features computed from odds sequence up to decision point.

    These features capture patterns across snapshots from opening to decision time,
    computed WITHOUT using closing data to avoid look-ahead bias / data leakage.

    The sequence should be filtered to only include snapshots up to decision_tier
    BEFORE being passed to the extractor. Features describe "what happened so far"
    at decision time, NOT what will happen between decision and closing.

    All fields are optional since they depend on having sufficient snapshots.
    """

    # Momentum features (direction and strength of movement TO DECISION POINT)
    prob_change_to_decision: float | None = None  # Opening to decision point delta
    avg_change_rate_to_decision: float | None = None  # Rate of change (per hour)
    max_prob_increase: float | None = None  # Largest single-snapshot increase
    max_prob_decrease: float | None = None  # Largest single-snapshot decrease
    net_movement_direction: float | None = None  # +1/-1 indicating overall direction

    # Volatility features (how much the line bounced around)
    prob_volatility: float | None = None  # Std dev of probabilities across snapshots
    prob_range: float | None = None  # Max - min probability
    odds_volatility: float | None = None  # Std dev of American odds
    movement_count: float | None = None  # Number of significant line changes

    # Trend features (shape of the trajectory)
    trend_slope: float | None = None  # Linear regression slope
    trend_strength: float | None = None  # Linear regression R²
    reversal_count: float | None = None  # Times direction changed
    acceleration: float | None = None  # Change in rate of change
    first_half_vs_recent: float | None = None  # First-half vs second-half comparison

    # Sharp money indicators
    sharp_prob_trajectory: float | None = None  # Total sharp prob change to decision
    sharp_retail_divergence_trend: float | None = None  # Change in sharp-retail gap
    sharp_leads_retail: float | None = None  # Did sharps move first? (binary)

    # Timing features (when did movement happen within available data)
    early_movement_pct: float | None = None  # % of total movement in first half
    recent_movement_pct: float | None = None  # % of total movement in recent half
    decision_tier_acceleration: float | None = None  # Movement rate in decision tier
    consistent_direction_pct: float | None = None  # % snapshots moving same direction

    # Sequence metadata
    num_snapshots: float | None = None  # Total snapshots in sequence
    sequence_duration_hours: float | None = None  # Time span of sequence

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array(
            [
                getattr(self, field.name) if getattr(self, field.name) is not None else np.nan
                for field in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return ordered list of feature names."""
        return [field.name for field in fields(cls)]


@dataclass
class HybridFeatures:
    """
    Combined tabular (single snapshot) and trajectory (full sequence) features.

    This container holds both feature types for models that benefit from
    both point-in-time snapshots and aggregate trajectory information.
    """

    tabular: TabularFeatures
    trajectory: TrajectoryFeatures

    def to_array(self) -> np.ndarray:
        """Concatenate tabular and trajectory features into single array."""
        return np.concatenate([self.tabular.to_array(), self.trajectory.to_array()])

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return combined feature names with prefixes for clarity."""
        tabular_names = [f"tab_{n}" for n in TabularFeatures.get_feature_names()]
        trajectory_names = [f"traj_{n}" for n in TrajectoryFeatures.get_feature_names()]
        return tabular_names + trajectory_names


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    Feature extractors transform raw odds data into model-ready features.
    Different model architectures require different feature extraction strategies:
    - Tabular models (XGBoost): Single snapshot → flat feature vector
    - Sequence models (LSTM): Time series → 3D tensor (batch, timesteps, features)
    - Transformers: Tokenized sequence → embeddings with attention masks

    Subclasses must implement:
    - extract_features(): Extract features from event and odds data
    - get_feature_names(): Return ordered list of feature names (for tabular models)
    """

    @abstractmethod
    def extract_features(
        self,
        event: BacktestEvent,
        odds_data: list[Odds] | list[list[Odds]],
        outcome: str | None = None,
        **kwargs,
    ) -> TabularFeatures | dict[str, Any]:
        """
        Extract features from event and odds data.

        Args:
            event: Event with final scores
            odds_data: Odds snapshot(s) - single list for tabular, list of lists for sequences
            outcome: Specific outcome to analyze (team name or "Over"/"Under")
            **kwargs: Additional extractor-specific parameters

        Returns:
            TabularFeatures instance for tabular extractors, or
            dict with "sequence" and "mask" keys for sequence extractors
        """

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return ordered list of feature names.

        For tabular models: List of scalar feature names
        For sequence models: May return feature names per timestep or None

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

    Extracts features from a single odds snapshot that capture:
    - Market consensus (average implied probabilities)
    - Sharp vs retail bookmaker discrepancies
    - Market efficiency (vig, hold percentages)
    - Best available odds (line shopping)
    - Team indicators (home/away)

    This extractor produces flat feature dictionaries suitable for
    gradient boosting and random forest models.

    Example:
        ```python
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            event=event,
            odds_data=odds_snapshot,  # Single snapshot at decision time
            outcome="Los Angeles Lakers"
        )
        # Returns: {"consensus_prob": 0.55, "sharp_prob": 0.52, ...}
        ```
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
        event: BacktestEvent,
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
        feature_values = {}

        # Filter for target market
        market_odds = filter_odds_by_market_outcome(odds_data, market)

        # Always set required fields
        feature_values["is_home_team"] = 1.0 if outcome == event.home_team else 0.0
        feature_values["is_away_team"] = 1.0 if outcome == event.away_team else 0.0

        if not market_odds:
            return TabularFeatures(**feature_values)

        # Sharp and retail bookmaker odds
        sharp_odds = filter_odds_by_bookmakers(market_odds, self.sharp_bookmakers)
        retail_odds = filter_odds_by_bookmakers(market_odds, self.retail_bookmakers)

        # 1. Market consensus features
        if market == "h2h":
            home_odds_list = filter_odds_by_market_outcome(market_odds, market, event.home_team)
            away_odds_list = filter_odds_by_market_outcome(market_odds, market, event.away_team)

            if home_odds_list and away_odds_list:
                feature_values["avg_home_odds"] = calculate_avg_price(home_odds_list)
                feature_values["avg_away_odds"] = calculate_avg_price(away_odds_list)
                feature_values["std_home_odds"] = float(np.std([o.price for o in home_odds_list]))
                feature_values["std_away_odds"] = float(np.std([o.price for o in away_odds_list]))

                # Market consensus (average implied probability)
                avg_home_prob = float(
                    np.mean([calculate_implied_probability(o.price) for o in home_odds_list])
                )
                avg_away_prob = float(
                    np.mean([calculate_implied_probability(o.price) for o in away_odds_list])
                )
                feature_values["home_consensus_prob"] = avg_home_prob
                feature_values["away_consensus_prob"] = avg_away_prob

                # Determine if analyzing home or away
                if outcome == event.home_team:
                    feature_values["consensus_prob"] = avg_home_prob
                    feature_values["opponent_consensus_prob"] = avg_away_prob
                elif outcome == event.away_team:
                    feature_values["consensus_prob"] = avg_away_prob
                    feature_values["opponent_consensus_prob"] = avg_home_prob

        # 2. Sharp vs Retail features (key for detecting value)
        if sharp_odds and retail_odds:
            sharp_home = next((o for o in sharp_odds if o.outcome_name == event.home_team), None)
            sharp_away = next((o for o in sharp_odds if o.outcome_name == event.away_team), None)

            if sharp_home and sharp_away:
                sharp_home_prob = calculate_implied_probability(sharp_home.price)
                sharp_away_prob = calculate_implied_probability(sharp_away.price)

                feature_values["sharp_home_prob"] = sharp_home_prob
                feature_values["sharp_away_prob"] = sharp_away_prob

                # Calculate sharp market hold (should be lower than retail)
                sharp_hold = calculate_market_hold([sharp_home.price, sharp_away.price])
                feature_values["sharp_market_hold"] = sharp_hold

                # Compare retail to sharp (deviation indicates potential value)
                retail_home_odds = filter_odds_by_market_outcome(
                    retail_odds, market, event.home_team
                )
                retail_away_odds = filter_odds_by_market_outcome(
                    retail_odds, market, event.away_team
                )

                if retail_home_odds:
                    avg_retail_home_prob = float(
                        np.mean([calculate_implied_probability(o.price) for o in retail_home_odds])
                    )
                    feature_values["retail_sharp_diff_home"] = (
                        avg_retail_home_prob - sharp_home_prob
                    )

                if retail_away_odds:
                    avg_retail_away_prob = float(
                        np.mean([calculate_implied_probability(o.price) for o in retail_away_odds])
                    )
                    feature_values["retail_sharp_diff_away"] = (
                        avg_retail_away_prob - sharp_away_prob
                    )

                # Set outcome-specific features
                if outcome == event.home_team:
                    feature_values["sharp_prob"] = sharp_home_prob
                    feature_values["opponent_sharp_prob"] = sharp_away_prob
                elif outcome == event.away_team:
                    feature_values["sharp_prob"] = sharp_away_prob
                    feature_values["opponent_sharp_prob"] = sharp_home_prob

        # 3. Market efficiency features
        all_books = {o.bookmaker_key for o in market_odds}
        feature_values["num_bookmakers"] = float(len(all_books))

        # Calculate average market hold across all books
        if market == "h2h":
            holds = []
            for book in all_books:
                book_odds = filter_odds_by_bookmakers(market_odds, [book])
                book_home = next((o for o in book_odds if o.outcome_name == event.home_team), None)
                book_away = next((o for o in book_odds if o.outcome_name == event.away_team), None)

                if book_home and book_away:
                    hold = calculate_market_hold([book_home.price, book_away.price])
                    holds.append(hold)

            if holds:
                feature_values["avg_market_hold"] = float(np.mean(holds))
                feature_values["std_market_hold"] = float(np.std(holds))

        # 4. Best available odds features (line shopping)
        if market == "h2h":
            home_market_odds = filter_odds_by_market_outcome(market_odds, market, event.home_team)
            away_market_odds = filter_odds_by_market_outcome(market_odds, market, event.away_team)
            home_prices = [o.price for o in home_market_odds]
            away_prices = [o.price for o in away_market_odds]

            if home_prices:
                feature_values["best_home_odds"] = float(max(home_prices))
                feature_values["worst_home_odds"] = float(min(home_prices))
                feature_values["home_odds_range"] = float(max(home_prices) - min(home_prices))

            if away_prices:
                feature_values["best_away_odds"] = float(max(away_prices))
                feature_values["worst_away_odds"] = float(min(away_prices))
                feature_values["away_odds_range"] = float(max(away_prices) - min(away_prices))

            # Set outcome-specific best odds
            if outcome == event.home_team and home_prices:
                feature_values["best_available_odds"] = float(max(home_prices))
                feature_values["odds_range"] = float(max(home_prices) - min(home_prices))
            elif outcome == event.away_team and away_prices:
                feature_values["best_available_odds"] = float(max(away_prices))
                feature_values["odds_range"] = float(max(away_prices) - min(away_prices))

        # 5. Decimal odds features (for model friendliness)
        if "best_available_odds" in feature_values:
            feature_values["best_available_decimal"] = american_to_decimal(
                int(feature_values["best_available_odds"])
            )

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


class SequenceFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for sequence models (LSTM, Transformers).

    Extracts time-series features from historical odds sequences, capturing
    line movement patterns and temporal dynamics that are critical for
    sequence models.

    Features extracted per timestep:
    - Odds values (American, decimal, implied probability)
    - Line movement (change from previous timestep, change from opening)
    - Market indicators (number of bookmakers, spread of odds)
    - Sharp vs retail differentials
    - Time encoding (hours until game start)

    The extractor resamples irregular snapshots to fixed timesteps and
    creates attention masks for variable-length sequences.

    Example:
        ```python
        from odds_lambda.storage.readers import OddsReader

        extractor = SequenceFeatureExtractor(
            lookback_hours=72,
            timesteps=24,  # One snapshot every 3 hours
        )

        # Query historical odds using OddsReader
        async with get_async_session() as session:
            reader = OddsReader(session)
            odds_history = await reader.get_line_movement(
                event_id="abc123",
                bookmaker_key="pinnacle",
                market_key="h2h",
                outcome_name="Lakers"
            )

        # Convert to list of snapshots (group by timestamp)
        snapshots = _group_odds_by_timestamp(odds_history)

        features = extractor.extract_features(
            event=event,
            odds_data=snapshots,  # List[List[Odds]]
            outcome="Lakers",
            market="h2h"
        )
        # Returns: {"sequence": np.ndarray(24, num_features), "mask": np.ndarray(24,)}
        ```

    Note:
        Uses uniform timestep allocation (nearest-neighbor resampling). For use cases
        with non-uniform data density, consider implementing a custom resampling strategy.
    """

    def __init__(
        self,
        lookback_hours: int = 72,
        timesteps: int = 24,
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
    ):
        """
        Initialize sequence feature extractor.

        Args:
            lookback_hours: Hours before game to start sequence (default: 72)
            timesteps: Number of timesteps in sequence (default: 24)
            sharp_bookmakers: Sharp bookmakers for line movement analysis (default: DEFAULT_SHARP_BOOKMAKERS)
            retail_bookmakers: Retail bookmakers for comparison (default: DEFAULT_RETAIL_BOOKMAKERS)
        """
        self.lookback_hours = lookback_hours
        self.timesteps = timesteps
        self.sharp_bookmakers = sharp_bookmakers or DEFAULT_SHARP_BOOKMAKERS
        self.retail_bookmakers = retail_bookmakers or DEFAULT_RETAIL_BOOKMAKERS

    @classmethod
    def from_config(cls, config: FeatureConfig) -> SequenceFeatureExtractor:
        """
        Create a SequenceFeatureExtractor from a FeatureConfig object.

        This factory method enables configuration-driven extractor instantiation,
        mapping all relevant FeatureConfig fields to extractor parameters.

        Args:
            config: FeatureConfig object with extraction parameters

        Returns:
            Configured SequenceFeatureExtractor instance

        Example:
            ```python
            from odds_analytics.training.config import FeatureConfig

            config = FeatureConfig(
                lookback_hours=48,
                timesteps=16,
                sharp_bookmakers=["pinnacle"],
                retail_bookmakers=["fanduel", "draftkings"],
            )
            extractor = SequenceFeatureExtractor.from_config(config)
            ```
        """
        return cls(
            lookback_hours=config.lookback_hours,
            timesteps=config.timesteps,
            sharp_bookmakers=config.sharp_bookmakers,
            retail_bookmakers=config.retail_bookmakers,
        )

    def extract_features(
        self,
        event: BacktestEvent,
        odds_data: list[list[Odds]],  # List of snapshots
        outcome: str | None = None,
        market: str = "h2h",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Extract time-series features from historical odds sequence.

        Args:
            event: Event with final scores
            odds_data: List of odds snapshots ordered by time (each snapshot is a list of Odds)
            outcome: Specific outcome to analyze (team name or Over/Under)
            market: Market to analyze (h2h, spreads, totals)

        Returns:
            Dictionary with:
            - "sequence": np.ndarray of shape (timesteps, num_features)
            - "mask": np.ndarray of shape (timesteps,) with True for valid data points

        Example:
            >>> extractor = SequenceFeatureExtractor(lookback_hours=48, timesteps=16)
            >>> result = extractor.extract_features(event, odds_snapshots, outcome="Lakers")
            >>> sequence = result["sequence"]  # Shape: (16, num_features)
            >>> mask = result["mask"]  # Shape: (16,) - True where data exists
        """
        # Initialize empty sequence
        feature_dim = len(SequenceFeatures.get_feature_names())
        sequence = np.zeros((self.timesteps, feature_dim))
        mask = np.zeros(self.timesteps, dtype=bool)

        # If no data, return empty sequence
        if not odds_data or all(len(snapshot) == 0 for snapshot in odds_data):
            return {"sequence": sequence, "mask": mask}

        # Extract timestamps from snapshots
        snapshot_times = []
        valid_snapshots = []

        for snapshot in odds_data:
            # Filter for target market and outcome
            filtered = filter_odds_by_market_outcome(snapshot, market, outcome)

            if filtered:
                valid_snapshots.append(snapshot)  # Keep full snapshot for market context
                snapshot_times.append(filtered[0].odds_timestamp)

        if not valid_snapshots:
            return {"sequence": sequence, "mask": mask}

        # Resample to fixed timesteps
        resampled_indices = self._resample_to_timesteps(snapshot_times, event.commence_time)

        # Track opening line for change calculation
        opening_odds = None

        # Extract features for each timestep
        for timestep_idx, snapshot_idx in enumerate(resampled_indices):
            if snapshot_idx is None:
                # No data for this timestep - leave as zeros with mask=False
                continue

            snapshot = valid_snapshots[snapshot_idx]
            snapshot_time = snapshot_times[snapshot_idx]

            # Extract features for this timestep
            timestep_features = self._extract_timestep_features(
                snapshot=snapshot,
                event=event,
                outcome=outcome,
                market=market,
                snapshot_time=snapshot_time,
                prev_snapshot=valid_snapshots[snapshot_idx - 1] if snapshot_idx > 0 else None,
                prev_time=snapshot_times[snapshot_idx - 1] if snapshot_idx > 0 else None,
            )

            # Store opening odds for reference and update opening-relative features
            if opening_odds is None and timestep_features is not None:
                opening_odds = timestep_features.american_odds
                # Create updated features with opening changes set to 0
                timestep_features = SequenceFeatures(
                    american_odds=timestep_features.american_odds,
                    decimal_odds=timestep_features.decimal_odds,
                    implied_prob=timestep_features.implied_prob,
                    num_bookmakers=timestep_features.num_bookmakers,
                    hours_to_game=timestep_features.hours_to_game,
                    time_of_day_sin=timestep_features.time_of_day_sin,
                    time_of_day_cos=timestep_features.time_of_day_cos,
                    odds_change_from_prev=timestep_features.odds_change_from_prev,
                    odds_change_from_opening=0.0,
                    implied_prob_change_from_prev=timestep_features.implied_prob_change_from_prev,
                    implied_prob_change_from_opening=0.0,
                    odds_std=timestep_features.odds_std,
                    sharp_odds=timestep_features.sharp_odds,
                    sharp_prob=timestep_features.sharp_prob,
                    retail_sharp_diff=timestep_features.retail_sharp_diff,
                )
            elif opening_odds is not None and timestep_features is not None:
                odds_change_from_opening = timestep_features.american_odds - opening_odds
                opening_prob = calculate_implied_probability(int(opening_odds))
                implied_prob_change_from_opening = timestep_features.implied_prob - opening_prob
                # Create updated features with opening changes calculated
                timestep_features = SequenceFeatures(
                    american_odds=timestep_features.american_odds,
                    decimal_odds=timestep_features.decimal_odds,
                    implied_prob=timestep_features.implied_prob,
                    num_bookmakers=timestep_features.num_bookmakers,
                    hours_to_game=timestep_features.hours_to_game,
                    time_of_day_sin=timestep_features.time_of_day_sin,
                    time_of_day_cos=timestep_features.time_of_day_cos,
                    odds_change_from_prev=timestep_features.odds_change_from_prev,
                    odds_change_from_opening=odds_change_from_opening,
                    implied_prob_change_from_prev=timestep_features.implied_prob_change_from_prev,
                    implied_prob_change_from_opening=implied_prob_change_from_opening,
                    odds_std=timestep_features.odds_std,
                    sharp_odds=timestep_features.sharp_odds,
                    sharp_prob=timestep_features.sharp_prob,
                    retail_sharp_diff=timestep_features.retail_sharp_diff,
                )

            # Convert to array
            if timestep_features is not None:
                sequence[timestep_idx] = timestep_features.to_array()
                mask[timestep_idx] = True

        return {"sequence": sequence, "mask": mask}

    def _resample_to_timesteps(
        self,
        snapshot_times: list,
        commence_time,
    ) -> list[int | None]:
        """
        Resample irregular snapshots to fixed timesteps.

        Uses nearest-neighbor resampling: each target timestep is mapped
        to the closest available snapshot (or None if too far).

        Args:
            snapshot_times: List of datetime objects for available snapshots
            commence_time: Game start time

        Returns:
            List of indices into snapshot_times (or None for missing data)
        """
        # Calculate target times working backwards from game start
        target_times = []
        for i in range(self.timesteps):
            hours_before = self.lookback_hours - (i * self.lookback_hours / self.timesteps)
            target_time = commence_time - timedelta(seconds=hours_before * 3600)
            target_times.append(target_time)

        # For each target time, find nearest snapshot
        resampled_indices = []
        tolerance_seconds = (self.lookback_hours / self.timesteps) * 3600 / 2  # Half interval

        for target_time in target_times:
            best_idx = None
            best_diff = float("inf")

            for idx, snap_time in enumerate(snapshot_times):
                diff_seconds = abs((snap_time - target_time).total_seconds())
                if diff_seconds < best_diff and diff_seconds <= tolerance_seconds:
                    best_diff = diff_seconds
                    best_idx = idx

            resampled_indices.append(best_idx)

        return resampled_indices

    def _extract_timestep_features(
        self,
        snapshot: list[Odds],
        event: BacktestEvent,
        outcome: str | None,
        market: str,
        snapshot_time,
        prev_snapshot: list[Odds] | None,
        prev_time,
    ) -> SequenceFeatures | None:
        """
        Extract features for a single timestep.

        Args:
            snapshot: Full odds snapshot at this timestep
            event: Event details
            outcome: Target outcome
            market: Market type
            snapshot_time: Timestamp of this snapshot
            prev_snapshot: Previous snapshot for change calculation
            prev_time: Timestamp of previous snapshot

        Returns:
            SequenceFeatures instance or None if no valid data
        """
        # Build feature values
        feature_values = {}

        # Filter for target market and outcome
        target_odds = filter_odds_by_market_outcome(snapshot, market, outcome)

        if not target_odds:
            return None

        # 1. Basic odds features (average across bookmakers)
        prices = [o.price for o in target_odds]
        avg_price = calculate_avg_price(target_odds)

        feature_values["american_odds"] = avg_price
        feature_values["decimal_odds"] = american_to_decimal(int(avg_price))
        feature_values["implied_prob"] = calculate_implied_probability(int(avg_price))

        # 2. Line movement features
        if prev_snapshot:
            prev_target_odds = filter_odds_by_market_outcome(prev_snapshot, market, outcome)

            if prev_target_odds:
                prev_avg_price = calculate_avg_price(prev_target_odds)
                feature_values["odds_change_from_prev"] = avg_price - prev_avg_price

                prev_prob = calculate_implied_probability(int(prev_avg_price))
                feature_values["implied_prob_change_from_prev"] = (
                    feature_values["implied_prob"] - prev_prob
                )
            else:
                feature_values["odds_change_from_prev"] = 0.0
                feature_values["implied_prob_change_from_prev"] = 0.0
        else:
            feature_values["odds_change_from_prev"] = 0.0
            feature_values["implied_prob_change_from_prev"] = 0.0

        # Placeholders for opening line changes (will be set by caller)
        # These will be overwritten by the caller based on opening odds
        feature_values["odds_change_from_opening"] = None
        feature_values["implied_prob_change_from_opening"] = None

        # 3. Market features
        bookmakers = {o.bookmaker_key for o in target_odds}
        feature_values["num_bookmakers"] = float(len(bookmakers))
        feature_values["odds_std"] = float(np.std(prices)) if len(prices) > 1 else None

        # 4. Sharp vs retail features
        sharp_retail = calculate_sharp_retail_metrics(
            target_odds, self.sharp_bookmakers, self.retail_bookmakers
        )
        feature_values["sharp_odds"] = sharp_retail["sharp_price"]
        feature_values["sharp_prob"] = sharp_retail["sharp_prob"]
        feature_values["retail_sharp_diff"] = sharp_retail["diff"]

        # 5. Time features (required)
        time_delta = event.commence_time - snapshot_time
        hours_to_game = time_delta.total_seconds() / 3600
        feature_values["hours_to_game"] = float(hours_to_game)

        # Cyclical time encoding (hour of day)
        hour_of_day = snapshot_time.hour + snapshot_time.minute / 60
        feature_values["time_of_day_sin"] = float(np.sin(2 * np.pi * hour_of_day / 24))
        feature_values["time_of_day_cos"] = float(np.cos(2 * np.pi * hour_of_day / 24))

        return SequenceFeatures(**feature_values)

    def get_feature_names(self) -> list[str]:
        """
        Return feature names per timestep.

        These features are extracted at each timestep in the sequence.
        The full sequence output has shape (timesteps, num_features).

        Returns:
            List of feature names

        Example:
            >>> extractor = SequenceFeatureExtractor()
            >>> names = extractor.get_feature_names()
            >>> "american_odds" in names
            True
        """
        return SequenceFeatures.get_feature_names()


class TrajectoryFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that computes aggregate features from full odds trajectory.

    Instead of only using opening/closing snapshots, this extractor analyzes
    all intermediate snapshots to capture momentum, volatility, trend patterns,
    and timing features that reveal how the market moved to its closing position.

    Example:
        ```python
        extractor = TrajectoryFeatureExtractor()
        features = extractor.extract_features(
            event=event,
            odds_data=all_snapshots,  # List[List[Odds]] - full sequence
            outcome="Los Angeles Lakers",
            market="h2h"
        )
        # Returns TrajectoryFeatures with momentum, volatility, trend features
        ```
    """

    def __init__(
        self,
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
        movement_threshold: float = 0.005,
    ):
        """
        Initialize trajectory feature extractor.

        Args:
            sharp_bookmakers: Sharp bookmakers for tracking (default: DEFAULT_SHARP_BOOKMAKERS)
            retail_bookmakers: Retail bookmakers for comparison (default: DEFAULT_RETAIL_BOOKMAKERS)
            movement_threshold: Probability change threshold for counting significant movements (default: 0.5%)
        """
        self.sharp_bookmakers = sharp_bookmakers or DEFAULT_SHARP_BOOKMAKERS
        self.retail_bookmakers = retail_bookmakers or DEFAULT_RETAIL_BOOKMAKERS
        self.movement_threshold = movement_threshold

    @classmethod
    def from_config(cls, config: FeatureConfig) -> TrajectoryFeatureExtractor:
        """Create TrajectoryFeatureExtractor from a FeatureConfig object."""
        movement_threshold = getattr(config, "movement_threshold", 0.005)
        return cls(
            sharp_bookmakers=config.sharp_bookmakers,
            retail_bookmakers=config.retail_bookmakers,
            movement_threshold=movement_threshold,
        )

    def extract_features(
        self,
        event: BacktestEvent,
        odds_data: list[list[Odds]],
        outcome: str | None = None,
        market: str = "h2h",
        **kwargs,
    ) -> TrajectoryFeatures:
        """
        Extract aggregate trajectory features from full odds sequence.

        Args:
            event: Event with final scores
            odds_data: List of odds snapshots ordered by time (each snapshot is List[Odds])
            outcome: Specific outcome to analyze (team name or Over/Under)
            market: Market to analyze (h2h, spreads, totals)

        Returns:
            TrajectoryFeatures instance with aggregate trajectory features
        """
        # Filter and extract probability time series
        prob_series = []
        odds_series = []
        timestamps = []
        sharp_probs = []
        retail_probs = []
        sharp_retail_diffs = []

        for snapshot in odds_data:
            target_odds = filter_odds_by_market_outcome(snapshot, market, outcome)

            if not target_odds:
                continue

            # Get average probability across bookmakers
            avg_price = calculate_avg_price(target_odds)
            avg_prob = calculate_avg_probability(target_odds)

            prob_series.append(avg_prob)
            odds_series.append(avg_price)
            timestamps.append(target_odds[0].odds_timestamp)

            # Track sharp vs retail
            sharp_retail = calculate_sharp_retail_metrics(
                target_odds, self.sharp_bookmakers, self.retail_bookmakers
            )
            if sharp_retail["sharp_prob"] is not None:
                sharp_probs.append(sharp_retail["sharp_prob"])
                if sharp_retail["retail_prob"] is not None:
                    retail_probs.append(sharp_retail["retail_prob"])
                if sharp_retail["diff"] is not None:
                    sharp_retail_diffs.append(sharp_retail["diff"])

        # Need at least 2 snapshots for trajectory features
        if len(prob_series) < 2:
            return TrajectoryFeatures(
                num_snapshots=float(len(prob_series)) if prob_series else None
            )

        # Compute all feature groups
        momentum = self._compute_momentum_features(prob_series, timestamps)
        volatility = self._compute_volatility_features(prob_series, odds_series)
        trend = self._compute_trend_features(prob_series, timestamps)
        sharp = self._compute_sharp_features(sharp_probs, retail_probs, sharp_retail_diffs)
        timing = self._compute_timing_features(prob_series, timestamps, event.commence_time)

        # Sequence metadata
        duration_hours = None
        if timestamps:
            duration = timestamps[-1] - timestamps[0]
            duration_hours = duration.total_seconds() / 3600

        return TrajectoryFeatures(
            # Momentum (to decision point)
            prob_change_to_decision=momentum.get("prob_change_to_decision"),
            avg_change_rate_to_decision=momentum.get("avg_change_rate_to_decision"),
            max_prob_increase=momentum.get("max_prob_increase"),
            max_prob_decrease=momentum.get("max_prob_decrease"),
            net_movement_direction=momentum.get("net_movement_direction"),
            # Volatility
            prob_volatility=volatility.get("prob_volatility"),
            prob_range=volatility.get("prob_range"),
            odds_volatility=volatility.get("odds_volatility"),
            movement_count=volatility.get("movement_count"),
            # Trend
            trend_slope=trend.get("trend_slope"),
            trend_strength=trend.get("trend_strength"),
            reversal_count=trend.get("reversal_count"),
            acceleration=trend.get("acceleration"),
            first_half_vs_recent=trend.get("first_half_vs_recent"),
            # Sharp
            sharp_prob_trajectory=sharp.get("sharp_prob_trajectory"),
            sharp_retail_divergence_trend=sharp.get("sharp_retail_divergence_trend"),
            sharp_leads_retail=sharp.get("sharp_leads_retail"),
            # Timing (within available data up to decision point)
            early_movement_pct=timing.get("early_movement_pct"),
            recent_movement_pct=timing.get("recent_movement_pct"),
            decision_tier_acceleration=timing.get("decision_tier_acceleration"),
            consistent_direction_pct=timing.get("consistent_direction_pct"),
            # Metadata
            num_snapshots=float(len(prob_series)),
            sequence_duration_hours=duration_hours,
        )

    def _compute_momentum_features(
        self, prob_series: list[float], timestamps: list
    ) -> dict[str, float]:
        """Compute momentum features from probability series (up to decision point)."""
        features = {}

        if len(prob_series) < 2:
            return features

        # Total change (opening to decision point, NOT to closing)
        total_change = prob_series[-1] - prob_series[0]
        features["prob_change_to_decision"] = total_change

        # Direction
        features["net_movement_direction"] = (
            1.0 if total_change > 0 else -1.0 if total_change < 0 else 0.0
        )

        # Rate of change (per hour)
        if timestamps:
            duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if duration > 0:
                features["avg_change_rate_to_decision"] = total_change / duration

        # Max increases/decreases
        changes = [prob_series[i + 1] - prob_series[i] for i in range(len(prob_series) - 1)]
        if changes:
            features["max_prob_increase"] = max(changes)
            features["max_prob_decrease"] = min(changes)

        return features

    def _compute_volatility_features(
        self, prob_series: list[float], odds_series: list[float]
    ) -> dict[str, float]:
        """Compute volatility features from probability and odds series."""
        features = {}

        if len(prob_series) < 2:
            return features

        # Probability volatility
        features["prob_volatility"] = float(np.std(prob_series))
        features["prob_range"] = float(max(prob_series) - min(prob_series))

        # Odds volatility
        features["odds_volatility"] = float(np.std(odds_series))

        # Count significant movements
        changes = [prob_series[i + 1] - prob_series[i] for i in range(len(prob_series) - 1)]
        significant_moves = sum(1 for c in changes if abs(c) >= self.movement_threshold)
        features["movement_count"] = float(significant_moves)

        return features

    def _compute_trend_features(
        self, prob_series: list[float], timestamps: list
    ) -> dict[str, float]:
        """Compute trend features using linear regression and pattern analysis."""
        features = {}

        if len(prob_series) < 2:
            return features

        # Linear regression for trend
        x = np.arange(len(prob_series))
        y = np.array(prob_series)

        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator > 0:
            slope = numerator / denominator
            features["trend_slope"] = float(slope)

            # R-squared
            y_pred = slope * (x - x_mean) + y_mean
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            if ss_tot > 0:
                features["trend_strength"] = float(1 - ss_res / ss_tot)

        # Reversal count
        changes = [prob_series[i + 1] - prob_series[i] for i in range(len(prob_series) - 1)]
        reversals = sum(
            1
            for i in range(1, len(changes))
            if (changes[i] > 0 and changes[i - 1] < 0) or (changes[i] < 0 and changes[i - 1] > 0)
        )
        features["reversal_count"] = float(reversals)

        # Acceleration (change in rate of change)
        if len(changes) >= 2:
            first_half_avg = np.mean(changes[: len(changes) // 2])
            second_half_avg = np.mean(changes[len(changes) // 2 :])
            features["acceleration"] = float(second_half_avg - first_half_avg)

        # First half vs recent (second half) momentum
        mid_idx = len(prob_series) // 2
        if mid_idx > 0:
            early_change = prob_series[mid_idx] - prob_series[0]
            recent_change = prob_series[-1] - prob_series[mid_idx]
            features["first_half_vs_recent"] = float(recent_change - early_change)

        return features

    def _compute_sharp_features(
        self,
        sharp_probs: list[float],
        retail_probs: list[float],
        sharp_retail_diffs: list[float],
    ) -> dict[str, float]:
        """Compute sharp money indicator features."""
        features = {}

        # Total sharp movement
        if len(sharp_probs) >= 2:
            features["sharp_prob_trajectory"] = float(sharp_probs[-1] - sharp_probs[0])

        # Sharp-retail divergence trend
        if len(sharp_retail_diffs) >= 2:
            features["sharp_retail_divergence_trend"] = float(
                sharp_retail_diffs[-1] - sharp_retail_diffs[0]
            )

        # Did sharps move before retail?
        if len(sharp_probs) >= 2 and len(retail_probs) >= 2:
            sharp_first_move = abs(sharp_probs[1] - sharp_probs[0])
            retail_first_move = abs(retail_probs[1] - retail_probs[0])
            features["sharp_leads_retail"] = 1.0 if sharp_first_move > retail_first_move else 0.0

        return features

    def _compute_timing_features(
        self,
        prob_series: list[float],
        timestamps: list,
        commence_time,
    ) -> dict[str, float]:
        """
        Compute timing-based features.

        Note: These features are computed from the sequence as passed in.
        The sequence should already be filtered to only include snapshots
        up to the decision tier to avoid look-ahead bias.
        """
        features = {}

        if len(prob_series) < 2:
            return features

        total_change = abs(prob_series[-1] - prob_series[0])

        # Split into early and recent halves
        mid_idx = len(prob_series) // 2
        if mid_idx > 0 and total_change > 0:
            early_change = abs(prob_series[mid_idx] - prob_series[0])
            recent_change = abs(prob_series[-1] - prob_series[mid_idx])

            features["early_movement_pct"] = float(early_change / total_change)
            features["recent_movement_pct"] = float(recent_change / total_change)

        # Decision tier acceleration (movement rate in final portion of available data)
        # Uses final 25% of snapshots (not fixed hours, since sequence is already filtered)
        if len(prob_series) >= 4:
            final_quarter_start = int(len(prob_series) * 0.75)
            final_probs = prob_series[final_quarter_start:]
            final_times = timestamps[final_quarter_start:]

            if len(final_probs) >= 2:
                final_change = final_probs[-1] - final_probs[0]
                final_duration = (final_times[-1] - final_times[0]).total_seconds() / 3600
                if final_duration > 0:
                    features["decision_tier_acceleration"] = float(final_change / final_duration)

        # Consistent direction percentage
        changes = [prob_series[i + 1] - prob_series[i] for i in range(len(prob_series) - 1)]
        if changes:
            positive_moves = sum(1 for c in changes if c > 0)
            negative_moves = sum(1 for c in changes if c < 0)
            dominant_direction = max(positive_moves, negative_moves)
            features["consistent_direction_pct"] = float(dominant_direction / len(changes))

        return features

    def get_feature_names(self) -> list[str]:
        """Return ordered list of trajectory feature names."""
        return TrajectoryFeatures.get_feature_names()


class HybridFeatureExtractor(FeatureExtractor):
    """
    Feature extractor combining tabular (single snapshot) and trajectory (full sequence) features.

    This extractor is designed for XGBoost and similar models that can benefit
    from both point-in-time features and aggregate trajectory information.

    Example:
        ```python
        extractor = HybridFeatureExtractor()
        features = extractor.extract_features(
            event=event,
            odds_data=all_snapshots,  # Full sequence
            outcome="Los Angeles Lakers",
            market="h2h",
            opening_snapshot_idx=0  # Which snapshot for tabular features
        )
        # Returns HybridFeatures with both tabular and trajectory features
        ```
    """

    def __init__(
        self,
        tabular_extractor: TabularFeatureExtractor | None = None,
        trajectory_extractor: TrajectoryFeatureExtractor | None = None,
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
        movement_threshold: float = 0.005,
    ):
        """
        Initialize hybrid feature extractor.

        Args:
            tabular_extractor: Pre-configured TabularFeatureExtractor (optional)
            trajectory_extractor: Pre-configured TrajectoryFeatureExtractor (optional)
            sharp_bookmakers: Sharp bookmakers (used if extractors not provided)
            retail_bookmakers: Retail bookmakers (used if extractors not provided)
            movement_threshold: Threshold for significant movement detection
        """
        sharp = sharp_bookmakers or ["pinnacle"]
        retail = retail_bookmakers or ["fanduel", "draftkings", "betmgm"]

        self.tabular_extractor = tabular_extractor or TabularFeatureExtractor(
            sharp_bookmakers=sharp,
            retail_bookmakers=retail,
        )
        self.trajectory_extractor = trajectory_extractor or TrajectoryFeatureExtractor(
            sharp_bookmakers=sharp,
            retail_bookmakers=retail,
            movement_threshold=movement_threshold,
        )

    @classmethod
    def from_config(cls, config: FeatureConfig) -> HybridFeatureExtractor:
        """Create HybridFeatureExtractor from a FeatureConfig object."""
        return cls(
            tabular_extractor=TabularFeatureExtractor.from_config(config),
            trajectory_extractor=TrajectoryFeatureExtractor.from_config(config),
        )

    def extract_features(
        self,
        event: BacktestEvent,
        odds_data: list[list[Odds]],
        outcome: str | None = None,
        market: str = "h2h",
        opening_snapshot_idx: int = 0,
        **kwargs,
    ) -> HybridFeatures:
        """
        Extract combined tabular and trajectory features.

        Args:
            event: Event with final scores
            odds_data: List of odds snapshots ordered by time (each snapshot is List[Odds])
            outcome: Specific outcome to analyze
            market: Market to analyze
            opening_snapshot_idx: Which snapshot to use for tabular features (default: 0 = first)

        Returns:
            HybridFeatures instance with both tabular and trajectory features
        """
        # Extract tabular features from specified snapshot
        if odds_data and opening_snapshot_idx < len(odds_data):
            tabular_snapshot = odds_data[opening_snapshot_idx]
            tabular_features = self.tabular_extractor.extract_features(
                event=event,
                odds_data=tabular_snapshot,
                outcome=outcome,
                market=market,
            )
        else:
            # Empty tabular features if no data
            tabular_features = TabularFeatures(
                is_home_team=1.0 if outcome == event.home_team else 0.0,
                is_away_team=1.0 if outcome == event.away_team else 0.0,
            )

        # Extract trajectory features from full sequence
        trajectory_features = self.trajectory_extractor.extract_features(
            event=event,
            odds_data=odds_data,
            outcome=outcome,
            market=market,
        )

        return HybridFeatures(tabular=tabular_features, trajectory=trajectory_features)

    def get_feature_names(self) -> list[str]:
        """Return combined feature names with prefixes."""
        return HybridFeatures.get_feature_names()
