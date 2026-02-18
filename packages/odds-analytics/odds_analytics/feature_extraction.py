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
)

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

__all__ = [
    "FeatureExtractor",
    "TabularFeatureExtractor",
    "SequenceFeatureExtractor",
    "TabularFeatures",
    "SequenceFeatures",
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
        feature_values: dict[str, float] = {}

        # Filter for target market
        market_odds = filter_odds_by_market_outcome(odds_data, market)

        if not market_odds:
            return TabularFeatures(**feature_values)

        # Sharp and retail bookmaker odds
        sharp_odds = filter_odds_by_bookmakers(market_odds, self.sharp_bookmakers)
        retail_odds = filter_odds_by_bookmakers(market_odds, self.retail_bookmakers)

        # Resolve outcome team
        if outcome == event.home_team:
            outcome_team = event.home_team
        elif outcome == event.away_team:
            outcome_team = event.away_team
        else:
            outcome_team = None

        # 1. Consensus probability
        if market == "h2h" and outcome_team:
            outcome_odds_list = filter_odds_by_market_outcome(market_odds, market, outcome_team)
            if outcome_odds_list:
                avg_outcome_prob = float(
                    np.mean([calculate_implied_probability(o.price) for o in outcome_odds_list])
                )
                feature_values["consensus_prob"] = avg_outcome_prob

        # 2. Sharp vs Retail features
        if sharp_odds and retail_odds and outcome_team:
            sharp_outcome = next((o for o in sharp_odds if o.outcome_name == outcome_team), None)
            if sharp_outcome:
                outcome_sharp_prob = calculate_implied_probability(sharp_outcome.price)
                feature_values["sharp_prob"] = outcome_sharp_prob

                # Retail-sharp diff for outcome side only
                retail_outcome_odds = filter_odds_by_market_outcome(
                    retail_odds, market, outcome_team
                )
                if retail_outcome_odds:
                    avg_retail_prob = float(
                        np.mean(
                            [calculate_implied_probability(o.price) for o in retail_outcome_odds]
                        )
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
        async with async_session_maker() as session:
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
