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
from datetime import timedelta
from typing import Any

import numpy as np
from odds_core.models import Odds

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.utils import (
    american_to_decimal,
    calculate_implied_probability,
    calculate_market_hold,
)

__all__ = [
    "FeatureExtractor",
    "TabularFeatureExtractor",
    "SequenceFeatureExtractor",
]


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
    ) -> dict[str, Any]:
        """
        Extract features from event and odds data.

        Args:
            event: Event with final scores
            odds_data: Odds snapshot(s) - single list for tabular, list of lists for sequences
            outcome: Specific outcome to analyze (team name or "Over"/"Under")
            **kwargs: Additional extractor-specific parameters

        Returns:
            Feature dictionary (structure depends on extractor type)
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
        self, features: dict[str, float], feature_names: list[str] | None = None
    ) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for model input.

        Args:
            features: Feature dictionary from extract_features()
            feature_names: Ordered list of feature names (uses get_feature_names() if None)

        Returns:
            Numpy array of feature values (fills missing with 0.0)
        """
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
            sharp_bookmakers: Sharp bookmakers for "true" odds (default: ["pinnacle"])
            retail_bookmakers: Retail bookmakers for comparison (default: ["fanduel", "draftkings", "betmgm"])
        """
        self.sharp_bookmakers = sharp_bookmakers or ["pinnacle"]
        self.retail_bookmakers = retail_bookmakers or ["fanduel", "draftkings", "betmgm"]

        # Feature names are dynamically determined based on market
        # This is a comprehensive list of all possible features
        self._feature_names = [
            # Consensus features
            "avg_home_odds",
            "avg_away_odds",
            "std_home_odds",
            "std_away_odds",
            "home_consensus_prob",
            "away_consensus_prob",
            "consensus_prob",
            "opponent_consensus_prob",
            # Sharp bookmaker features
            "sharp_home_prob",
            "sharp_away_prob",
            "sharp_market_hold",
            "sharp_prob",
            "opponent_sharp_prob",
            # Retail vs sharp features
            "retail_sharp_diff_home",
            "retail_sharp_diff_away",
            # Market efficiency features
            "num_bookmakers",
            "avg_market_hold",
            "std_market_hold",
            # Best odds features (line shopping)
            "best_home_odds",
            "worst_home_odds",
            "home_odds_range",
            "best_away_odds",
            "worst_away_odds",
            "away_odds_range",
            "best_available_odds",
            "odds_range",
            "best_available_decimal",
            # Team indicators
            "is_home_team",
            "is_away_team",
        ]

    def extract_features(
        self,
        event: BacktestEvent,
        odds_data: list[Odds],
        outcome: str | None = None,
        market: str = "h2h",
        **kwargs,
    ) -> dict[str, float]:
        """
        Extract ML features from event and odds snapshot.

        Args:
            event: Event with final scores
            odds_data: Odds at decision time (single snapshot)
            outcome: Specific outcome to analyze (if None, analyzes both sides)
            market: Market to analyze (h2h, spreads, totals)

        Returns:
            Dictionary of feature names to values
        """
        features = {}
        odds_snapshot = odds_data  # Alias for clarity

        # Filter for target market
        market_odds = [o for o in odds_snapshot if o.market_key == market]

        if not market_odds:
            return features

        # Sharp bookmaker features
        sharp_odds = [o for o in market_odds if o.bookmaker_key in self.sharp_bookmakers]

        # Retail bookmaker features
        retail_odds = [o for o in market_odds if o.bookmaker_key in self.retail_bookmakers]

        # 1. Market consensus features
        if market == "h2h":
            home_odds_list = [o for o in market_odds if o.outcome_name == event.home_team]
            away_odds_list = [o for o in market_odds if o.outcome_name == event.away_team]

            if home_odds_list and away_odds_list:
                features["avg_home_odds"] = float(np.mean([o.price for o in home_odds_list]))
                features["avg_away_odds"] = float(np.mean([o.price for o in away_odds_list]))
                features["std_home_odds"] = float(np.std([o.price for o in home_odds_list]))
                features["std_away_odds"] = float(np.std([o.price for o in away_odds_list]))

                # Market consensus (average implied probability)
                avg_home_prob = float(
                    np.mean([calculate_implied_probability(o.price) for o in home_odds_list])
                )
                avg_away_prob = float(
                    np.mean([calculate_implied_probability(o.price) for o in away_odds_list])
                )
                features["home_consensus_prob"] = avg_home_prob
                features["away_consensus_prob"] = avg_away_prob

                # Determine if analyzing home or away
                if outcome == event.home_team:
                    features["consensus_prob"] = avg_home_prob
                    features["opponent_consensus_prob"] = avg_away_prob
                elif outcome == event.away_team:
                    features["consensus_prob"] = avg_away_prob
                    features["opponent_consensus_prob"] = avg_home_prob

        # 2. Sharp vs Retail features (key for detecting value)
        if sharp_odds and retail_odds:
            sharp_home = next((o for o in sharp_odds if o.outcome_name == event.home_team), None)
            sharp_away = next((o for o in sharp_odds if o.outcome_name == event.away_team), None)

            if sharp_home and sharp_away:
                sharp_home_prob = calculate_implied_probability(sharp_home.price)
                sharp_away_prob = calculate_implied_probability(sharp_away.price)

                features["sharp_home_prob"] = sharp_home_prob
                features["sharp_away_prob"] = sharp_away_prob

                # Calculate sharp market hold (should be lower than retail)
                sharp_hold = calculate_market_hold([sharp_home.price, sharp_away.price])
                features["sharp_market_hold"] = sharp_hold

                # Compare retail to sharp (deviation indicates potential value)
                retail_home_odds = [o for o in retail_odds if o.outcome_name == event.home_team]
                retail_away_odds = [o for o in retail_odds if o.outcome_name == event.away_team]

                if retail_home_odds:
                    avg_retail_home_prob = float(
                        np.mean([calculate_implied_probability(o.price) for o in retail_home_odds])
                    )
                    features["retail_sharp_diff_home"] = avg_retail_home_prob - sharp_home_prob

                if retail_away_odds:
                    avg_retail_away_prob = float(
                        np.mean([calculate_implied_probability(o.price) for o in retail_away_odds])
                    )
                    features["retail_sharp_diff_away"] = avg_retail_away_prob - sharp_away_prob

                # Set outcome-specific features
                if outcome == event.home_team:
                    features["sharp_prob"] = sharp_home_prob
                    features["opponent_sharp_prob"] = sharp_away_prob
                elif outcome == event.away_team:
                    features["sharp_prob"] = sharp_away_prob
                    features["opponent_sharp_prob"] = sharp_home_prob

        # 3. Market efficiency features
        all_books = {o.bookmaker_key for o in market_odds}
        features["num_bookmakers"] = float(len(all_books))

        # Calculate average market hold across all books
        if market == "h2h":
            holds = []
            for book in all_books:
                book_odds = [o for o in market_odds if o.bookmaker_key == book]
                book_home = next((o for o in book_odds if o.outcome_name == event.home_team), None)
                book_away = next((o for o in book_odds if o.outcome_name == event.away_team), None)

                if book_home and book_away:
                    hold = calculate_market_hold([book_home.price, book_away.price])
                    holds.append(hold)

            if holds:
                features["avg_market_hold"] = float(np.mean(holds))
                features["std_market_hold"] = float(np.std(holds))

        # 4. Best available odds features (line shopping)
        if market == "h2h":
            home_prices = [o.price for o in market_odds if o.outcome_name == event.home_team]
            away_prices = [o.price for o in market_odds if o.outcome_name == event.away_team]

            if home_prices:
                features["best_home_odds"] = float(max(home_prices))
                features["worst_home_odds"] = float(min(home_prices))
                features["home_odds_range"] = float(max(home_prices) - min(home_prices))

            if away_prices:
                features["best_away_odds"] = float(max(away_prices))
                features["worst_away_odds"] = float(min(away_prices))
                features["away_odds_range"] = float(max(away_prices) - min(away_prices))

            # Set outcome-specific best odds
            if outcome == event.home_team and home_prices:
                features["best_available_odds"] = float(max(home_prices))
                features["odds_range"] = float(max(home_prices) - min(home_prices))
            elif outcome == event.away_team and away_prices:
                features["best_available_odds"] = float(max(away_prices))
                features["odds_range"] = float(max(away_prices) - min(away_prices))

        # 5. Decimal odds features (for model friendliness)
        if "best_available_odds" in features:
            features["best_available_decimal"] = american_to_decimal(
                int(features["best_available_odds"])
            )

        # 6. Binary features
        features["is_home_team"] = 1.0 if outcome == event.home_team else 0.0
        features["is_away_team"] = 1.0 if outcome == event.away_team else 0.0

        return features

    def get_feature_names(self) -> list[str]:
        """
        Return ordered list of feature names.

        Note: Not all features may be present for every event (depends on
        available bookmakers and market type). Missing features are filled
        with 0.0 in create_feature_vector().

        Returns:
            Ordered list of all possible feature names
        """
        return self._feature_names


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
            sharp_bookmakers: Sharp bookmakers for line movement analysis (default: ["pinnacle"])
            retail_bookmakers: Retail bookmakers for comparison (default: ["fanduel", "draftkings", "betmgm"])
        """
        self.lookback_hours = lookback_hours
        self.timesteps = timesteps
        self.sharp_bookmakers = sharp_bookmakers or ["pinnacle"]
        self.retail_bookmakers = retail_bookmakers or ["fanduel", "draftkings", "betmgm"]

        # Feature names (per timestep)
        self._feature_names = [
            "american_odds",
            "decimal_odds",
            "implied_prob",
            "odds_change_from_prev",
            "odds_change_from_opening",
            "implied_prob_change_from_prev",
            "implied_prob_change_from_opening",
            "num_bookmakers",
            "odds_std",
            "sharp_odds",
            "sharp_prob",
            "retail_sharp_diff",
            "hours_to_game",
            "time_of_day_sin",
            "time_of_day_cos",
        ]

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
        feature_dim = len(self._feature_names)
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
            filtered = [o for o in snapshot if o.market_key == market]
            if outcome:
                filtered = [o for o in filtered if o.outcome_name == outcome]

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
            features_dict = self._extract_timestep_features(
                snapshot=snapshot,
                event=event,
                outcome=outcome,
                market=market,
                snapshot_time=snapshot_time,
                prev_snapshot=valid_snapshots[snapshot_idx - 1] if snapshot_idx > 0 else None,
                prev_time=snapshot_times[snapshot_idx - 1] if snapshot_idx > 0 else None,
            )

            # Store opening odds for reference
            if opening_odds is None and "american_odds" in features_dict:
                opening_odds = features_dict["american_odds"]
                features_dict["odds_change_from_opening"] = 0.0
                features_dict["implied_prob_change_from_opening"] = 0.0
            elif opening_odds is not None:
                features_dict["odds_change_from_opening"] = (
                    features_dict.get("american_odds", opening_odds) - opening_odds
                )
                opening_prob = calculate_implied_probability(int(opening_odds))
                current_prob = features_dict.get("implied_prob", opening_prob)
                features_dict["implied_prob_change_from_opening"] = current_prob - opening_prob

            # Convert to array
            sequence[timestep_idx] = self._features_dict_to_array(features_dict)
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
    ) -> dict[str, float]:
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
            Dictionary of feature names to values
        """
        features = {}

        # Filter for target market and outcome
        market_odds = [o for o in snapshot if o.market_key == market]
        if outcome:
            target_odds = [o for o in market_odds if o.outcome_name == outcome]
        else:
            target_odds = market_odds

        if not target_odds:
            return self._empty_features()

        # 1. Basic odds features (average across bookmakers)
        prices = [o.price for o in target_odds]
        avg_price = float(np.mean(prices))

        features["american_odds"] = avg_price
        features["decimal_odds"] = american_to_decimal(int(avg_price))
        features["implied_prob"] = calculate_implied_probability(int(avg_price))

        # 2. Line movement features
        if prev_snapshot:
            prev_market_odds = [o for o in prev_snapshot if o.market_key == market]
            if outcome:
                prev_target_odds = [o for o in prev_market_odds if o.outcome_name == outcome]
            else:
                prev_target_odds = prev_market_odds

            if prev_target_odds:
                prev_avg_price = float(np.mean([o.price for o in prev_target_odds]))
                features["odds_change_from_prev"] = avg_price - prev_avg_price

                prev_prob = calculate_implied_probability(int(prev_avg_price))
                features["implied_prob_change_from_prev"] = features["implied_prob"] - prev_prob
            else:
                features["odds_change_from_prev"] = 0.0
                features["implied_prob_change_from_prev"] = 0.0
        else:
            features["odds_change_from_prev"] = 0.0
            features["implied_prob_change_from_prev"] = 0.0

        # Placeholders for opening line changes (filled by caller)
        features["odds_change_from_opening"] = 0.0
        features["implied_prob_change_from_opening"] = 0.0

        # 3. Market features
        bookmakers = {o.bookmaker_key for o in target_odds}
        features["num_bookmakers"] = float(len(bookmakers))
        features["odds_std"] = float(np.std(prices)) if len(prices) > 1 else 0.0

        # 4. Sharp vs retail features
        sharp_odds_list = [o for o in target_odds if o.bookmaker_key in self.sharp_bookmakers]
        retail_odds_list = [o for o in target_odds if o.bookmaker_key in self.retail_bookmakers]

        if sharp_odds_list:
            sharp_price = float(np.mean([o.price for o in sharp_odds_list]))
            features["sharp_odds"] = sharp_price
            features["sharp_prob"] = calculate_implied_probability(int(sharp_price))

            if retail_odds_list:
                retail_price = float(np.mean([o.price for o in retail_odds_list]))
                retail_prob = calculate_implied_probability(int(retail_price))
                features["retail_sharp_diff"] = retail_prob - features["sharp_prob"]
            else:
                features["retail_sharp_diff"] = 0.0
        else:
            features["sharp_odds"] = avg_price
            features["sharp_prob"] = features["implied_prob"]
            features["retail_sharp_diff"] = 0.0

        # 5. Time features
        time_delta = event.commence_time - snapshot_time
        hours_to_game = time_delta.total_seconds() / 3600
        features["hours_to_game"] = float(hours_to_game)

        # Cyclical time encoding (hour of day)
        hour_of_day = snapshot_time.hour + snapshot_time.minute / 60
        features["time_of_day_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
        features["time_of_day_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

        return features

    def _empty_features(self) -> dict[str, float]:
        """Return dictionary of features with zero values."""
        return dict.fromkeys(self._feature_names, 0.0)

    def _features_dict_to_array(self, features_dict: dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to ordered numpy array."""
        return np.array([features_dict.get(name, 0.0) for name in self._feature_names])

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
        return self._feature_names
