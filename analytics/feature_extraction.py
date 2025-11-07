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
from typing import Any

import numpy as np

from analytics.backtesting import BacktestEvent
from analytics.utils import american_to_decimal, calculate_implied_probability, calculate_market_hold
from core.models import Odds

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

    def create_feature_vector(self, features: dict[str, float], feature_names: list[str] | None = None) -> np.ndarray:
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
                avg_home_prob = float(np.mean(
                    [calculate_implied_probability(o.price) for o in home_odds_list]
                ))
                avg_away_prob = float(np.mean(
                    [calculate_implied_probability(o.price) for o in away_odds_list]
                ))
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
            sharp_home = next(
                (o for o in sharp_odds if o.outcome_name == event.home_team), None
            )
            sharp_away = next(
                (o for o in sharp_odds if o.outcome_name == event.away_team), None
            )

            if sharp_home and sharp_away:
                sharp_home_prob = calculate_implied_probability(sharp_home.price)
                sharp_away_prob = calculate_implied_probability(sharp_away.price)

                features["sharp_home_prob"] = sharp_home_prob
                features["sharp_away_prob"] = sharp_away_prob

                # Calculate sharp market hold (should be lower than retail)
                sharp_hold = calculate_market_hold([sharp_home.price, sharp_away.price])
                features["sharp_market_hold"] = sharp_hold

                # Compare retail to sharp (deviation indicates potential value)
                retail_home_odds = [
                    o for o in retail_odds if o.outcome_name == event.home_team
                ]
                retail_away_odds = [
                    o for o in retail_odds if o.outcome_name == event.away_team
                ]

                if retail_home_odds:
                    avg_retail_home_prob = float(np.mean(
                        [calculate_implied_probability(o.price) for o in retail_home_odds]
                    ))
                    features["retail_sharp_diff_home"] = avg_retail_home_prob - sharp_home_prob

                if retail_away_odds:
                    avg_retail_away_prob = float(np.mean(
                        [calculate_implied_probability(o.price) for o in retail_away_odds]
                    ))
                    features["retail_sharp_diff_away"] = avg_retail_away_prob - sharp_away_prob

                # Set outcome-specific features
                if outcome == event.home_team:
                    features["sharp_prob"] = sharp_home_prob
                    features["opponent_sharp_prob"] = sharp_away_prob
                elif outcome == event.away_team:
                    features["sharp_prob"] = sharp_away_prob
                    features["opponent_sharp_prob"] = sharp_home_prob

        # 3. Market efficiency features
        all_books = set(o.bookmaker_key for o in market_odds)
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

    STUB - Future implementation for time-series models that need
    historical odds sequences rather than single snapshots.

    Planned features:
    - Line movement patterns over 72 hours
    - Time-to-game encoding
    - Sharp money indicators (reverse line movement)
    - Steam moves (rapid line changes)
    - Betting percentage vs line movement divergence

    Expected input: List of odds snapshots ordered by time
    Expected output: 3D tensor (timesteps, features) or dictionary with sequences

    Example (future):
        ```python
        extractor = SequenceFeatureExtractor(
            lookback_hours=72,
            timesteps=24,  # One snapshot every 3 hours
        )

        # odds_history: List of snapshots from 72h before game to decision time
        features = extractor.extract_features(
            event=event,
            odds_data=odds_history,  # List[List[Odds]]
            outcome="Lakers"
        )
        # Returns: {"sequence": np.ndarray(24, 50), "mask": np.ndarray(24,)}
        ```

    Note:
        This is a placeholder for future LSTM/Transformer implementations.
        The actual interface may evolve based on model requirements.
    """

    def __init__(
        self,
        lookback_hours: int = 72,
        timesteps: int = 24,
        sharp_bookmakers: list[str] | None = None,
    ):
        """
        Initialize sequence feature extractor.

        Args:
            lookback_hours: Hours before game to start sequence (default: 72)
            timesteps: Number of timesteps in sequence (default: 24)
            sharp_bookmakers: Sharp bookmakers for line movement analysis
        """
        self.lookback_hours = lookback_hours
        self.timesteps = timesteps
        self.sharp_bookmakers = sharp_bookmakers or ["pinnacle"]

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
            odds_data: List of odds snapshots ordered by time
            outcome: Specific outcome to analyze
            market: Market to analyze (h2h, spreads, totals)

        Returns:
            Dictionary with sequence data (e.g., {"sequence": array, "mask": array})

        Raises:
            NotImplementedError: This is a stub for future implementation
        """
        raise NotImplementedError(
            "SequenceFeatureExtractor is a stub for future LSTM/Transformer support. "
            "To implement:\n"
            "1. Query OddsReader.get_line_movement() to fetch historical snapshots\n"
            "2. Resample/interpolate to fixed timesteps (e.g., every 3 hours)\n"
            "3. Extract features at each timestep (odds, implied prob, line changes)\n"
            "4. Stack into 3D tensor (timesteps, features)\n"
            "5. Create attention mask for variable-length sequences\n"
            "6. Return dict with 'sequence' and 'mask' arrays"
        )

    def get_feature_names(self) -> list[str]:
        """
        Return feature names per timestep.

        For sequence models, this might return per-timestep feature names
        or None if features are learned embeddings.

        Returns:
            List of feature names or empty list for stub
        """
        # Future: Return per-timestep feature names
        # e.g., ["odds", "implied_prob", "line_change", "volume_indicator"]
        return []
