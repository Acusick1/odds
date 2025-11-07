"""
XGBoost ML Strategy Example.

This module demonstrates how to build an ML-based betting strategy using XGBoost.
It serves as a reference implementation for creating machine learning strategies.

Key Features:
- Feature engineering from historical odds data (via pluggable FeatureExtractor)
- XGBoost binary classifier for win probability prediction
- Integration with backtesting framework via BetOpportunity.confidence
- Model persistence (save/load weights)
- Comprehensive feature set including line movement, market efficiency, consensus

Dependencies (install with uv):
    uv add xgboost scikit-learn numpy

Architecture:
- XGBoostStrategy uses dependency injection to accept any FeatureExtractor
- Default: TabularFeatureExtractor for snapshot-based features
- Extensible: Can swap in SequenceFeatureExtractor for LSTM models
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity, BettingStrategy
from analytics.feature_extraction import FeatureExtractor, TabularFeatureExtractor
from analytics.utils import american_to_decimal, calculate_implied_probability, calculate_market_hold
from core.models import Odds


class FeatureEngineering:
    """
    Feature engineering for sports betting ML models.

    DEPRECATED: Use TabularFeatureExtractor from analytics.feature_extraction instead.
    This class is maintained for backward compatibility and will be removed in a future version.

    Example migration:
        ```python
        # Old (deprecated)
        features = FeatureEngineering.extract_features(event, odds, outcome="Lakers")

        # New (recommended)
        from analytics.feature_extraction import TabularFeatureExtractor
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(event, odds, outcome="Lakers")
        ```
    """

    _warned = False  # Class-level flag to warn only once

    @staticmethod
    def extract_features(
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        market: str = "h2h",
        outcome: str | None = None,
    ) -> dict[str, float]:
        """
        Extract ML features from event and odds snapshot.

        DEPRECATED: Use TabularFeatureExtractor instead.

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time
            market: Market to analyze (h2h, spreads, totals)
            outcome: Specific outcome to analyze (if None, analyzes both sides)

        Returns:
            Dictionary of feature names to values
        """
        if not FeatureEngineering._warned:
            warnings.warn(
                "FeatureEngineering is deprecated. Use TabularFeatureExtractor from "
                "analytics.feature_extraction instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            FeatureEngineering._warned = True

        # Delegate to TabularFeatureExtractor for backward compatibility
        extractor = TabularFeatureExtractor()
        return extractor.extract_features(event, odds_snapshot, outcome=outcome, market=market)

    @staticmethod
    def create_feature_vector(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for model input.

        DEPRECATED: Use TabularFeatureExtractor.create_feature_vector() instead.

        Args:
            features: Feature dictionary from extract_features()
            feature_names: Ordered list of feature names

        Returns:
            Numpy array of feature values (fills missing with 0.0)
        """
        if not FeatureEngineering._warned:
            warnings.warn(
                "FeatureEngineering is deprecated. Use TabularFeatureExtractor from "
                "analytics.feature_extraction instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            FeatureEngineering._warned = True

        # Delegate to TabularFeatureExtractor for backward compatibility
        extractor = TabularFeatureExtractor()
        return extractor.create_feature_vector(features, feature_names)


class XGBoostStrategy(BettingStrategy):
    """
    ML-based betting strategy using XGBoost classifier.

    This strategy uses an XGBoost binary classifier to predict win probabilities,
    then bets when the model's probability exceeds the implied probability from
    odds (indicating positive expected value).

    The model is trained on historical data with features including:
    - Line movement patterns
    - Sharp vs retail odds discrepancies
    - Market consensus and efficiency
    - Closing line value

    Architecture:
    - Uses dependency injection to accept any FeatureExtractor
    - Default: TabularFeatureExtractor for snapshot-based features
    - Extensible: Can swap in custom feature extractors (e.g., SequenceFeatureExtractor for LSTM)

    Example:
        ```python
        # Default tabular features
        strategy = XGBoostStrategy()
        strategy.load_model("models/xgboost_h2h.pkl")

        # Custom feature extractor
        from analytics.feature_extraction import TabularFeatureExtractor
        extractor = TabularFeatureExtractor(sharp_bookmakers=["pinnacle", "circa"])
        strategy = XGBoostStrategy(feature_extractor=extractor)
        strategy.load_model("models/xgboost_h2h.pkl")

        # Use in backtesting
        result = await backtest_engine.run()
        ```
    """

    def __init__(
        self,
        model_path: str | None = None,
        market: str = "h2h",
        min_edge_threshold: float = 0.03,
        min_confidence: float = 0.52,
        bookmakers: list[str] | None = None,
        feature_names: list[str] | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ):
        """
        Initialize XGBoost strategy.

        Args:
            model_path: Path to saved model file (loads on init if provided)
            market: Market to bet on (h2h, spreads, totals)
            min_edge_threshold: Minimum edge required to bet (e.g., 0.03 = 3% edge)
            min_confidence: Minimum model probability to consider betting
            bookmakers: List of bookmakers to consider (default: all major books)
            feature_names: List of feature names used by model (auto-set on load)
            feature_extractor: Feature extractor to use (default: TabularFeatureExtractor)
        """
        if bookmakers is None:
            bookmakers = [
                "pinnacle",
                "fanduel",
                "draftkings",
                "betmgm",
                "williamhill_us",
                "betrivers",
                "bovada",
                "circasports",
            ]

        super().__init__(
            name="XGBoost",
            model_path=model_path,
            market=market,
            min_edge_threshold=min_edge_threshold,
            min_confidence=min_confidence,
            bookmakers=bookmakers,
        )

        self.model: Any = None  # XGBoost classifier
        self.feature_names: list[str] = feature_names or []
        self.feature_extractor: FeatureExtractor = feature_extractor or TabularFeatureExtractor()

        if model_path:
            self.load_model(model_path)

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """
        Evaluate betting opportunities using trained ML model.

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time
            config: Backtest configuration

        Returns:
            List of BetOpportunity objects with model predictions as confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        opportunities = []

        # Filter for target market and bookmakers
        market = self.params["market"]
        bookmakers = self.params["bookmakers"]
        min_edge = self.params["min_edge_threshold"]
        min_conf = self.params["min_confidence"]

        market_odds = [
            o
            for o in odds_snapshot
            if o.market_key == market and o.bookmaker_key in bookmakers
        ]

        if not market_odds:
            return []

        # Evaluate both home and away team
        for outcome in [event.home_team, event.away_team]:
            # Extract features for this outcome using injected extractor
            features = self.feature_extractor.extract_features(
                event, odds_snapshot, market=market, outcome=outcome
            )

            if not features:
                continue

            # Convert to feature vector
            feature_vector = self.feature_extractor.create_feature_vector(
                features, self.feature_names
            )

            # Get model prediction (probability of winning)
            model_prob = self._predict_probability(feature_vector)

            # Skip if below minimum confidence
            if model_prob < min_conf:
                continue

            # Find best available odds for this outcome
            outcome_odds = [o for o in market_odds if o.outcome_name == outcome]
            if not outcome_odds:
                continue

            best_odd = max(outcome_odds, key=lambda o: o.price)
            implied_prob = calculate_implied_probability(best_odd.price)

            # Calculate edge (model prob - implied prob)
            edge = model_prob - implied_prob

            # Only bet if we have positive edge above threshold
            if edge >= min_edge:
                opportunities.append(
                    BetOpportunity(
                        event_id=event.id,
                        market=market,
                        outcome=outcome,
                        bookmaker=best_odd.bookmaker_key,
                        odds=best_odd.price,
                        line=best_odd.point,
                        confidence=model_prob,  # Model probability used for Kelly sizing
                        rationale=f"XGBoost edge: {edge:.2%} (Model: {model_prob:.2%}, "
                        f"Implied: {implied_prob:.2%}) at {best_odd.bookmaker_key}",
                    )
                )

        return opportunities

    def _predict_probability(self, feature_vector: np.ndarray) -> float:
        """
        Predict win probability using trained model.

        Args:
            feature_vector: Feature array for single prediction

        Returns:
            Predicted probability (0-1)
        """
        # XGBoost predict_proba returns [[prob_class_0, prob_class_1]]
        # We want probability of class 1 (win)
        prediction = self.model.predict_proba(feature_vector.reshape(1, -1))
        return float(prediction[0][1])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        **xgb_params,
    ) -> None:
        """
        Train XGBoost model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            feature_names: List of feature names in order
            **xgb_params: Additional XGBoost parameters

        Note:
            Requires xgboost package installed
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost not installed. Install with: uv add xgboost"
            ) from e

        self.feature_names = feature_names

        # Default parameters (can be overridden)
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        }

        # Merge with user-provided params
        params = {**default_params, **xgb_params}

        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train)

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (e.g., 'models/xgboost_h2h.pkl')
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model and feature names
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

        # Update params if saved (for backward compatibility)
        if "params" in model_data:
            self.params.update(model_data["params"])

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Train or load model first.")

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores, strict=True))
