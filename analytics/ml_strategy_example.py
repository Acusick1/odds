"""
XGBoost ML Strategy Example.

This module demonstrates how to build an ML-based betting strategy using XGBoost.
It serves as a reference implementation for creating machine learning strategies.

Key Features:
- Feature engineering from historical odds data
- XGBoost binary classifier for win probability prediction
- Integration with backtesting framework via BetOpportunity.confidence
- Model persistence (save/load weights)
- Comprehensive feature set including line movement, market efficiency, consensus

Dependencies (install with uv):
    uv add xgboost scikit-learn numpy
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity, BettingStrategy
from analytics.utils import american_to_decimal, calculate_implied_probability, calculate_market_hold
from core.models import Odds


class FeatureEngineering:
    """
    Feature engineering for sports betting ML models.

    Extracts features from event data and odds snapshots that capture:
    - Line movement (sharp vs public)
    - Market efficiency (vig, consensus)
    - Closing line value
    - Time-to-game factors
    - Sharp vs retail odds discrepancies
    """

    @staticmethod
    def extract_features(
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        market: str = "h2h",
        outcome: str | None = None,
    ) -> dict[str, float]:
        """
        Extract ML features from event and odds snapshot.

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time
            market: Market to analyze (h2h, spreads, totals)
            outcome: Specific outcome to analyze (if None, analyzes both sides)

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Filter for target market
        market_odds = [o for o in odds_snapshot if o.market_key == market]

        if not market_odds:
            return features

        # Sharp bookmaker features (Pinnacle as baseline)
        sharp_book = "pinnacle"
        sharp_odds = [o for o in market_odds if o.bookmaker_key == sharp_book]

        # Retail bookmaker features (FanDuel, DraftKings, BetMGM)
        retail_books = ["fanduel", "draftkings", "betmgm"]
        retail_odds = [o for o in market_odds if o.bookmaker_key in retail_books]

        # 1. Market consensus features
        if market == "h2h":
            home_odds_list = [o for o in market_odds if o.outcome_name == event.home_team]
            away_odds_list = [o for o in market_odds if o.outcome_name == event.away_team]

            if home_odds_list and away_odds_list:
                features["avg_home_odds"] = np.mean([o.price for o in home_odds_list])
                features["avg_away_odds"] = np.mean([o.price for o in away_odds_list])
                features["std_home_odds"] = np.std([o.price for o in home_odds_list])
                features["std_away_odds"] = np.std([o.price for o in away_odds_list])

                # Market consensus (average implied probability)
                avg_home_prob = np.mean(
                    [calculate_implied_probability(o.price) for o in home_odds_list]
                )
                avg_away_prob = np.mean(
                    [calculate_implied_probability(o.price) for o in away_odds_list]
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
                    avg_retail_home_prob = np.mean(
                        [calculate_implied_probability(o.price) for o in retail_home_odds]
                    )
                    features["retail_sharp_diff_home"] = avg_retail_home_prob - sharp_home_prob

                if retail_away_odds:
                    avg_retail_away_prob = np.mean(
                        [calculate_implied_probability(o.price) for o in retail_away_odds]
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
        all_books = set(o.bookmaker_key for o in market_odds)
        features["num_bookmakers"] = len(all_books)

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
                features["avg_market_hold"] = np.mean(holds)
                features["std_market_hold"] = np.std(holds)

        # 4. Best available odds features (line shopping)
        if market == "h2h":
            home_prices = [o.price for o in market_odds if o.outcome_name == event.home_team]
            away_prices = [o.price for o in market_odds if o.outcome_name == event.away_team]

            if home_prices:
                features["best_home_odds"] = max(home_prices)
                features["worst_home_odds"] = min(home_prices)
                features["home_odds_range"] = max(home_prices) - min(home_prices)

            if away_prices:
                features["best_away_odds"] = max(away_prices)
                features["worst_away_odds"] = min(away_prices)
                features["away_odds_range"] = max(away_prices) - min(away_prices)

            # Set outcome-specific best odds
            if outcome == event.home_team and home_prices:
                features["best_available_odds"] = max(home_prices)
                features["odds_range"] = max(home_prices) - min(home_prices)
            elif outcome == event.away_team and away_prices:
                features["best_available_odds"] = max(away_prices)
                features["odds_range"] = max(away_prices) - min(away_prices)

        # 5. Decimal odds features (for model friendliness)
        if "best_available_odds" in features:
            features["best_available_decimal"] = american_to_decimal(
                int(features["best_available_odds"])
            )

        # 6. Binary features
        features["is_home_team"] = 1.0 if outcome == event.home_team else 0.0
        features["is_away_team"] = 1.0 if outcome == event.away_team else 0.0

        return features

    @staticmethod
    def create_feature_vector(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for model input.

        Args:
            features: Feature dictionary from extract_features()
            feature_names: Ordered list of feature names

        Returns:
            Numpy array of feature values (fills missing with 0.0)
        """
        return np.array([features.get(name, 0.0) for name in feature_names])


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

    Example:
        ```python
        # Train model (see notebook)
        strategy = XGBoostStrategy()
        await strategy.train(training_data, labels)
        strategy.save_model("models/xgboost_h2h.pkl")

        # Use in backtesting
        strategy = XGBoostStrategy()
        strategy.load_model("models/xgboost_h2h.pkl")
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
            # Extract features for this outcome
            features = FeatureEngineering.extract_features(
                event, odds_snapshot, market=market, outcome=outcome
            )

            if not features:
                continue

            # Convert to feature vector
            feature_vector = FeatureEngineering.create_feature_vector(
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
