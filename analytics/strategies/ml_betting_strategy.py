"""ML-based betting strategy using feature extraction and model predictions.

This strategy bridges the new ML infrastructure (core abstractions, features, models)
with the existing backtesting system.
"""

from datetime import timedelta

from analytics.backtesting import BacktestConfig, BetOpportunity, BettingStrategy
from analytics.betting.observations import OddsObservation
from analytics.betting.problems import BettingEvent
from analytics.core.features import FeatureExtractor
from analytics.core.models import ModelPredictor
from analytics.utils import calculate_ev, calculate_implied_probability


class MLBettingStrategy(BettingStrategy):
    """ML-based betting strategy.

    Uses a feature extractor and model predictor to identify +EV betting opportunities.

    The strategy:
    1. Converts backtest events to domain objects
    2. Extracts features using the provided feature extractor
    3. Makes predictions using the model
    4. Calculates EV by comparing model predictions to retail odds
    5. Returns opportunities where EV exceeds threshold

    Works with any FeatureExtractor + ModelPredictor combination:
    - TabularFeatureExtractor + XGBoostPredictor
    - SequentialFeatureExtractor + LSTMPredictor
    - Custom extractors + custom models

    Example:
        >>> from analytics.features import TabularFeatureExtractor
        >>> from analytics.models import XGBoostPredictor
        >>> from analytics.features.betting_features import compute_sharp_retail_diff
        >>>
        >>> extractor = TabularFeatureExtractor([compute_sharp_retail_diff])
        >>> model = XGBoostPredictor("models/xgb.pkl", ["home_win", "away_win"])
        >>> strategy = MLBettingStrategy(extractor, model, min_ev_threshold=0.05)
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        model_predictor: ModelPredictor,
        min_ev_threshold: float = 0.05,
        retail_books: list[str] | None = None,
        markets: list[str] | None = None,
        max_confidence_threshold: float | None = None,
        min_confidence_threshold: float | None = None,
    ):
        """Initialize ML betting strategy.

        Args:
            feature_extractor: Feature extractor for computing features from odds
            model_predictor: Trained model for making predictions
            min_ev_threshold: Minimum expected value to bet (default: 5%)
            retail_books: List of bookmakers to bet at (default: major retail books)
            markets: Markets to consider (default: h2h, spreads, totals)
            max_confidence_threshold: Only bet if model confidence <= this (avoid overconfident)
            min_confidence_threshold: Only bet if model confidence >= this (avoid uncertain)

        Note: Feature type (tabular/sequential) must match model requirements
        """
        if retail_books is None:
            retail_books = ["fanduel", "draftkings", "betmgm", "caesars"]

        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        super().__init__(
            name=f"ML_{model_predictor.get_model_type()}",
            feature_extractor=feature_extractor,
            model_predictor=model_predictor,
            min_ev_threshold=min_ev_threshold,
            retail_books=retail_books,
            markets=markets,
            max_confidence_threshold=max_confidence_threshold,
            min_confidence_threshold=min_confidence_threshold,
        )

        # Validate feature type matches model
        extractor_type = (
            "tabular" if "Tabular" in feature_extractor.__class__.__name__ else "sequential"
        )
        required_type = model_predictor.get_required_feature_type()

        if extractor_type != required_type:
            raise ValueError(
                f"Feature extractor type ({extractor_type}) doesn't match "
                f"model requirements ({required_type})"
            )

        self.feature_extractor = feature_extractor
        self.model_predictor = model_predictor

    async def evaluate_opportunity(
        self,
        event: BettingEvent,
        observations: list[OddsObservation],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """Evaluate betting opportunities using ML predictions.

        Args:
            event: The betting event to evaluate
            observations: Odds observations at decision time
            config: Backtest configuration

        Returns:
            List of betting opportunities with positive EV
        """
        opportunities = []

        if not observations:
            return []

        # Calculate decision time based on config
        decision_time = event.commence_time - timedelta(hours=config.decision_hours_before_game)

        # Extract features (automatically filters observations by decision_time)
        try:
            features = self.feature_extractor.extract(event, observations, decision_time)
        except Exception:
            # If feature extraction fails (e.g., insufficient data), skip this event
            return []

        # Get model prediction
        prediction = self.model_predictor.predict(features)

        # Apply confidence thresholds if set
        if self.params.get("min_confidence_threshold") is not None:
            if prediction.confidence < self.params["min_confidence_threshold"]:
                return []

        if self.params.get("max_confidence_threshold") is not None:
            if prediction.confidence > self.params["max_confidence_threshold"]:
                return []

        # Find +EV opportunities by comparing predictions to retail odds
        for market in self.params["markets"]:
            market_obs = [o for o in observations if o.market == market]

            if not market_obs:
                continue

            # For each outcome, check if we have +EV at any retail book
            for outcome_name, predicted_prob in prediction.predictions.items():
                # Map prediction outcome to team names
                # (assumes model outputs match team names or standard outcomes)
                target_outcome = self._map_outcome_to_team(outcome_name, event, market)

                if target_outcome is None:
                    continue

                # Check each retail book for +EV
                for bookmaker in self.params["retail_books"]:
                    # Find odds for this outcome at this bookmaker
                    outcome_obs = next(
                        (
                            o
                            for o in market_obs
                            if o.bookmaker == bookmaker and o.outcome == target_outcome
                        ),
                        None,
                    )

                    if outcome_obs is None:
                        continue

                    # Calculate expected value
                    retail_implied_prob = calculate_implied_probability(outcome_obs.odds)
                    ev = calculate_ev(predicted_prob, outcome_obs.odds)

                    # If EV exceeds threshold, create opportunity
                    if ev >= self.params["min_ev_threshold"]:
                        opportunities.append(
                            BetOpportunity(
                                event_id=event.id,
                                market=market,
                                outcome=target_outcome,
                                bookmaker=bookmaker,
                                odds=outcome_obs.odds,
                                line=outcome_obs.line,
                                confidence=prediction.confidence,
                                rationale=(
                                    f"ML prediction: {predicted_prob:.3f} "
                                    f"vs retail: {retail_implied_prob:.3f} "
                                    f"(EV: {ev:.3f})"
                                ),
                            )
                        )

        return opportunities

    def _map_outcome_to_team(
        self, outcome_name: str, event: BettingEvent, market: str
    ) -> str | None:
        """Map model output name to team/outcome name.

        Args:
            outcome_name: Model's outcome name (e.g., "home_win", "away_win")
            event: The betting event
            market: Market type

        Returns:
            Mapped outcome name for the odds, or None if can't map

        Note: This is a simple heuristic mapper. For production, you might want
        a more sophisticated mapping strategy.
        """
        outcome_lower = outcome_name.lower()

        if market == "h2h":
            # Moneyline: map to team names
            if "home" in outcome_lower:
                return event.home_competitor
            elif "away" in outcome_lower:
                return event.away_competitor
        elif market == "spreads":
            # Spreads: map to team names (model should predict cover probability)
            if "home" in outcome_lower:
                return event.home_competitor
            elif "away" in outcome_lower:
                return event.away_competitor
        elif market == "totals":
            # Totals: map to Over/Under
            if "over" in outcome_lower:
                return "Over"
            elif "under" in outcome_lower:
                return "Under"

        # If outcome_name matches team name exactly, use it
        if outcome_name == event.home_competitor or outcome_name == event.away_competitor:
            return outcome_name

        # If outcome_name matches Over/Under exactly, use it
        if outcome_name in ["Over", "Under"]:
            return outcome_name

        return None

    def get_feature_extractor(self) -> FeatureExtractor:
        """Get the feature extractor.

        Returns:
            Feature extractor instance
        """
        return self.feature_extractor

    def get_model_predictor(self) -> ModelPredictor:
        """Get the model predictor.

        Returns:
            Model predictor instance
        """
        return self.model_predictor
