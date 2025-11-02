"""Betting strategy abstractions."""

from abc import ABC, abstractmethod
from typing import Any

from analytics.core import FeatureExtractor, ModelPredictor

from .observations import OddsObservation
from .problems import BettingEvent


class BettingStrategy(ABC):
    """Base class for all betting strategies.

    A betting strategy evaluates betting events and identifies betting opportunities.
    It can use:
    - Rule-based logic (e.g., bet when odds move)
    - Statistical models (e.g., bet when EV > threshold)
    - Machine learning models (e.g., XGBoost, LSTM predictions)

    The strategy has access to:
    - Feature extractors (to compute features from odds history)
    - Model predictors (to make ML predictions)
    - Strategy-specific parameters
    """

    def __init__(
        self,
        name: str,
        feature_extractor: FeatureExtractor | None = None,
        model_predictor: ModelPredictor | None = None,
        **params: Any,
    ):
        """Initialize betting strategy.

        Args:
            name: Strategy name (for logging/reporting)
            feature_extractor: Optional feature extractor for ML strategies
            model_predictor: Optional model predictor for ML strategies
            **params: Strategy-specific parameters
        """
        self.name = name
        self.feature_extractor = feature_extractor
        self.model_predictor = model_predictor
        self.params = params

    @abstractmethod
    async def evaluate_opportunity(
        self,
        event: BettingEvent,
        observations: list[OddsObservation],
        config: Any,
    ) -> list[Any]:  # Returns BetOpportunity but avoid circular import
        """Evaluate a betting event and identify opportunities.

        Args:
            event: The betting event to evaluate
            observations: Historical odds observations for this event
            config: Configuration (e.g., BacktestConfig)

        Returns:
            List of betting opportunities identified by this strategy

        Note: Implementations should respect config.decision_hours_before_game
        or similar timing constraints to prevent look-ahead bias.
        """
        pass

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name
        """
        return self.name

    def get_params(self) -> dict[str, Any]:
        """Get strategy parameters.

        Returns:
            Dictionary of strategy parameters
        """
        return self.params.copy()

    def has_model(self) -> bool:
        """Check if strategy uses a trained model.

        Returns:
            True if model_predictor is set
        """
        return self.model_predictor is not None

    def has_feature_extractor(self) -> bool:
        """Check if strategy uses feature extraction.

        Returns:
            True if feature_extractor is set
        """
        return self.feature_extractor is not None

    def is_ml_strategy(self) -> bool:
        """Check if this is a machine learning strategy.

        Returns:
            True if both feature extractor and model are present
        """
        return self.has_feature_extractor() and self.has_model()


class RuleBasedStrategy(BettingStrategy):
    """Base class for rule-based strategies.

    Rule-based strategies don't use ML models - they make decisions based on
    logic, thresholds, and statistical calculations.

    Examples:
    - Bet when EV > threshold using sharp vs retail odds
    - Bet when line moves by X points
    - Bet when market hold < Y%
    """

    def __init__(self, name: str, **params: Any):
        """Initialize rule-based strategy.

        Args:
            name: Strategy name
            **params: Strategy parameters
        """
        # Rule-based strategies don't use feature extractors or models
        super().__init__(name, feature_extractor=None, model_predictor=None, **params)
