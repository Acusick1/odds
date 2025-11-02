"""Model prediction abstractions.

Defines how models make predictions from features.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .features import FeatureSet


class Prediction(BaseModel):
    """Output from a model prediction.

    Flexible structure that works for both classification and regression:
    - Classification: predictions are probabilities for each class
    - Regression: predictions are predicted values
    """

    predictions: dict[str, float] = Field(
        description="Model predictions - probabilities or values keyed by outcome/target"
    )

    confidence: float = Field(
        description="Overall confidence in prediction (0-1)",
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the prediction",
    )

    def get_most_likely_outcome(self) -> tuple[str, float]:
        """Get the outcome with highest predicted probability/value.

        Returns:
            Tuple of (outcome_name, probability/value)
        """
        max_outcome = max(self.predictions.items(), key=lambda x: x[1])
        return max_outcome

    def get_prediction_for(self, outcome: str) -> float | None:
        """Get prediction for a specific outcome.

        Args:
            outcome: The outcome name to get prediction for

        Returns:
            Predicted probability/value, or None if outcome not found
        """
        return self.predictions.get(outcome)


class ModelPredictor(ABC):
    """Base class for all model predictors.

    Wraps any type of trained model (XGBoost, LSTM, etc.) with a
    consistent interface for making predictions.

    The predictor is responsible for:
    1. Loading a trained model
    2. Converting features to model's expected format
    3. Running inference
    4. Converting outputs to Prediction format
    """

    @abstractmethod
    def predict(self, features: FeatureSet) -> Prediction:
        """Make a prediction from features.

        Args:
            features: Extracted features (tabular or sequential)

        Returns:
            Prediction containing probabilities/values and metadata

        Raises:
            ValueError: If feature format doesn't match model requirements
                       (e.g., LSTM getting tabular features)
        """
        pass

    def get_model_type(self) -> str:
        """Return a string identifying the model type.

        Returns:
            Model type like "xgboost", "lstm", "random_forest"

        This is useful for logging and debugging.
        """
        return self.__class__.__name__

    def get_required_feature_type(self) -> str:
        """Return the required feature type for this model.

        Returns:
            "tabular" or "sequential"

        This helps validate that the feature extractor matches the model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement get_required_feature_type()"
        )
