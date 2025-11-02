"""XGBoost model wrapper for predictions.

Provides a consistent interface for XGBoost models to work with the
feature extraction and backtesting infrastructure.
"""

import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from analytics.core.features import FeatureSet, TabularFeatureSet
from analytics.core.models import ModelPredictor, Prediction


class XGBoostPredictor(ModelPredictor):
    """Wrapper for XGBoost models.

    Loads a trained XGBoost model and makes predictions from tabular features.

    Example:
        >>> predictor = XGBoostPredictor(
        ...     model_path="models/xgb_model.pkl",
        ...     output_names=["home_win", "away_win"]
        ... )
        >>> features = TabularFeatureSet({...})
        >>> prediction = predictor.predict(features)
        >>> prediction.predictions  # {"home_win": 0.45, "away_win": 0.55}
    """

    def __init__(
        self,
        model_path: str | Path,
        output_names: list[str],
        feature_names: list[str] | None = None,
    ):
        """Initialize XGBoost predictor.

        Args:
            model_path: Path to saved model file (.pkl or .json)
            output_names: Names of output classes/targets (e.g., ["home_win", "away_win"])
            feature_names: Expected feature names in order. If None, will accept any features.

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model can't be loaded
        """
        self.model_path = Path(model_path)
        self.output_names = output_names
        self.feature_names = feature_names

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = self._load_model()

    def _load_model(self) -> xgb.Booster | xgb.XGBClassifier | xgb.XGBRegressor:
        """Load the XGBoost model from disk.

        Returns:
            Loaded XGBoost model

        Raises:
            ValueError: If model can't be loaded
        """
        try:
            # Try loading as pickle first (common format)
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            # Try loading as native XGBoost JSON format
            try:
                model = xgb.Booster()
                model.load_model(str(self.model_path))
                return model
            except Exception as e2:
                raise ValueError(
                    f"Failed to load model from {self.model_path}. "
                    f"Pickle error: {e}. JSON error: {e2}"
                ) from e2

    def predict(self, features: FeatureSet) -> Prediction:
        """Make prediction from tabular features.

        Args:
            features: TabularFeatureSet containing the features

        Returns:
            Prediction with probabilities/values for each output

        Raises:
            ValueError: If features are not TabularFeatureSet
            ValueError: If feature names don't match expected (if feature_names set)
        """
        if not isinstance(features, TabularFeatureSet):
            raise ValueError(
                f"XGBoostPredictor requires TabularFeatureSet, got {type(features).__name__}"
            )

        # Validate feature names if specified
        if self.feature_names is not None:
            feature_names = features.get_feature_names()
            if feature_names != self.feature_names:
                raise ValueError(
                    f"Feature names mismatch. Expected {self.feature_names}, "
                    f"got {feature_names}"
                )

        # Convert to array and reshape for XGBoost
        feature_array = features.to_array().reshape(1, -1)

        # Make prediction based on model type
        if isinstance(self.model, xgb.XGBClassifier | xgb.XGBRegressor):
            # Scikit-learn API
            if hasattr(self.model, "predict_proba"):
                # Classifier with probabilities
                pred_probs = self.model.predict_proba(feature_array)[0]
            else:
                # Regressor or classifier without probabilities
                pred_values = self.model.predict(feature_array)
                pred_probs = pred_values if len(pred_values.shape) > 1 else [pred_values[0]]
        else:
            # Native XGBoost Booster API
            dmatrix = xgb.DMatrix(feature_array)
            pred_probs = self.model.predict(dmatrix)[0]

            # Handle single output case
            if not isinstance(pred_probs, list | np.ndarray):
                pred_probs = [pred_probs]

        # Convert to dict mapping output names to probabilities
        # Handle binary classification case where model returns single probability
        if len(pred_probs) == 1 and len(self.output_names) == 2:
            # Binary classification: p(class_1) = pred, p(class_0) = 1 - pred
            prob_positive = float(pred_probs[0])
            predictions_dict = {
                self.output_names[0]: 1 - prob_positive,
                self.output_names[1]: prob_positive,
            }
            pred_probs = [1 - prob_positive, prob_positive]
        elif len(pred_probs) != len(self.output_names):
            raise ValueError(
                f"Model output length ({len(pred_probs)}) doesn't match "
                f"output_names length ({len(self.output_names)})"
            )
        else:
            predictions_dict = {
                name: float(prob) for name, prob in zip(self.output_names, pred_probs, strict=False)
            }

        # Calculate confidence as the max probability
        confidence = float(max(pred_probs))

        return Prediction(
            predictions=predictions_dict,
            confidence=confidence,
            metadata={
                "model_type": "xgboost",
                "model_path": str(self.model_path),
                "feature_count": len(features.to_array()),
            },
        )

    def get_required_feature_type(self) -> str:
        """Return required feature type.

        Returns:
            "tabular" - XGBoost requires tabular features
        """
        return "tabular"

    def get_model_type(self) -> str:
        """Return model type identifier.

        Returns:
            "xgboost"
        """
        return "xgboost"
