"""LSTM model wrapper for predictions.

Provides a consistent interface for LSTM/RNN models to work with the
feature extraction and backtesting infrastructure.

PyTorch-only implementation.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from analytics.core.features import FeatureSet, SequentialFeatureSet
from analytics.core.models import ModelPredictor, Prediction


class LSTMPredictor(ModelPredictor):
    """Wrapper for PyTorch LSTM/RNN models.

    Loads a trained sequential model and makes predictions from time series features.

    Supports:
    - PyTorch models (.pt, .pth)
    - Pickled PyTorch models (.pkl)

    Example:
        >>> predictor = LSTMPredictor(
        ...     model_path="models/lstm_model.pt",
        ...     output_names=["home_win", "away_win"],
        ... )
        >>> features = SequentialFeatureSet(...)
        >>> prediction = predictor.predict(features)
        >>> prediction.predictions  # {"home_win": 0.45, "away_win": 0.55}
    """

    def __init__(
        self,
        model_path: str | Path,
        output_names: list[str],
        feature_names: list[str] | None = None,
    ):
        """Initialize LSTM predictor.

        Args:
            model_path: Path to saved PyTorch model file
            output_names: Names of output classes/targets
            feature_names: Expected feature names. If None, will accept any features.

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

    def _load_model(self) -> nn.Module:
        """Load the PyTorch model from disk.

        Returns:
            Loaded PyTorch model in evaluation mode

        Raises:
            ValueError: If model can't be loaded
        """
        try:
            # Try loading as PyTorch model
            # Note: Using weights_only=False for compatibility with custom models
            # This is safe for models we control, but be cautious with untrusted sources
            model = torch.load(str(self.model_path), weights_only=False)

            # If it's a state dict, we'd need the model architecture
            # For now, assume it's the full model
            if not isinstance(model, nn.Module):
                raise ValueError(
                    f"Loaded object is not a PyTorch model. Got {type(model)}. "
                    "Please save the full model, not just the state dict."
                )

            model.eval()  # Set to evaluation mode
            return model

        except Exception as e:
            # Try loading as pickle
            try:
                with open(self.model_path, "rb") as f:
                    model = pickle.load(f)

                if not isinstance(model, nn.Module):
                    raise ValueError(f"Pickled object is not a PyTorch model. Got {type(model)}")

                model.eval()
                return model

            except Exception as e2:
                raise ValueError(
                    f"Failed to load PyTorch model from {self.model_path}. "
                    f"torch.load error: {e}. Pickle error: {e2}"
                ) from e2

    def predict(self, features: FeatureSet) -> Prediction:
        """Make prediction from sequential features.

        Args:
            features: SequentialFeatureSet containing the time series

        Returns:
            Prediction with probabilities/values for each output

        Raises:
            ValueError: If features are not SequentialFeatureSet
            ValueError: If feature names don't match expected
        """
        if not isinstance(features, SequentialFeatureSet):
            raise ValueError(
                f"LSTMPredictor requires SequentialFeatureSet, got {type(features).__name__}"
            )

        # Validate feature names if specified
        if self.feature_names is not None:
            feature_names = features.get_feature_names()
            if feature_names != self.feature_names:
                raise ValueError(
                    f"Feature names mismatch. Expected {self.feature_names}, "
                    f"got {feature_names}"
                )

        # Get array and add batch dimension: (seq_len, n_features) -> (1, seq_len, n_features)
        feature_array = features.to_array()
        feature_array = np.expand_dims(feature_array, axis=0)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(feature_array).float()

        # Make prediction
        with torch.no_grad():
            output = self.model(tensor)

        # Convert back to numpy
        prediction = output.cpu().numpy()

        # Handle different output formats
        if len(prediction.shape) == 2:
            # (1, n_outputs) -> (n_outputs,)
            pred_probs = prediction[0]
        else:
            # Single output
            pred_probs = np.array([prediction[0]])

        # Convert to dict mapping output names to probabilities
        if len(pred_probs) != len(self.output_names):
            raise ValueError(
                f"Model output length ({len(pred_probs)}) doesn't match "
                f"output_names length ({len(self.output_names)})"
            )

        predictions_dict = {
            name: float(prob) for name, prob in zip(self.output_names, pred_probs, strict=True)
        }

        # Calculate confidence as the max probability
        confidence = float(max(pred_probs))

        return Prediction(
            predictions=predictions_dict,
            confidence=confidence,
            metadata={
                "model_type": "lstm",
                "framework": "pytorch",
                "model_path": str(self.model_path),
                "sequence_length": features.get_shape()[0],
                "feature_count": features.get_shape()[1],
            },
        )

    def get_required_feature_type(self) -> str:
        """Return required feature type.

        Returns:
            "sequential" - LSTM requires sequential features
        """
        return "sequential"

    def get_model_type(self) -> str:
        """Return model type identifier.

        Returns:
            "lstm"
        """
        return "lstm"
