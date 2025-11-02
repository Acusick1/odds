"""Model wrappers for ML predictions."""

from analytics.models.lstm_model import LSTMPredictor
from analytics.models.xgboost_model import XGBoostPredictor

__all__ = ["XGBoostPredictor", "LSTMPredictor"]
