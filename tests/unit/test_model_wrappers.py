"""Unit tests for model wrappers (XGBoost and LSTM)."""

import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn
import xgboost as xgb

from analytics.core.features import SequentialFeatureSet, TabularFeatureSet
from analytics.core.models import Prediction
from analytics.models import LSTMPredictor, XGBoostPredictor


class SimpleLSTMModel(nn.Module):
    """Simple LSTM model for testing (defined at module level for pickling)."""

    def __init__(self, input_size=5, hidden_size=16, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        fc_out = self.fc(last_output)
        return self.softmax(fc_out)


class TestXGBoostPredictor:
    """Tests for XGBoostPredictor."""

    @pytest.fixture
    def dummy_xgb_model_path(self, tmp_path):
        """Create a dummy XGBoost model for testing."""
        # Use the train() API (Booster) which is more stable
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "binary:logistic", "max_depth": 3}
        model = xgb.train(params, dtrain, num_boost_round=10)

        # Save to temporary file as pickle (XGBoost model wrapper handles both)
        model_path = tmp_path / "xgb_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return model_path

    @pytest.fixture
    def dummy_xgb_booster_path(self, tmp_path):
        """Create a dummy XGBoost Booster for testing."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "binary:logistic", "max_depth": 3}
        booster = xgb.train(params, dtrain, num_boost_round=10)

        # Save to temporary file
        model_path = tmp_path / "xgb_booster.json"
        booster.save_model(str(model_path))

        return model_path

    def test_load_sklearn_api_model(self, dummy_xgb_model_path):
        """Test loading XGBoost model with scikit-learn API."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.model is not None
        assert predictor.output_names == ["class_0", "class_1"]

    def test_load_booster_model(self, dummy_xgb_booster_path):
        """Test loading XGBoost Booster model."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_booster_path,
            output_names=["probability"],
        )

        assert predictor.model is not None

    def test_file_not_found(self):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            XGBoostPredictor(
                model_path="/nonexistent/model.pkl",
                output_names=["class_0", "class_1"],
            )

    def test_predict_with_tabular_features(self, dummy_xgb_model_path):
        """Test making predictions with tabular features."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        features = TabularFeatureSet(
            {
                "feat_0": 0.5,
                "feat_1": 0.3,
                "feat_2": 0.7,
                "feat_3": 0.2,
                "feat_4": 0.9,
            }
        )

        prediction = predictor.predict(features)

        assert isinstance(prediction, Prediction)
        assert set(prediction.predictions.keys()) == {"class_0", "class_1"}
        assert 0 <= prediction.confidence <= 1
        assert prediction.metadata["model_type"] == "xgboost"

    def test_predict_with_wrong_feature_type(self, dummy_xgb_model_path):
        """Test error when using sequential features with XGBoost."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        from datetime import datetime

        sequences = np.random.rand(10, 5)
        timestamps = [datetime(2024, 1, 1) for _ in range(10)]
        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            timestamps=timestamps,
        )

        with pytest.raises(ValueError, match="requires TabularFeatureSet"):
            predictor.predict(features)

    def test_feature_name_validation(self, dummy_xgb_model_path):
        """Test feature name validation."""
        expected_names = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
            feature_names=expected_names,
        )

        # Correct feature names should work
        features = TabularFeatureSet({name: 0.5 for name in expected_names})
        prediction = predictor.predict(features)
        assert isinstance(prediction, Prediction)

        # Wrong feature names should fail
        wrong_features = TabularFeatureSet(
            {"wrong_0": 0.5, "wrong_1": 0.3, "wrong_2": 0.7, "wrong_3": 0.2, "wrong_4": 0.9}
        )
        with pytest.raises(ValueError, match="Feature names mismatch"):
            predictor.predict(wrong_features)

    def test_get_required_feature_type(self, dummy_xgb_model_path):
        """Test that XGBoost requires tabular features."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.get_required_feature_type() == "tabular"

    def test_get_model_type(self, dummy_xgb_model_path):
        """Test model type identifier."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.get_model_type() == "xgboost"

    def test_prediction_probabilities_sum_to_one(self, dummy_xgb_model_path):
        """Test that probabilities roughly sum to 1 for binary classification."""
        predictor = XGBoostPredictor(
            model_path=dummy_xgb_model_path,
            output_names=["class_0", "class_1"],
        )

        features = TabularFeatureSet(
            {
                "feat_0": 0.5,
                "feat_1": 0.3,
                "feat_2": 0.7,
                "feat_3": 0.2,
                "feat_4": 0.9,
            }
        )

        prediction = predictor.predict(features)
        prob_sum = sum(prediction.predictions.values())

        # Should be close to 1.0 (allowing for floating point errors)
        assert 0.99 <= prob_sum <= 1.01


class TestLSTMPredictor:
    """Tests for LSTMPredictor."""

    @pytest.fixture
    def dummy_pytorch_model_path(self, tmp_path):
        """Create a dummy PyTorch LSTM model."""

        # Use module-level SimpleLSTMModel class
        model = SimpleLSTMModel(input_size=5, hidden_size=16, output_size=2)
        model.eval()

        # Save model
        model_path = tmp_path / "lstm_model.pt"
        torch.save(model, str(model_path))

        return model_path

    def test_load_pytorch_model(self, dummy_pytorch_model_path):
        """Test loading PyTorch LSTM model."""
        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.model is not None

    def test_file_not_found(self):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LSTMPredictor(
                model_path="/nonexistent/model.pt",
                output_names=["class_0", "class_1"],
            )

    def test_predict_with_sequential_features(self, dummy_pytorch_model_path):
        """Test making predictions with sequential features."""
        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        from datetime import datetime

        sequences = np.random.rand(10, 5)
        timestamps = [datetime(2024, 1, 1) for _ in range(10)]
        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            timestamps=timestamps,
        )

        prediction = predictor.predict(features)

        assert isinstance(prediction, Prediction)
        assert set(prediction.predictions.keys()) == {"class_0", "class_1"}
        assert 0 <= prediction.confidence <= 1
        assert prediction.metadata["model_type"] == "lstm"
        assert prediction.metadata["framework"] == "pytorch"
        assert prediction.metadata["sequence_length"] == 10
        assert prediction.metadata["feature_count"] == 5

    def test_predict_with_wrong_feature_type(self, dummy_pytorch_model_path):
        """Test error when using tabular features with LSTM."""
        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        features = TabularFeatureSet({"f1": 0.5, "f2": 0.3, "f3": 0.7, "f4": 0.2, "f5": 0.9})

        with pytest.raises(ValueError, match="requires SequentialFeatureSet"):
            predictor.predict(features)

    def test_feature_name_validation(self, dummy_pytorch_model_path):
        """Test feature name validation."""

        expected_names = ["f1", "f2", "f3", "f4", "f5"]
        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
            feature_names=expected_names,
        )

        from datetime import datetime

        # Correct feature names should work
        sequences = np.random.rand(10, 5)
        timestamps = [datetime(2024, 1, 1) for _ in range(10)]
        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=expected_names,
            timestamps=timestamps,
        )
        prediction = predictor.predict(features)
        assert isinstance(prediction, Prediction)

        # Wrong feature names should fail
        wrong_features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["wrong1", "wrong2", "wrong3", "wrong4", "wrong5"],
            timestamps=timestamps,
        )
        with pytest.raises(ValueError, match="Feature names mismatch"):
            predictor.predict(wrong_features)

    def test_get_required_feature_type(self, dummy_pytorch_model_path):
        """Test that LSTM requires sequential features."""

        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.get_required_feature_type() == "sequential"

    def test_get_model_type(self, dummy_pytorch_model_path):
        """Test model type identifier."""

        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        assert predictor.get_model_type() == "lstm"

    def test_prediction_probabilities_sum_to_one(self, dummy_pytorch_model_path):
        """Test that probabilities sum to 1 for classification."""

        predictor = LSTMPredictor(
            model_path=dummy_pytorch_model_path,
            output_names=["class_0", "class_1"],
        )

        from datetime import datetime

        sequences = np.random.rand(10, 5)
        timestamps = [datetime(2024, 1, 1) for _ in range(10)]
        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            timestamps=timestamps,
        )

        prediction = predictor.predict(features)
        prob_sum = sum(prediction.predictions.values())

        # Should be close to 1.0 (softmax output)
        assert 0.99 <= prob_sum <= 1.01


class TestModelCompatibility:
    """Test that models work correctly with the abstraction layer."""

    def test_xgboost_predictor_interface(self, tmp_path):
        """Test that XGBoostPredictor implements ModelPredictor interface."""
        # Create dummy model using Booster API
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "binary:logistic", "max_depth": 2}
        model = xgb.train(params, dtrain, num_boost_round=5)

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        predictor = XGBoostPredictor(
            model_path=model_path,
            output_names=["loss", "win"],
        )

        # Should implement all required methods
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "get_required_feature_type")
        assert hasattr(predictor, "get_model_type")

        # Should return correct types
        features = TabularFeatureSet({"f1": 0.5, "f2": 0.3, "f3": 0.8})
        prediction = predictor.predict(features)
        assert isinstance(prediction, Prediction)
        assert isinstance(predictor.get_required_feature_type(), str)
        assert isinstance(predictor.get_model_type(), str)

    def test_lstm_predictor_interface(self, tmp_path):
        """Test that LSTMPredictor implements ModelPredictor interface."""

        # Use module-level SimpleLSTMModel
        model = SimpleLSTMModel(input_size=3, hidden_size=8, output_size=2)
        model.eval()

        model_path = tmp_path / "lstm.pt"
        torch.save(model, str(model_path))

        predictor = LSTMPredictor(
            model_path=model_path,
            output_names=["loss", "win"],
        )

        # Should implement all required methods
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "get_required_feature_type")
        assert hasattr(predictor, "get_model_type")

        # Should return correct types
        from datetime import datetime

        sequences = np.random.rand(5, 3)
        timestamps = [datetime(2024, 1, 1) for _ in range(5)]
        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["f1", "f2", "f3"],
            timestamps=timestamps,
        )
        prediction = predictor.predict(features)
        assert isinstance(prediction, Prediction)
        assert isinstance(predictor.get_required_feature_type(), str)
        assert isinstance(predictor.get_model_type(), str)
