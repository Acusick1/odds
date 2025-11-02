"""Integration tests for ML pipeline (feature extraction → model → strategy → backtest).

Tests the complete ML workflow:
1. Feature extraction from historical odds
2. Model prediction
3. Strategy evaluation
4. (Optional) Full backtesting integration

Covers both:
- Tabular features + XGBoost
- Sequential features + LSTM
"""

import pickle
from datetime import datetime, timedelta

import numpy as np
import pytest

# Import torch at module level for proper pickling
import torch
import torch.nn as nn
import xgboost as xgb

from analytics.backtesting import BacktestConfig, BetOpportunity
from analytics.betting.problems import BettingEvent
from analytics.core.features import SequentialFeatureSet, TabularFeatureSet
from analytics.features.betting_features import (
    compute_market_hold,
    compute_sharp_retail_diff,
    compute_timing_features,
)
from analytics.features.sequential import SequentialFeatureExtractor
from analytics.features.tabular import TabularFeatureExtractor
from analytics.models import LSTMPredictor, XGBoostPredictor
from analytics.strategies import MLBettingStrategy
from core.models import Odds


class SimpleLSTMModel(nn.Module):
    """Simple LSTM model for testing (defined at module level for pickling)."""

    def __init__(self, input_size=3, hidden_size=16, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        fc_out = self.fc(last_output)
        return self.softmax(fc_out)


class TestXGBoostPipeline:
    """Test complete pipeline with XGBoost model."""

    @pytest.fixture
    def trained_xgb_model(self, tmp_path):
        """Create and save a trained XGBoost model."""
        # Create synthetic training data
        # 5 features to match the feature extractors:
        # - sharp_retail_diff_home, sharp_retail_diff_away (2)
        # - market_hold_avg (1)
        # - hours_to_game, day_of_week (2)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)  # Binary classification

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "xgb_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return model_path

    @pytest.fixture
    def tabular_extractor(self):
        """Create tabular feature extractor with betting features."""
        return TabularFeatureExtractor(
            feature_computers=[
                compute_sharp_retail_diff,
                compute_market_hold,
                compute_timing_features,
            ]
        )

    @pytest.fixture
    def xgb_predictor(self, trained_xgb_model):
        """Create XGBoost predictor."""
        return XGBoostPredictor(
            model_path=trained_xgb_model,
            output_names=["home_win", "away_win"],
        )

    def test_tabular_feature_extraction(self, tabular_extractor):
        """Test that tabular features can be extracted from odds."""
        from analytics.betting.observations import OddsObservation
        from analytics.betting.problems import BettingEvent

        # Create test event
        event_time = datetime(2024, 1, 15, 19, 0)
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Warriors",
            commence_time=event_time,
            home_score=None,
            away_score=None,
        )

        # Create mock observations (one per outcome per bookmaker)
        obs_time = event_time - timedelta(hours=2)
        observations = [
            # Pinnacle - Lakers
            OddsObservation(
                event_id="game1",
                observation_time=obs_time,
                bookmaker="pinnacle",
                market="h2h",
                outcome="Lakers",
                odds=-110,
                line=None,
            ),
            # Pinnacle - Warriors
            OddsObservation(
                event_id="game1",
                observation_time=obs_time,
                bookmaker="pinnacle",
                market="h2h",
                outcome="Warriors",
                odds=-110,
                line=None,
            ),
            # FanDuel - Lakers
            OddsObservation(
                event_id="game1",
                observation_time=obs_time,
                bookmaker="fanduel",
                market="h2h",
                outcome="Lakers",
                odds=-105,
                line=None,
            ),
            # FanDuel - Warriors
            OddsObservation(
                event_id="game1",
                observation_time=obs_time,
                bookmaker="fanduel",
                market="h2h",
                outcome="Warriors",
                odds=-115,
                line=None,
            ),
        ]

        # Extract features
        decision_time = event_time - timedelta(hours=1)
        features = tabular_extractor.extract(event, observations, decision_time)

        # Verify features
        assert isinstance(features, TabularFeatureSet)
        assert len(features.get_feature_names()) > 0

    def test_xgboost_prediction_from_features(self, xgb_predictor):
        """Test that XGBoost can make predictions from features."""
        features = TabularFeatureSet(
            {
                "sharp_retail_diff_home": 0.05,
                "sharp_retail_diff_away": -0.03,
                "market_hold_avg": 0.045,
                "hours_to_game": 2.0,
                "day_of_week": 3.0,
            }
        )

        prediction = xgb_predictor.predict(features)

        assert "home_win" in prediction.predictions
        assert "away_win" in prediction.predictions
        assert 0 <= prediction.confidence <= 1

    async def test_ml_strategy_with_xgboost(self, tabular_extractor, xgb_predictor, tmp_path):
        """Test MLBettingStrategy with XGBoost end-to-end."""
        strategy = MLBettingStrategy(
            feature_extractor=tabular_extractor,
            model_predictor=xgb_predictor,
            min_ev_threshold=0.03,
        )

        # Create test event
        event = BettingEvent(
            event_id="game1",
            commence_time=datetime(2024, 1, 15, 19, 0),
            home_competitor="Lakers",
            away_competitor="Warriors",
            home_score=110,
            away_score=105,
            status="final",
        )

        # Create odds snapshot
        odds_time = event.commence_time - timedelta(hours=2)
        odds_snapshot = [
            # Pinnacle (sharp)
            Odds(
                event_id="game1",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price=-110,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            Odds(
                event_id="game1",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Warriors",
                price=-110,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            # FanDuel (retail)
            Odds(
                event_id="game1",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name="Lakers",
                price=100,  # Better odds than sharp (potential +EV)
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            Odds(
                event_id="game1",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name="Warriors",
                price=-120,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
        ]

        # Create config
        config = BacktestConfig(
            initial_bankroll=1000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            decision_hours_before_game=1,
        )

        # Evaluate opportunities
        opportunities = await strategy.evaluate_opportunity(event, odds_snapshot, config)

        # Should return a list (might be empty depending on model predictions)
        assert isinstance(opportunities, list)
        # Each opportunity should be valid
        for opp in opportunities:
            assert isinstance(opp, BetOpportunity)
            assert opp.event_id == "game1"
            assert 0 <= opp.confidence <= 1


class TestLSTMPipeline:
    """Test complete pipeline with LSTM model."""

    @pytest.fixture
    def trained_lstm_model(self, tmp_path):
        """Create and save a trained PyTorch LSTM model."""

        # Use module-level SimpleLSTMModel class
        model = SimpleLSTMModel(input_size=3, hidden_size=16, output_size=2)
        model.eval()

        # Save model
        model_path = tmp_path / "lstm_model.pt"
        torch.save(model, str(model_path))

        return model_path

    @pytest.fixture
    def sequential_extractor(self):
        """Create sequential feature extractor."""
        return SequentialFeatureExtractor(
            feature_computers=[
                compute_sharp_retail_diff,
                compute_market_hold,
            ],
            sequence_length=5,
            step_size=timedelta(hours=1),
        )

    @pytest.fixture
    def lstm_predictor(self, trained_lstm_model):
        """Create LSTM predictor."""
        return LSTMPredictor(
            model_path=trained_lstm_model,
            output_names=["home_win", "away_win"],
        )

    def test_sequential_feature_extraction(self, sequential_extractor):
        """Test that sequential features can be extracted from odds."""
        from analytics.betting.observations import OddsObservation
        from analytics.betting.problems import BettingEvent

        # Create test event
        event_time = datetime(2024, 1, 15, 19, 0)
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Warriors",
            commence_time=event_time,
            home_score=None,
            away_score=None,
        )

        # Create observations at different times (one per outcome per bookmaker)
        observations = []
        for i in range(10):  # More than sequence_length
            obs_time = event_time - timedelta(hours=10 - i)
            # Pinnacle observations
            observations.extend(
                [
                    OddsObservation(
                        event_id="game1",
                        observation_time=obs_time,
                        bookmaker="pinnacle",
                        market="h2h",
                        outcome="Lakers",
                        odds=-110 + i,
                        line=None,
                    ),
                    OddsObservation(
                        event_id="game1",
                        observation_time=obs_time,
                        bookmaker="pinnacle",
                        market="h2h",
                        outcome="Warriors",
                        odds=-110 - i,
                        line=None,
                    ),
                ]
            )
            # FanDuel observations
            observations.extend(
                [
                    OddsObservation(
                        event_id="game1",
                        observation_time=obs_time,
                        bookmaker="fanduel",
                        market="h2h",
                        outcome="Lakers",
                        odds=-105 + i,
                        line=None,
                    ),
                    OddsObservation(
                        event_id="game1",
                        observation_time=obs_time,
                        bookmaker="fanduel",
                        market="h2h",
                        outcome="Warriors",
                        odds=-115 - i,
                        line=None,
                    ),
                ]
            )

        # Extract features
        decision_time = event_time - timedelta(hours=1)
        features = sequential_extractor.extract(event, observations, decision_time)

        # Verify features
        assert isinstance(features, SequentialFeatureSet)
        seq_len, n_features = features.get_shape()
        assert seq_len == 5  # sequence_length
        assert n_features > 0

    def test_lstm_prediction_from_features(self, lstm_predictor):
        """Test that LSTM can make predictions from sequential features."""

        sequences = np.random.rand(5, 3)  # 5 steps, 3 features
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(5)]

        features = SequentialFeatureSet(
            sequences=sequences,
            feature_names=["feat1", "feat2", "feat3"],
            timestamps=timestamps,
        )

        prediction = lstm_predictor.predict(features)

        assert "home_win" in prediction.predictions
        assert "away_win" in prediction.predictions
        assert 0 <= prediction.confidence <= 1

    async def test_ml_strategy_with_lstm(self, sequential_extractor, lstm_predictor, tmp_path):
        """Test MLBettingStrategy with LSTM end-to-end."""

        strategy = MLBettingStrategy(
            feature_extractor=sequential_extractor,
            model_predictor=lstm_predictor,
            min_ev_threshold=0.03,
        )

        # Create test event
        event = BettingEvent(
            event_id="game1",
            commence_time=datetime(2024, 1, 15, 19, 0),
            home_competitor="Lakers",
            away_competitor="Warriors",
            home_score=110,
            away_score=105,
            status="final",
        )

        # Create odds snapshot with multiple time points
        odds_snapshot = []
        for i in range(10):
            odds_time = event.commence_time - timedelta(hours=10 - i)
            # Pinnacle
            odds_snapshot.extend(
                [
                    Odds(
                        event_id="game1",
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=-110 + i,
                        point=None,
                        odds_timestamp=odds_time,
                        last_update=odds_time,
                    ),
                    Odds(
                        event_id="game1",
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Warriors",
                        price=-110 - i,
                        point=None,
                        odds_timestamp=odds_time,
                        last_update=odds_time,
                    ),
                ]
            )
            # FanDuel
            odds_snapshot.extend(
                [
                    Odds(
                        event_id="game1",
                        bookmaker_key="fanduel",
                        bookmaker_title="FanDuel",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=100 + i,
                        point=None,
                        odds_timestamp=odds_time,
                        last_update=odds_time,
                    ),
                    Odds(
                        event_id="game1",
                        bookmaker_key="fanduel",
                        bookmaker_title="FanDuel",
                        market_key="h2h",
                        outcome_name="Warriors",
                        price=-120 - i,
                        point=None,
                        odds_timestamp=odds_time,
                        last_update=odds_time,
                    ),
                ]
            )

        # Create config
        config = BacktestConfig(
            initial_bankroll=1000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            decision_hours_before_game=1,
        )

        # Evaluate opportunities
        opportunities = await strategy.evaluate_opportunity(event, odds_snapshot, config)

        # Should return a list
        assert isinstance(opportunities, list)
        # Each opportunity should be valid
        for opp in opportunities:
            assert isinstance(opp, BetOpportunity)
            assert opp.event_id == "game1"
            assert 0 <= opp.confidence <= 1


class TestMLStrategyValidation:
    """Test ML strategy validation and error handling."""

    def test_feature_type_mismatch_error(self, tmp_path):
        """Test that mismatched feature/model types raise error."""
        # Create XGBoost model (requires tabular)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        model = xgb.XGBClassifier(n_estimators=5)
        model.fit(X, y)

        model_path = tmp_path / "xgb.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        xgb_predictor = XGBoostPredictor(
            model_path=model_path,
            output_names=["home_win", "away_win"],
        )

        # Try to use with sequential extractor (mismatch!)
        sequential_extractor = SequentialFeatureExtractor(
            feature_computers=[compute_market_hold],
            sequence_length=5,
            step_size=timedelta(hours=1),
        )

        with pytest.raises(ValueError, match="doesn't match model requirements"):
            MLBettingStrategy(
                feature_extractor=sequential_extractor,
                model_predictor=xgb_predictor,
            )

    def test_confidence_thresholds(self, tmp_path):
        """Test that confidence thresholds filter predictions."""
        # Create model
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        model = xgb.XGBClassifier(n_estimators=5)
        model.fit(X, y)

        model_path = tmp_path / "xgb.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        xgb_predictor = XGBoostPredictor(
            model_path=model_path,
            output_names=["home_win", "away_win"],
        )

        tabular_extractor = TabularFeatureExtractor(feature_computers=[compute_market_hold])

        # Create strategy with tight confidence bounds
        strategy = MLBettingStrategy(
            feature_extractor=tabular_extractor,
            model_predictor=xgb_predictor,
            min_confidence_threshold=0.9,  # Very high threshold
            max_confidence_threshold=0.95,
        )

        # Verify thresholds are stored
        assert strategy.params["min_confidence_threshold"] == 0.9
        assert strategy.params["max_confidence_threshold"] == 0.95
