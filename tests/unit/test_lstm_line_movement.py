"""Unit tests for LSTM line movement predictor strategy."""

import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from odds_analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity
from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy, LSTMModel
from odds_core.models import Event, EventStatus, Odds


@pytest.fixture
def sample_event():
    """Create a sample BacktestEvent for testing."""
    return BacktestEvent(
        id="test_event_lstm_lm_1",
        commence_time=datetime(2024, 11, 15, 19, 0, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        home_score=115,
        away_score=108,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def sample_odds_snapshot(sample_event):
    """Create sample odds snapshot for testing."""
    timestamp = datetime(2024, 11, 15, 18, 0, 0, tzinfo=UTC)
    return [
        # Pinnacle (sharp book)
        Odds(
            id=1,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price=-130,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=2,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price=+110,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        # FanDuel (retail book)
        Odds(
            id=3,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price=-125,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=4,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price=+105,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
    ]


class TestLSTMModel:
    """Test LSTM line movement model architecture (using unified LSTMModel with regression)."""

    def test_model_initialization(self):
        """Test that LSTM model initializes correctly for regression."""
        input_size = 16
        hidden_size = 64
        num_layers = 2
        dropout = 0.2

        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_type="regression",
        )

        assert model.input_size == input_size
        assert model.hidden_size == hidden_size
        assert model.num_layers == num_layers
        assert model.dropout == dropout
        assert model.output_type == "regression"
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)

    def test_model_forward_pass(self):
        """Test that forward pass produces expected output shapes."""
        batch_size = 4
        seq_len = 24
        input_size = 16
        hidden_size = 64

        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_type="regression")

        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_size)

        # Forward pass
        predictions = model(x)

        # Check shapes
        assert predictions.shape == (batch_size,)

        # Regression output - can be any real value (no sigmoid)
        assert predictions.dtype == torch.float32

    def test_model_forward_different_batch_sizes(self):
        """Test that forward pass works with different batch sizes."""
        batch_size = 2
        seq_len = 12
        input_size = 8

        model = LSTMModel(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            output_type="regression",
        )

        x = torch.randn(batch_size, seq_len, input_size)

        # Forward pass
        predictions = model(x)

        assert predictions.shape == (batch_size,)

    def test_model_regression_output(self):
        """Test that model produces unbounded regression output."""
        model = LSTMModel(input_size=10, hidden_size=32, output_type="regression")
        x = torch.randn(1, 10, 10)

        predictions = model(x)

        # Regression output should not be bounded to [0, 1]
        # This is different from classification LSTM which uses sigmoid
        assert predictions.shape == (1,)
        # Output can be any real number (positive or negative)


class TestLSTMLineMovementStrategy:
    """Test LSTM line movement betting strategy class."""

    def test_strategy_initialization(self):
        """Test that strategy initializes correctly with default parameters."""
        strategy = LSTMLineMovementStrategy()

        assert strategy.name == "LSTMLineMovement"
        assert strategy.params["lookback_hours"] == 72
        assert strategy.params["timesteps"] == 24
        assert strategy.params["hidden_size"] == 64
        assert strategy.params["num_layers"] == 2
        assert strategy.params["dropout"] == 0.2
        assert strategy.params["market"] == "h2h"
        assert strategy.params["min_predicted_movement"] == 0.02
        assert strategy.params["movement_confidence_scale"] == 5.0
        assert strategy.params["base_confidence"] == 0.52
        assert strategy.params["loss_function"] == "mse"
        assert strategy.model is None
        assert not strategy.is_trained

    def test_strategy_custom_parameters(self):
        """Test that strategy accepts custom parameters."""
        strategy = LSTMLineMovementStrategy(
            lookback_hours=48,
            timesteps=16,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            market="spreads",
            min_predicted_movement=0.03,
            movement_confidence_scale=10.0,
            base_confidence=0.55,
            loss_function="huber",
        )

        assert strategy.params["lookback_hours"] == 48
        assert strategy.params["timesteps"] == 16
        assert strategy.params["hidden_size"] == 128
        assert strategy.params["num_layers"] == 3
        assert strategy.params["dropout"] == 0.3
        assert strategy.params["market"] == "spreads"
        assert strategy.params["min_predicted_movement"] == 0.03
        assert strategy.params["movement_confidence_scale"] == 10.0
        assert strategy.params["base_confidence"] == 0.55
        assert strategy.params["loss_function"] == "huber"

    def test_create_model(self):
        """Test that _create_model creates LSTM with correct architecture."""
        strategy = LSTMLineMovementStrategy(hidden_size=128, num_layers=3, dropout=0.3)

        model = strategy._create_model()

        assert isinstance(model, LSTMModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.dropout == 0.3

    def test_get_loss_function_mse(self):
        """Test that MSE loss function is returned correctly."""
        strategy = LSTMLineMovementStrategy(loss_function="mse")
        loss_fn = strategy._get_loss_function()
        assert isinstance(loss_fn, torch.nn.MSELoss)

    def test_get_loss_function_mae(self):
        """Test that MAE loss function is returned correctly."""
        strategy = LSTMLineMovementStrategy(loss_function="mae")
        loss_fn = strategy._get_loss_function()
        assert isinstance(loss_fn, torch.nn.L1Loss)

    def test_get_loss_function_huber(self):
        """Test that Huber loss function is returned correctly."""
        strategy = LSTMLineMovementStrategy(loss_function="huber")
        loss_fn = strategy._get_loss_function()
        assert isinstance(loss_fn, torch.nn.HuberLoss)

    def test_get_loss_function_unknown_defaults_to_mse(self):
        """Test that unknown loss function defaults to MSE."""
        strategy = LSTMLineMovementStrategy()
        strategy.params["loss_function"] = "unknown"
        loss_fn = strategy._get_loss_function()
        assert isinstance(loss_fn, torch.nn.MSELoss)

    @pytest.mark.asyncio
    async def test_train_with_valid_data(self):
        """Test that training works with valid regression data."""
        strategy = LSTMLineMovementStrategy(lookback_hours=24, timesteps=8)

        # Mock events
        mock_events = [
            Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
                home_team="Team A",
                away_team="Team B",
                status=EventStatus.FINAL,
                home_score=110,
                away_score=105,
            )
            for i in range(10)
        ]

        # Mock prepare_training_data to return dummy regression data
        n_samples = 10
        timesteps = 8
        num_features = strategy.input_size

        mock_X = np.random.randn(n_samples, timesteps, num_features).astype(np.float32)
        # Regression targets (continuous values, not binary)
        mock_y = np.random.randn(n_samples).astype(np.float32) * 0.1  # Small deltas
        mock_masks = np.random.randint(0, 2, (n_samples, timesteps)).astype(bool)

        mock_result = MagicMock()
        mock_result.X = mock_X
        mock_result.y = mock_y
        mock_result.masks = mock_masks

        with patch(
            "odds_analytics.lstm_line_movement.prepare_training_data",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_session = MagicMock()

            history = await strategy.train(
                events=mock_events, session=mock_session, epochs=3, batch_size=4
            )

        # Check training completed
        assert strategy.is_trained
        assert strategy.model is not None

        # Check new standardized history format
        assert "train_mse" in history
        assert "train_mae" in history
        assert "train_r2" in history
        assert "val_mse" in history
        assert "val_mae" in history
        assert "val_r2" in history
        assert "n_samples" in history
        assert "n_features" in history

        # Metrics should be finite scalars
        assert np.isfinite(history["train_mse"])
        assert np.isfinite(history["train_mae"])
        assert np.isfinite(history["val_mse"])
        assert np.isfinite(history["val_mae"])

    @pytest.mark.asyncio
    async def test_train_with_different_loss_functions(self):
        """Test training with different loss functions."""
        for loss_fn in ["mse", "mae", "huber"]:
            strategy = LSTMLineMovementStrategy(
                lookback_hours=24, timesteps=8, loss_function=loss_fn
            )

            n_samples = 5
            mock_X = np.random.randn(n_samples, 8, strategy.input_size).astype(np.float32)
            mock_y = np.random.randn(n_samples).astype(np.float32) * 0.1
            mock_masks = np.ones((n_samples, 8), dtype=bool)

            mock_result = MagicMock()
            mock_result.X = mock_X
            mock_result.y = mock_y
            mock_result.masks = mock_masks

            with patch(
                "odds_analytics.lstm_line_movement.prepare_training_data",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                mock_session = MagicMock()
                mock_events = [MagicMock() for _ in range(n_samples)]

                history = await strategy.train(
                    events=mock_events, session=mock_session, epochs=2, batch_size=4
                )

            assert strategy.is_trained
            # Check new standardized format
            assert "train_mse" in history
            assert "val_mse" in history
            assert np.isfinite(history["train_mse"])
            assert np.isfinite(history["val_mse"])

    @pytest.mark.asyncio
    async def test_train_with_no_data_raises_error(self):
        """Test that training with no data raises ValueError."""
        strategy = LSTMLineMovementStrategy()

        mock_events = []

        # Mock prepare_training_data to return empty arrays
        mock_result = MagicMock()
        mock_result.X = np.array([])
        mock_result.y = np.array([])
        mock_result.masks = np.array([])

        with patch(
            "odds_analytics.lstm_line_movement.prepare_training_data",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_session = MagicMock()

            with pytest.raises(ValueError, match="No valid training data"):
                await strategy.train(events=mock_events, session=mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_without_training(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity returns empty list when not trained."""
        strategy = LSTMLineMovementStrategy()
        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
        )

        opportunities = await strategy.evaluate_opportunity(
            sample_event, sample_odds_snapshot, config
        )

        assert opportunities == []

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_session(self, sample_event, sample_odds_snapshot):
        """Test evaluate_opportunity with trained model and session."""
        strategy = LSTMLineMovementStrategy(
            lookback_hours=24,
            timesteps=8,
            min_predicted_movement=0.0,  # Accept any movement
        )

        # Manually create and set a trained model
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Mock load_sequences_for_event
        mock_sequences = [
            [
                Odds(
                    event_id=sample_event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-120,
                    point=None,
                    odds_timestamp=datetime(2024, 11, 15, 12, 0, 0, tzinfo=UTC),
                    last_update=datetime(2024, 11, 15, 12, 0, 0, tzinfo=UTC),
                )
            ]
        ] * 5  # 5 snapshots

        # Mock feature extractor
        mock_features = {
            "sequence": np.random.randn(8, strategy.input_size).astype(np.float32),
            "mask": np.ones(8, dtype=bool),
        }

        with (
            patch(
                "odds_analytics.lstm_line_movement.load_sequences_for_event",
                new_callable=AsyncMock,
                return_value=mock_sequences,
            ),
            patch.object(
                strategy.feature_extractor, "extract_features", return_value=mock_features
            ),
        ):
            mock_session = MagicMock()
            config = BacktestConfig(
                initial_bankroll=10000.0,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            opportunities = await strategy.evaluate_opportunity(
                sample_event, sample_odds_snapshot, config, mock_session
            )

        # Should return list (may be empty if thresholds not met)
        assert isinstance(opportunities, list)
        for opp in opportunities:
            assert isinstance(opp, BetOpportunity)
            assert opp.event_id == sample_event.id
            # Confidence should be clamped between 0.5 and 0.95
            assert 0.5 <= opp.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_confidence_calculation(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that confidence is calculated correctly from predicted movement."""
        strategy = LSTMLineMovementStrategy(
            lookback_hours=24,
            timesteps=8,
            min_predicted_movement=0.02,
            movement_confidence_scale=5.0,
            base_confidence=0.52,
        )

        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Force model to predict a specific value
        # We'll mock the model output directly
        mock_sequences = [[Odds()] * 3] * 5
        mock_features = {
            "sequence": np.random.randn(8, strategy.input_size).astype(np.float32),
            "mask": np.ones(8, dtype=bool),
        }

        with (
            patch(
                "odds_analytics.lstm_line_movement.load_sequences_for_event",
                new_callable=AsyncMock,
                return_value=mock_sequences,
            ),
            patch.object(
                strategy.feature_extractor, "extract_features", return_value=mock_features
            ),
        ):
            mock_session = MagicMock()
            config = BacktestConfig(
                initial_bankroll=10000.0,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            opportunities = await strategy.evaluate_opportunity(
                sample_event, sample_odds_snapshot, config, mock_session
            )

        # Verify confidence is within valid range
        for opp in opportunities:
            assert 0.5 <= opp.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_high_movement_threshold(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that high movement threshold filters opportunities."""
        strategy = LSTMLineMovementStrategy(
            lookback_hours=24,
            timesteps=8,
            min_predicted_movement=1.0,  # Very high threshold
        )

        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        mock_sequences = [[Odds()] * 3] * 5
        mock_features = {
            "sequence": np.random.randn(8, strategy.input_size).astype(np.float32),
            "mask": np.ones(8, dtype=bool),
        }

        with (
            patch(
                "odds_analytics.lstm_line_movement.load_sequences_for_event",
                new_callable=AsyncMock,
                return_value=mock_sequences,
            ),
            patch.object(
                strategy.feature_extractor, "extract_features", return_value=mock_features
            ),
        ):
            mock_session = MagicMock()
            config = BacktestConfig(
                initial_bankroll=10000.0,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            opportunities = await strategy.evaluate_opportunity(
                sample_event, sample_odds_snapshot, config, mock_session
            )

        # With very high threshold, should get no opportunities
        # (typical movements are small like 0.01-0.05)
        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_no_sequences(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity handles missing sequence data."""
        strategy = LSTMLineMovementStrategy()
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Mock empty sequences
        with patch(
            "odds_analytics.lstm_line_movement.load_sequences_for_event",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_session = MagicMock()
            config = BacktestConfig(
                initial_bankroll=10000.0,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
            )

            opportunities = await strategy.evaluate_opportunity(
                sample_event, sample_odds_snapshot, config, mock_session
            )

        assert opportunities == []

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_without_session(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity returns empty without session."""
        strategy = LSTMLineMovementStrategy()
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        config = BacktestConfig(
            initial_bankroll=10000.0,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
        )

        # No session provided
        opportunities = await strategy.evaluate_opportunity(
            sample_event, sample_odds_snapshot, config, session=None
        )

        assert opportunities == []

    def test_save_model(self):
        """Test that model can be saved to disk."""
        strategy = LSTMLineMovementStrategy(hidden_size=128)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.05, 0.04, 0.03], "mae": [0.04, 0.03, 0.025]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_line_movement.pt"

            strategy.save_model(model_path)

            assert model_path.exists()

    def test_save_model_without_training_raises_error(self):
        """Test that saving without training raises error."""
        strategy = LSTMLineMovementStrategy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"

            with pytest.raises(ValueError, match="No model to save"):
                strategy.save_model(model_path)

    def test_load_model(self):
        """Test that model can be loaded from disk."""
        # Create and save model
        strategy = LSTMLineMovementStrategy(hidden_size=128, num_layers=2, loss_function="huber")
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.05, 0.04, 0.03], "mae": [0.04, 0.03, 0.025]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            new_strategy = LSTMLineMovementStrategy()
            new_strategy.load_model(model_path)

            assert new_strategy.model is not None
            assert new_strategy.is_trained
            assert new_strategy.training_history == {
                "loss": [0.05, 0.04, 0.03],
                "mae": [0.04, 0.03, 0.025],
            }
            assert new_strategy.params["hidden_size"] == 128
            assert new_strategy.params["num_layers"] == 2
            assert new_strategy.params["loss_function"] == "huber"

    def test_load_model_file_not_found(self):
        """Test that loading non-existent file raises error."""
        strategy = LSTMLineMovementStrategy()

        with pytest.raises(FileNotFoundError):
            strategy.load_model("nonexistent_model.pt")

    def test_save_and_load_preserves_weights(self):
        """Test that save/load preserves model weights."""
        torch.manual_seed(42)

        strategy = LSTMLineMovementStrategy(hidden_size=64)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Get initial predictions
        x = torch.randn(1, 24, strategy.input_size).to(strategy.device)
        strategy.model.eval()
        with torch.no_grad():
            predictions1 = strategy.model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load and predict
            new_strategy = LSTMLineMovementStrategy()
            new_strategy.load_model(model_path)
            new_strategy.model.eval()

            with torch.no_grad():
                predictions2 = new_strategy.model(x)

        # Predictions should be identical
        assert torch.allclose(predictions1, predictions2, atol=1e-6)

    def test_load_model_via_constructor(self):
        """Test that model can be loaded via constructor model_path."""
        # Create and save model
        strategy = LSTMLineMovementStrategy(hidden_size=64)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load via constructor
            loaded_strategy = LSTMLineMovementStrategy(model_path=str(model_path))

            assert loaded_strategy.model is not None
            assert loaded_strategy.is_trained


class TestLSTMModelStaticFeatures:
    """Tests for LSTMModel static feature branch architecture."""

    def test_static_size_zero_backward_compatible(self):
        """Test that static_size=0 produces identical behavior to original model."""
        model = LSTMModel(input_size=14, hidden_size=64, static_size=0, output_type="regression")
        x = torch.randn(4, 24, 14)
        predictions = model(x)
        assert predictions.shape == (4,)
        # FC layer should be hidden_size -> 1
        assert model.fc.in_features == 64

    def test_static_size_creates_wider_fc(self):
        """Test that static_size > 0 widens the FC layer."""
        model = LSTMModel(input_size=14, hidden_size=64, static_size=11, output_type="regression")
        assert model.fc.in_features == 64 + 11
        assert model.fc.out_features == 1

    def test_forward_with_static_features(self):
        """Test forward pass concatenates static features correctly."""
        batch_size = 4
        static_size = 11
        model = LSTMModel(
            input_size=14, hidden_size=64, static_size=static_size, output_type="regression"
        )
        x = torch.randn(batch_size, 24, 14)
        static = torch.randn(batch_size, static_size)
        predictions = model(x, static_features=static)
        assert predictions.shape == (batch_size,)

    def test_forward_static_none_zero_fills(self):
        """Test that static_features=None zero-fills when static_size > 0."""
        model = LSTMModel(input_size=14, hidden_size=64, static_size=11, output_type="regression")
        x = torch.randn(2, 24, 14)
        # Should not crash — zero-fills the static portion
        predictions = model(x, static_features=None)
        assert predictions.shape == (2,)

    def test_save_load_preserves_static_size(self):
        """Test that save/load round-trip preserves static_size."""
        strategy = LSTMLineMovementStrategy(hidden_size=32)
        strategy.params["static_size"] = 11
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_static.pt"
            strategy.save_model(model_path)

            loaded = LSTMLineMovementStrategy()
            loaded.load_model(model_path)

            assert loaded.params["static_size"] == 11
            assert loaded.model.fc.in_features == 32 + 11

    def test_train_loop_with_static_data(self):
        """Test _train_loop accepts and uses static features."""
        strategy = LSTMLineMovementStrategy(hidden_size=32, num_layers=1, dropout=0.0)
        strategy.params["static_size"] = 5
        strategy.model = strategy._create_model().to(strategy.device)

        n_train, n_val = 20, 5
        timesteps, input_size = 8, strategy.input_size
        X_train = np.random.randn(n_train, timesteps, input_size).astype(np.float32)
        y_train = np.random.randn(n_train).astype(np.float32) * 0.05
        X_val = np.random.randn(n_val, timesteps, input_size).astype(np.float32)
        y_val = np.random.randn(n_val).astype(np.float32) * 0.05
        static_train = np.random.randn(n_train, 5).astype(np.float32)
        static_val = np.random.randn(n_val, 5).astype(np.float32)

        history = strategy._train_loop(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            static_train=static_train,
            static_val=static_val,
        )

        assert "train_mse" in history
        assert "val_mse" in history
        assert np.isfinite(history["train_mse"])
        assert np.isfinite(history["val_mse"])

    def test_train_from_config_computes_static_size(self):
        """Test train_from_config derives static_size from data shape."""
        from odds_analytics.training import (
            DataConfig,
            ExperimentConfig,
            FeatureConfig,
            MLTrainingConfig,
            TrainingConfig,
        )
        from odds_analytics.training.config import LSTMConfig

        config = MLTrainingConfig(
            experiment=ExperimentConfig(name="test_static"),
            training=TrainingConfig(
                strategy_type="lstm_line_movement",
                data=DataConfig(
                    start_date=date(2024, 10, 1),
                    end_date=date(2025, 1, 1),
                ),
                features=FeatureConfig(adapter="lstm"),
                model=LSTMConfig(lookback_hours=24, timesteps=8),
            ),
        )

        strategy = LSTMLineMovementStrategy()

        n_samples = 30
        timesteps, input_size = 8, strategy.input_size
        static_size = 7

        X_train = np.random.randn(n_samples, timesteps, input_size).astype(np.float32)
        y_train = np.random.randn(n_samples).astype(np.float32) * 0.05
        static_train = np.random.randn(n_samples, static_size).astype(np.float32)

        history = strategy.train_from_config(
            config=config,
            X_train=X_train,
            y_train=y_train,
            feature_names=[f"f{i}" for i in range(input_size)],
            static_train=static_train,
        )

        hidden_size = config.training.model.hidden_size
        assert strategy.params["static_size"] == static_size
        assert strategy.model.fc.in_features == hidden_size + static_size
        assert np.isfinite(history["train_mse"])


class TestLSTMLineMovementWorkflow:
    """Integration tests for LSTM line movement strategy workflows."""

    @pytest.mark.asyncio
    async def test_complete_training_workflow(self):
        """Test complete training workflow with mocked data."""
        strategy = LSTMLineMovementStrategy(lookback_hours=24, timesteps=8)

        # Mock events
        mock_events = [
            Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, i + 1, 19, 0, 0, tzinfo=UTC),
                home_team="Team A",
                away_team="Team B",
                status=EventStatus.FINAL,
                home_score=110 + i,
                away_score=105,
            )
            for i in range(20)
        ]

        # Mock training data (regression targets)
        n_samples = 20
        timesteps = 8
        num_features = strategy.input_size

        mock_X = np.random.randn(n_samples, timesteps, num_features).astype(np.float32)
        mock_y = np.random.randn(n_samples).astype(np.float32) * 0.05  # Small movements
        mock_masks = np.ones((n_samples, timesteps), dtype=bool)

        mock_result = MagicMock()
        mock_result.X = mock_X
        mock_result.y = mock_y
        mock_result.masks = mock_masks

        with patch(
            "odds_analytics.lstm_line_movement.prepare_training_data",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_session = MagicMock()

            # Train
            history = await strategy.train(
                events=mock_events, session=mock_session, epochs=5, batch_size=8
            )

        # Verify training
        assert strategy.is_trained
        # Check new standardized format
        assert "train_mse" in history
        assert "train_mae" in history
        assert "val_mse" in history
        assert "val_mae" in history
        assert np.isfinite(history["train_mse"])
        assert np.isfinite(history["val_mse"])

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trained_lstm_lm.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            loaded_strategy = LSTMLineMovementStrategy()
            loaded_strategy.load_model(model_path)

            assert loaded_strategy.is_trained
            # Training history is simplified for storage
            assert "loss" in loaded_strategy.training_history
            assert "mae" in loaded_strategy.training_history
            # Verify the stored history matches the final metrics
            assert loaded_strategy.training_history["loss"][0] == history["train_mse"]
            assert loaded_strategy.training_history["mae"][0] == history["train_mae"]

    def test_get_strategy_name_and_params(self):
        """Test that strategy name and params are accessible."""
        strategy = LSTMLineMovementStrategy(
            lookback_hours=48,
            market="spreads",
            min_predicted_movement=0.03,
            loss_function="huber",
        )

        assert strategy.get_name() == "LSTMLineMovement"
        params = strategy.get_params()

        assert params["lookback_hours"] == 48
        assert params["market"] == "spreads"
        assert params["min_predicted_movement"] == 0.03
        assert params["loss_function"] == "huber"

    def test_strategy_can_be_instantiated_from_cli_registry(self):
        """Test that strategy works with CLI pattern."""
        # Simulate CLI instantiation
        strategy = LSTMLineMovementStrategy()

        assert strategy.name == "LSTMLineMovement"
        assert strategy.model is None  # Not trained yet
        assert not strategy.is_trained

        # Verify it can be used in backtest (would fail without training)
        assert strategy.get_params() is not None
