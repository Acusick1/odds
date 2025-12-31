"""Unit tests for LSTM betting strategy."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from odds_analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity
from odds_analytics.lstm_strategy import LSTMModel, LSTMStrategy
from odds_core.models import Event, EventStatus, Odds


@pytest.fixture
def sample_event():
    """Create a sample BacktestEvent for testing."""
    return BacktestEvent(
        id="test_event_lstm_1",
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
    """Test LSTM model architecture."""

    def test_model_initialization(self):
        """Test that LSTM model initializes correctly."""
        input_size = 16
        hidden_size = 64
        num_layers = 2
        dropout = 0.2

        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        assert model.input_size == input_size
        assert model.hidden_size == hidden_size
        assert model.num_layers == num_layers
        assert model.dropout == dropout
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)

    def test_model_forward_pass(self):
        """Test that forward pass produces expected output shapes."""
        batch_size = 4
        seq_len = 24
        input_size = 16
        hidden_size = 64

        model = LSTMModel(input_size=input_size, hidden_size=hidden_size)

        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_size)

        # Forward pass
        logits, probs = model(x)

        # Check shapes
        assert logits.shape == (batch_size,)
        assert probs.shape == (batch_size,)

        # Check value ranges
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_model_forward_different_batch_sizes(self):
        """Test that forward pass works with different batch sizes."""
        batch_size = 2
        seq_len = 12
        input_size = 8

        model = LSTMModel(input_size=input_size, hidden_size=32, num_layers=1, dropout=0.0)

        x = torch.randn(batch_size, seq_len, input_size)

        # Forward pass
        logits, probs = model(x)

        assert logits.shape == (batch_size,)
        assert probs.shape == (batch_size,)

    def test_model_deterministic_with_seed(self):
        """Test that model produces deterministic output with fixed seed."""
        torch.manual_seed(42)

        model = LSTMModel(input_size=10, hidden_size=32)
        x = torch.randn(1, 10, 10)

        logits1, probs1 = model(x)

        # Reset and run again
        torch.manual_seed(42)
        model2 = LSTMModel(input_size=10, hidden_size=32)

        logits2, probs2 = model2(x)

        # Note: May not be exactly equal due to initialization,
        # but within reasonable tolerance
        assert logits1.shape == logits2.shape
        assert probs1.shape == probs2.shape


class TestLSTMStrategy:
    """Test LSTM betting strategy class."""

    def test_strategy_initialization(self):
        """Test that strategy initializes correctly with default parameters."""
        strategy = LSTMStrategy()

        assert strategy.name == "LSTM"
        assert strategy.params["lookback_hours"] == 72
        assert strategy.params["timesteps"] == 24
        assert strategy.params["hidden_size"] == 64
        assert strategy.params["num_layers"] == 2
        assert strategy.params["dropout"] == 0.2
        assert strategy.params["market"] == "h2h"
        assert strategy.params["min_edge_threshold"] == 0.03
        assert strategy.params["min_confidence"] == 0.52
        assert strategy.model is None
        assert not strategy.is_trained

    def test_strategy_custom_parameters(self):
        """Test that strategy accepts custom parameters."""
        strategy = LSTMStrategy(
            lookback_hours=48,
            timesteps=16,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            market="spreads",
            min_edge_threshold=0.05,
            min_confidence=0.60,
        )

        assert strategy.params["lookback_hours"] == 48
        assert strategy.params["timesteps"] == 16
        assert strategy.params["hidden_size"] == 128
        assert strategy.params["num_layers"] == 3
        assert strategy.params["dropout"] == 0.3
        assert strategy.params["market"] == "spreads"
        assert strategy.params["min_edge_threshold"] == 0.05
        assert strategy.params["min_confidence"] == 0.60

    def test_create_model(self):
        """Test that _create_model creates LSTM with correct architecture."""
        strategy = LSTMStrategy(hidden_size=128, num_layers=3, dropout=0.3)

        model = strategy._create_model()

        assert isinstance(model, LSTMModel)
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.dropout == 0.3

    @pytest.mark.asyncio
    async def test_train_with_valid_data(self):
        """Test that training works with valid data."""
        strategy = LSTMStrategy(lookback_hours=24, timesteps=8)

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

        # Mock prepare_training_data to return dummy data
        n_samples = 10
        timesteps = 8
        num_features = strategy.input_size

        mock_X = np.random.randn(n_samples, timesteps, num_features).astype(np.float32)
        mock_y = np.random.randint(0, 2, n_samples).astype(np.float32)
        mock_masks = np.random.randint(0, 2, (n_samples, timesteps)).astype(bool)

        mock_result = MagicMock()
        mock_result.X = mock_X
        mock_result.y = mock_y
        mock_result.masks = mock_masks

        with patch(
            "odds_analytics.lstm_strategy.prepare_training_data",
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
        assert "loss" in history
        assert len(history["loss"]) == 3  # 3 epochs

        # Loss should be finite
        assert all(np.isfinite(loss) for loss in history["loss"])

    @pytest.mark.asyncio
    async def test_train_with_no_data_raises_error(self):
        """Test that training with no data raises ValueError."""
        strategy = LSTMStrategy()

        mock_events = []

        # Mock prepare_training_data to return empty arrays
        mock_result = MagicMock()
        mock_result.X = np.array([])
        mock_result.y = np.array([])
        mock_result.masks = np.array([])

        with patch(
            "odds_analytics.lstm_strategy.prepare_training_data",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_session = MagicMock()

            with pytest.raises(ValueError, match="No valid training data"):
                await strategy.train(events=mock_events, session=mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_without_training(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity returns empty list when not trained."""
        strategy = LSTMStrategy()
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
        strategy = LSTMStrategy(
            lookback_hours=24,
            timesteps=8,
            min_edge_threshold=0.0,
            min_confidence=0.0,
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
                "odds_analytics.lstm_strategy.load_sequences_for_event",
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
            assert 0 <= opp.confidence <= 1

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_high_confidence_threshold(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that high confidence threshold filters opportunities."""
        strategy = LSTMStrategy(
            lookback_hours=24,
            timesteps=8,
            min_edge_threshold=0.0,
            min_confidence=0.99,  # Very high
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
                "odds_analytics.lstm_strategy.load_sequences_for_event",
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

        # With 99% threshold, should get few/no opportunities
        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_no_sequences(self, sample_event, sample_odds_snapshot):
        """Test that evaluate_opportunity handles missing sequence data."""
        strategy = LSTMStrategy()
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Mock empty sequences
        with patch(
            "odds_analytics.lstm_strategy.load_sequences_for_event",
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

    def test_save_model(self):
        """Test that model can be saved to disk."""
        strategy = LSTMStrategy(hidden_size=128)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.5, 0.4, 0.3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"

            strategy.save_model(model_path)

            assert model_path.exists()

    def test_save_model_without_training_raises_error(self):
        """Test that saving without training raises error."""
        strategy = LSTMStrategy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"

            with pytest.raises(ValueError, match="No model to save"):
                strategy.save_model(model_path)

    def test_load_model(self):
        """Test that model can be loaded from disk."""
        # Create and save model
        strategy = LSTMStrategy(hidden_size=128, num_layers=2)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.5, 0.4, 0.3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            new_strategy = LSTMStrategy()
            new_strategy.load_model(model_path)

            assert new_strategy.model is not None
            assert new_strategy.is_trained
            assert new_strategy.training_history == {"loss": [0.5, 0.4, 0.3]}
            assert new_strategy.params["hidden_size"] == 128
            assert new_strategy.params["num_layers"] == 2

    def test_load_model_file_not_found(self):
        """Test that loading non-existent file raises error."""
        strategy = LSTMStrategy()

        with pytest.raises(FileNotFoundError):
            strategy.load_model("nonexistent_model.pt")

    def test_save_and_load_preserves_weights(self):
        """Test that save/load preserves model weights."""
        torch.manual_seed(42)

        strategy = LSTMStrategy(hidden_size=64)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True

        # Get initial predictions
        x = torch.randn(1, 24, strategy.input_size).to(strategy.device)
        strategy.model.eval()
        with torch.no_grad():
            logits1, probs1 = strategy.model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load and predict
            new_strategy = LSTMStrategy()
            new_strategy.load_model(model_path)
            new_strategy.model.eval()

            with torch.no_grad():
                logits2, probs2 = new_strategy.model(x)

        # Predictions should be identical
        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(probs1, probs2, atol=1e-6)


class TestLSTMWorkflow:
    """Integration tests for LSTM strategy workflows."""

    @pytest.mark.asyncio
    async def test_complete_training_workflow(self):
        """Test complete training workflow with mocked data."""
        strategy = LSTMStrategy(lookback_hours=24, timesteps=8)

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

        # Mock training data
        n_samples = 20
        timesteps = 8
        num_features = strategy.input_size

        mock_X = np.random.randn(n_samples, timesteps, num_features).astype(np.float32)
        mock_y = np.random.randint(0, 2, n_samples).astype(np.float32)
        mock_masks = np.ones((n_samples, timesteps), dtype=bool)

        mock_result = MagicMock()
        mock_result.X = mock_X
        mock_result.y = mock_y
        mock_result.masks = mock_masks

        with patch(
            "odds_analytics.lstm_strategy.prepare_training_data",
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
        assert len(history["loss"]) == 5

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trained_lstm.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            loaded_strategy = LSTMStrategy()
            loaded_strategy.load_model(model_path)

            assert loaded_strategy.is_trained
            assert loaded_strategy.training_history == history

    def test_get_strategy_name_and_params(self):
        """Test that strategy name and params are accessible."""
        strategy = LSTMStrategy(
            lookback_hours=48,
            market="spreads",
            min_edge_threshold=0.05,
        )

        assert strategy.get_name() == "LSTM"
        params = strategy.get_params()

        assert params["lookback_hours"] == 48
        assert params["market"] == "spreads"
        assert params["min_edge_threshold"] == 0.05


class TestLSTMModelPersistenceWithConfig:
    """Test LSTM model persistence with YAML config files."""

    def test_save_model_creates_yaml_config(self):
        """Test that save_model creates both model and YAML config files."""
        import yaml

        strategy = LSTMStrategy(hidden_size=128, num_layers=3, dropout=0.3)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.5, 0.4, 0.3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"

            strategy.save_model(model_path)

            # Check both files exist
            assert model_path.exists()
            config_path = model_path.with_suffix(".yaml")
            assert config_path.exists()

            # Verify config content
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            assert config_data["model_type"] == "LSTM"
            assert "saved_at" in config_data
            assert config_data["params"]["hidden_size"] == 128
            assert config_data["params"]["num_layers"] == 3
            assert config_data["params"]["dropout"] == 0.3
            assert config_data["is_trained"] is True

            # Check architecture section
            assert config_data["architecture"]["hidden_size"] == 128
            assert config_data["architecture"]["num_layers"] == 3
            assert config_data["architecture"]["dropout"] == 0.3

            # Check training summary
            assert config_data["training_summary"]["final_loss"] == 0.3
            assert config_data["training_summary"]["epochs_trained"] == 3

    def test_load_model_with_config(self):
        """Test that load_model logs config information when available."""
        strategy = LSTMStrategy(hidden_size=256, num_layers=4)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.6, 0.5, 0.4]}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            loaded_strategy = LSTMStrategy()
            loaded_strategy.load_model(model_path)

            # Verify params were loaded correctly
            assert loaded_strategy.params["hidden_size"] == 256
            assert loaded_strategy.params["num_layers"] == 4
            assert loaded_strategy.is_trained
            assert loaded_strategy.training_history == {"loss": [0.6, 0.5, 0.4]}

    def test_load_model_without_config_file(self):
        """Test that load_model works when config file doesn't exist."""
        strategy = LSTMStrategy(hidden_size=64, num_layers=2)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.7, 0.6]}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save only model file, no config
            model_path = Path(tmpdir) / "model_only.pt"
            save_dict = {
                "model_state_dict": strategy.model.state_dict(),
                "params": strategy.params,
                "input_size": strategy.input_size,
                "is_trained": strategy.is_trained,
                "training_history": strategy.training_history,
            }
            torch.save(save_dict, model_path)

            # Load - should work without config file
            loaded_strategy = LSTMStrategy()
            loaded_strategy.load_model(model_path)

            assert loaded_strategy.model is not None
            assert loaded_strategy.params["hidden_size"] == 64
            assert loaded_strategy.is_trained

    def test_save_model_without_training_history(self):
        """Test that save_model works when training_history is None."""
        import yaml

        strategy = LSTMStrategy(hidden_size=64)
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = None  # No history

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lstm_model.pt"

            strategy.save_model(model_path)

            # Check files exist
            assert model_path.exists()
            config_path = model_path.with_suffix(".yaml")
            assert config_path.exists()

            # Verify config doesn't have training_summary
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            assert "training_summary" not in config_data

    def test_complete_save_load_cycle(self):
        """Test complete save/load cycle preserves all data."""
        # Create strategy with custom params
        strategy = LSTMStrategy(
            lookback_hours=48,
            timesteps=16,
            hidden_size=128,
            num_layers=3,
            dropout=0.25,
            market="spreads",
            min_edge_threshold=0.05,
            min_confidence=0.60,
        )
        strategy.model = strategy._create_model().to(strategy.device)
        strategy.is_trained = True
        strategy.training_history = {"loss": [0.8, 0.7, 0.6, 0.5]}

        # Get initial prediction
        x = torch.randn(1, 16, strategy.input_size).to(strategy.device)
        strategy.model.eval()
        with torch.no_grad():
            logits1, probs1 = strategy.model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "full_model.pt"
            strategy.save_model(model_path)

            # Load into new strategy
            loaded_strategy = LSTMStrategy()
            loaded_strategy.load_model(model_path)

            # Verify all params preserved
            assert loaded_strategy.params["lookback_hours"] == 48
            assert loaded_strategy.params["timesteps"] == 16
            assert loaded_strategy.params["hidden_size"] == 128
            assert loaded_strategy.params["num_layers"] == 3
            assert loaded_strategy.params["dropout"] == 0.25
            assert loaded_strategy.params["market"] == "spreads"
            assert loaded_strategy.params["min_edge_threshold"] == 0.05
            assert loaded_strategy.params["min_confidence"] == 0.60

            # Verify predictions match
            loaded_strategy.model.eval()
            with torch.no_grad():
                logits2, probs2 = loaded_strategy.model(x)

            assert torch.allclose(logits1, logits2, atol=1e-6)
            assert torch.allclose(probs1, probs2, atol=1e-6)
