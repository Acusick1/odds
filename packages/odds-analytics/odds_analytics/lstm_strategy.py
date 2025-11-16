"""
LSTM-based betting strategy for sequence modeling of line movement.

This module provides a minimal, extensible LSTM framework for time-series betting models.
The architecture emphasizes clean integration and ease of modification over performance
optimization, enabling rapid experimentation with different model architectures.

Key Features:
- Simple 2-layer LSTM with 64 hidden units (easily customizable)
- Integrates with SequenceFeatureExtractor via dependency injection
- Returns training history for user analysis/visualization
- Supports model persistence (torch.save/load)
- Compatible with backtesting engine and Kelly Criterion sizing

Example:
    ```python
    from odds_analytics.lstm_strategy import LSTMStrategy
    from odds_analytics.backtesting import BacktestEngine, BacktestConfig
    from odds_lambda.storage.readers import OddsReader

    # Create and train strategy
    strategy = LSTMStrategy(
        lookback_hours=72,
        timesteps=24,
        hidden_size=64,
        num_layers=2,
        market="h2h",
        min_edge_threshold=0.03,
        min_confidence=0.52
    )

    # Train on historical data
    async with get_async_session() as session:
        reader = OddsReader(session)
        events = await reader.get_events_by_date_range(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            status=EventStatus.FINAL
        )

        history = await strategy.train(
            events=events,
            session=session,
            epochs=20,
            batch_size=32,
            learning_rate=0.001
        )

        # Save trained model
        strategy.save_model("models/lstm_h2h_v1.pt")

    # Use in backtesting
    config = BacktestConfig(
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 3, 31, tzinfo=UTC),
        initial_bankroll=10000.0,
        bet_sizing_method="fractional_kelly",
        kelly_fraction=0.25
    )

    async with get_async_session() as session:
        reader = OddsReader(session)
        engine = BacktestEngine(strategy, config, reader)
        result = await engine.run()
    ```
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
from odds_core.models import Event, Odds
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import (
    BacktestConfig,
    BacktestEvent,
    BetOpportunity,
    BettingStrategy,
)
from odds_analytics.feature_extraction import SequenceFeatureExtractor
from odds_analytics.sequence_loader import load_sequences_for_event, prepare_lstm_training_data

logger = structlog.get_logger()

__all__ = ["LSTMModel", "LSTMStrategy"]


class LSTMModel(nn.Module):
    """
    Simple 2-layer LSTM for binary outcome prediction.

    Architecture kept simple and readable for easy experimentation.
    Users can modify this class to test different architectures.

    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of hidden units in LSTM layers (default: 64)
        num_layers: Number of stacked LSTM layers (default: 2)
        dropout: Dropout rate between LSTM layers (default: 0.2)

    Input:
        x: Tensor of shape (batch_size, seq_len, input_size)
        mask: Optional attention mask of shape (batch_size, seq_len)

    Output:
        logits: Tensor of shape (batch_size,) - raw prediction scores
        probs: Tensor of shape (batch_size,) - probability of outcome=1
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output layer (binary classification)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            mask: Attention mask (batch_size, seq_len) - True for valid timesteps

        Returns:
            Tuple of (logits, probabilities):
            - logits: Raw prediction scores (batch_size,)
            - probs: Sigmoid probabilities (batch_size,)
        """
        # LSTM forward pass
        # output shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from last layer
        # h_n[-1] shape: (batch_size, hidden_size)
        final_hidden = h_n[-1]

        # Fully connected layer
        # logits shape: (batch_size, 1)
        logits = self.fc(final_hidden)

        # Squeeze to (batch_size,)
        logits = logits.squeeze(-1)

        # Apply sigmoid for probabilities
        probs = torch.sigmoid(logits)

        return logits, probs


class LSTMStrategy(BettingStrategy):
    """
    LSTM-based betting strategy using sequence modeling of line movement.

    This strategy trains an LSTM model on historical odds sequences to predict
    game outcomes, then identifies betting opportunities where the model's
    probability estimate differs from the market consensus.

    Design Philosophy:
    - Keep implementation simple and extensible
    - Return training history for user analysis
    - No advanced features (early stopping, LR scheduling) - users add as needed
    - Focus on clean integration with existing backtesting infrastructure

    Args:
        lookback_hours: Hours of historical data to use (default: 72)
        timesteps: Number of sequence timesteps (default: 24)
        hidden_size: LSTM hidden units (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.2)
        market: Market to analyze (default: "h2h")
        min_edge_threshold: Minimum edge to bet (default: 0.03 = 3%)
        min_confidence: Minimum model confidence to bet (default: 0.52)
        sharp_bookmakers: Sharp bookmakers for features (default: ["pinnacle"])
        retail_bookmakers: Retail bookmakers for features (default: ["fanduel", "draftkings", "betmgm"])
    """

    def __init__(
        self,
        lookback_hours: int = 72,
        timesteps: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        market: str = "h2h",
        min_edge_threshold: float = 0.03,
        min_confidence: float = 0.52,
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
    ):
        """Initialize LSTM betting strategy."""
        # Call parent constructor
        super().__init__(
            name="LSTM",
            lookback_hours=lookback_hours,
            timesteps=timesteps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            market=market,
            min_edge_threshold=min_edge_threshold,
            min_confidence=min_confidence,
            sharp_bookmakers=sharp_bookmakers or ["pinnacle"],
            retail_bookmakers=retail_bookmakers or ["fanduel", "draftkings", "betmgm"],
        )

        # Initialize feature extractor
        self.feature_extractor = SequenceFeatureExtractor(
            lookback_hours=lookback_hours,
            timesteps=timesteps,
            sharp_bookmakers=sharp_bookmakers,
            retail_bookmakers=retail_bookmakers,
        )

        # Get feature dimension
        self.input_size = len(self.feature_extractor.get_feature_names())

        # Initialize model (will be created on first training or load)
        self.model: LSTMModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training metadata
        self.is_trained = False
        self.training_history: dict[str, list[float]] | None = None

    def _create_model(self) -> LSTMModel:
        """Create new LSTM model with configured architecture."""
        return LSTMModel(
            input_size=self.input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        )

    async def train(
        self,
        events: list[Event],
        session: AsyncSession,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        outcome: str = "home",
    ) -> dict[str, list[float]]:
        """
        Train LSTM model on historical events.

        This is a basic training loop without early stopping or learning rate
        scheduling. Users can modify this method to add advanced features.

        Args:
            events: List of events with final scores (status=FINAL)
            session: Async database session
            epochs: Number of training epochs (default: 20)
            batch_size: Batch size for training (default: 32)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            outcome: What to predict - "home" or "away" (default: "home")

        Returns:
            Training history dict with "loss" key containing loss values per epoch

        Example:
            >>> history = await strategy.train(events, session, epochs=20)
            >>> print(f"Final loss: {history['loss'][-1]:.4f}")
        """
        logger.info(
            "training_lstm",
            num_events=len(events),
            epochs=epochs,
            batch_size=batch_size,
            outcome=outcome,
        )

        # Prepare training data
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=session,
            outcome=outcome,
            market=self.params["market"],
            lookback_hours=self.params["lookback_hours"],
            timesteps=self.params["timesteps"],
            sharp_bookmakers=self.params["sharp_bookmakers"],
            retail_bookmakers=self.params["retail_bookmakers"],
        )

        if len(X) == 0:
            logger.error("no_training_data")
            raise ValueError("No valid training data available")

        logger.info("training_data_prepared", num_samples=len(X), num_features=self.input_size)

        # Create model
        self.model = self._create_model().to(self.device)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        masks_tensor = torch.BoolTensor(masks).to(self.device)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, masks_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y, batch_mask in dataloader:
                optimizer.zero_grad()

                # Forward pass
                logits, probs = self.model(batch_X, batch_mask)

                # Calculate loss
                loss = criterion(logits, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Record average loss for epoch
            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)

            logger.info("training_epoch", epoch=epoch + 1, loss=avg_loss)

        self.is_trained = True
        self.training_history = history

        logger.info("training_complete", final_loss=history["loss"][-1])

        return history

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """
        Evaluate betting opportunities using trained LSTM model.

        This method is called by BacktestEngine for each event. It loads the
        historical odds sequence, runs inference through the LSTM model, and
        identifies betting opportunities where the model's probability estimate
        differs significantly from the market consensus.

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time (used for market odds comparison)
            config: Backtest configuration

        Returns:
            List of BetOpportunity objects (may be empty if no edge found)

        Edge Detection:
            If model predicts home win probability of 0.60 but market implies 0.52,
            the edge is 0.08 (8%). If this exceeds min_edge_threshold, a bet is placed.
        """
        if not self.is_trained or self.model is None:
            logger.warning("model_not_trained", event_id=event.id)
            return []

        # Need to load historical sequence for this event
        # Note: In production, we would get the session from somewhere
        # For now, this is a limitation - we can't easily get session in evaluate_opportunity
        # This is a design issue that needs to be addressed
        # For now, return empty list and log warning
        logger.warning(
            "lstm_evaluate_needs_session",
            event_id=event.id,
            message="LSTMStrategy.evaluate_opportunity requires database session to load sequences. "
            "Use LSTMStrategy.evaluate_opportunity_with_session instead.",
        )
        return []

    async def evaluate_opportunity_with_session(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
        session: AsyncSession,
    ) -> list[BetOpportunity]:
        """
        Evaluate betting opportunities with database session access.

        This is an extended version of evaluate_opportunity that accepts a session
        parameter to load historical odds sequences. This is necessary for LSTM
        models that require time-series data.

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time (used for market odds comparison)
            config: Backtest configuration
            session: Database session for loading historical sequences

        Returns:
            List of BetOpportunity objects (may be empty if no edge found)
        """
        if not self.is_trained or self.model is None:
            logger.warning("model_not_trained", event_id=event.id)
            return []

        opportunities = []

        # Load historical odds sequence for this event
        try:
            odds_sequences = await load_sequences_for_event(event.id, session)
        except Exception as e:
            logger.error("failed_to_load_sequences", event_id=event.id, error=str(e))
            return []

        if not odds_sequences or all(len(seq) == 0 for seq in odds_sequences):
            logger.debug("no_sequence_data", event_id=event.id)
            return []

        # Extract features for home team prediction
        try:
            features_dict = self.feature_extractor.extract_features(
                event=event,
                odds_data=odds_sequences,
                outcome=event.home_team,
                market=self.params["market"],
            )
        except Exception as e:
            logger.error("failed_to_extract_features", event_id=event.id, error=str(e))
            return []

        sequence = features_dict["sequence"]  # (timesteps, num_features)
        mask = features_dict["mask"]  # (timesteps,)

        # Skip if no valid timesteps
        if not mask.any():
            logger.debug("no_valid_timesteps", event_id=event.id)
            return []

        # Convert to PyTorch tensors and add batch dimension
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, timesteps, features)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)  # (1, timesteps)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(X, mask_tensor)
            home_win_prob = probs.item()

        # Calculate market consensus from odds snapshot
        market_odds = [
            o for o in odds_snapshot if o.market_key == self.params["market"] and o.outcome_name == event.home_team
        ]

        if not market_odds:
            logger.debug("no_market_odds", event_id=event.id)
            return []

        # Use average market implied probability as consensus
        from odds_analytics.utils import calculate_implied_probability

        market_probs = [calculate_implied_probability(o.price) for o in market_odds]
        market_consensus = float(np.mean(market_probs))

        # Calculate edge (model prob - market prob)
        edge = home_win_prob - market_consensus

        # Check if edge exceeds threshold and confidence is sufficient
        if edge >= self.params["min_edge_threshold"] and home_win_prob >= self.params["min_confidence"]:
            # Find best available odds for home team
            best_odds = max(market_odds, key=lambda o: o.price)

            opportunities.append(
                BetOpportunity(
                    event_id=event.id,
                    market=self.params["market"],
                    outcome=event.home_team,
                    bookmaker=best_odds.bookmaker_key,
                    odds=best_odds.price,
                    line=best_odds.point,
                    confidence=home_win_prob,  # Model probability used for Kelly sizing
                    rationale=f"LSTM edge: {edge:.2%} (model: {home_win_prob:.2%}, market: {market_consensus:.2%})",
                )
            )

        return opportunities

    def save_model(self, filepath: str | Path) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model weights and metadata

        Example:
            >>> strategy.save_model("models/lstm_h2h_v1.pt")
        """
        if self.model is None:
            raise ValueError("No model to save (train first)")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict and metadata
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "params": self.params,
            "input_size": self.input_size,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
        }

        torch.save(save_dict, filepath)
        logger.info("model_saved", filepath=str(filepath))

    def load_model(self, filepath: str | Path) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model file

        Example:
            >>> strategy = LSTMStrategy()
            >>> strategy.load_model("models/lstm_h2h_v1.pt")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore parameters (in case they differ from constructor args)
        self.params.update(checkpoint["params"])
        self.input_size = checkpoint["input_size"]

        # Create model with loaded architecture
        self.model = self._create_model().to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore metadata
        self.is_trained = checkpoint.get("is_trained", True)
        self.training_history = checkpoint.get("training_history")

        logger.info("model_loaded", filepath=str(filepath), is_trained=self.is_trained)
