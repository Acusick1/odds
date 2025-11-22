"""
LSTM-based line movement predictor for regression-based betting.

This module provides an LSTM model that predicts line movement deltas (closing - opening)
using time-series sequence data. Unlike classification-based approaches, this strategy
predicts continuous values representing how betting lines are expected to move.

Key Features:
- LSTM regressor for predicting continuous line movement deltas
- Multiple loss function support (MSE, MAE, Huber)
- Uses SequenceFeatureExtractor for sequential feature handling
- Converts predicted movements to betting confidence for Kelly sizing
- Model persistence (save/load weights)
- Integration with backtesting framework

The core insight is that if a line is predicted to move significantly in one
direction (e.g., probability increasing for home team), it suggests the opening
line undervalues that outcome, creating a potential +EV betting opportunity.

Example:
    ```python
    from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy
    from odds_analytics.backtesting import BacktestEngine, BacktestConfig
    from odds_lambda.storage.readers import OddsReader

    # Create and train strategy
    strategy = LSTMLineMovementStrategy(
        lookback_hours=72,
        timesteps=24,
        hidden_size=64,
        num_layers=2,
        market="h2h",
        min_predicted_movement=0.02,
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
        )

        # Save trained model
        strategy.save_model("models/lstm_line_movement_v1.pt")

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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
import torch.nn as nn
from odds_core.models import Event, Odds
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from odds_analytics.training.config import MLTrainingConfig

from odds_analytics.backtesting import (
    BacktestConfig,
    BacktestEvent,
    BetOpportunity,
    BettingStrategy,
)
from odds_analytics.feature_extraction import SequenceFeatureExtractor
from odds_analytics.lstm_strategy import LSTMModel
from odds_analytics.sequence_loader import (
    TargetType,
    load_sequences_for_event,
    prepare_lstm_training_data,
)

logger = structlog.get_logger()

__all__ = ["LSTMLineMovementStrategy"]


class LSTMLineMovementStrategy(BettingStrategy):
    """
    LSTM-based betting strategy for line movement regression prediction.

    This strategy trains an LSTM model on historical odds sequences to predict
    how betting lines will move from opening to closing. It then identifies
    betting opportunities where significant favorable movement is predicted.

    The model predicts:
    - For h2h: Probability delta (closing_prob - opening_prob)
    - For spreads/totals: Point delta (closing_point - opening_point)

    A positive predicted delta for an outcome indicates the market is expected
    to move in favor of that outcome, suggesting it's currently undervalued.

    Args:
        model_path: Path to saved model file (loads on init if provided)
        lookback_hours: Hours of historical data to use (default: 72)
        timesteps: Number of sequence timesteps (default: 24)
        hidden_size: LSTM hidden units (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.2)
        market: Market to analyze (default: "h2h")
        min_predicted_movement: Minimum predicted movement to trigger bet (default: 0.02)
        movement_confidence_scale: Scale factor for confidence (default: 5.0)
        base_confidence: Base confidence at threshold (default: 0.52)
        loss_function: Loss function to use (default: "mse", options: "mse", "mae", "huber")
        sharp_bookmakers: Sharp bookmakers for features (default: ["pinnacle"])
        retail_bookmakers: Retail bookmakers for features (default: ["fanduel", "draftkings", "betmgm"])
    """

    def __init__(
        self,
        model_path: str | None = None,
        lookback_hours: int = 72,
        timesteps: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        market: str = "h2h",
        min_predicted_movement: float = 0.02,
        movement_confidence_scale: float = 5.0,
        base_confidence: float = 0.52,
        loss_function: str = "mse",
        sharp_bookmakers: list[str] | None = None,
        retail_bookmakers: list[str] | None = None,
    ):
        """Initialize LSTM Line Movement betting strategy."""
        # Call parent constructor
        super().__init__(
            name="LSTMLineMovement",
            model_path=model_path,
            lookback_hours=lookback_hours,
            timesteps=timesteps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            market=market,
            min_predicted_movement=min_predicted_movement,
            movement_confidence_scale=movement_confidence_scale,
            base_confidence=base_confidence,
            loss_function=loss_function,
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

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def _create_model(self) -> LSTMModel:
        """Create new LSTM model with configured architecture for regression."""
        return LSTMModel(
            input_size=self.input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
            output_type="regression",
        )

    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on configuration."""
        loss_name = self.params.get("loss_function", "mse").lower()

        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.HuberLoss()
        else:
            logger.warning(f"Unknown loss function '{loss_name}', defaulting to MSE")
            return nn.MSELoss()

    async def train(
        self,
        events: list[Event],
        session: AsyncSession,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        outcome: str = "home",
        opening_hours_before: float = 48.0,
        closing_hours_before: float = 0.5,
    ) -> dict[str, list[float]]:
        """
        Train LSTM model on historical events for line movement regression.

        Args:
            events: List of events with final scores (status=FINAL)
            session: Async database session
            epochs: Number of training epochs (default: 20)
            batch_size: Batch size for training (default: 32)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            outcome: What to predict - "home" or "away" (default: "home")
            opening_hours_before: Hours before game for opening line (default: 48)
            closing_hours_before: Hours before game for closing line (default: 0.5)

        Returns:
            Training history dict with loss values per epoch

        Example:
            >>> history = await strategy.train(events, session, epochs=20)
            >>> print(f"Final loss: {history['loss'][-1]:.4f}")
        """
        logger.info(
            "training_lstm_line_movement",
            num_events=len(events),
            epochs=epochs,
            batch_size=batch_size,
            outcome=outcome,
            loss_function=self.params.get("loss_function", "mse"),
        )

        # Prepare training data with regression targets
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=session,
            outcome=outcome,
            market=self.params["market"],
            lookback_hours=self.params["lookback_hours"],
            timesteps=self.params["timesteps"],
            sharp_bookmakers=self.params["sharp_bookmakers"],
            retail_bookmakers=self.params["retail_bookmakers"],
            target_type=TargetType.REGRESSION,
            opening_hours_before=opening_hours_before,
            closing_hours_before=closing_hours_before,
        )

        if len(X) == 0:
            logger.error("no_training_data")
            raise ValueError("No valid training data available")

        logger.info(
            "training_data_prepared",
            num_samples=len(X),
            num_features=self.input_size,
            target_mean=float(np.mean(y)),
            target_std=float(np.std(y)),
        )

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
        criterion = self._get_loss_function()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        history: dict[str, list[float]] = {"loss": [], "mae": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0

            for batch_X, batch_y, _batch_mask in dataloader:
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_X)

                # Calculate loss
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Calculate MAE for tracking regardless of loss function
                with torch.no_grad():
                    mae = torch.mean(torch.abs(predictions - batch_y)).item()
                    epoch_mae += mae
                num_batches += 1

            # Record average metrics for epoch
            avg_loss = epoch_loss / num_batches
            avg_mae = epoch_mae / num_batches
            history["loss"].append(avg_loss)
            history["mae"].append(avg_mae)

            logger.info("training_epoch", epoch=epoch + 1, loss=avg_loss, mae=avg_mae)

        self.is_trained = True
        self.training_history = history

        logger.info(
            "training_complete",
            final_loss=history["loss"][-1],
            final_mae=history["mae"][-1],
        )

        return history

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
        session: AsyncSession | None = None,
    ) -> list[BetOpportunity]:
        """
        Evaluate betting opportunities using line movement predictions.

        The strategy:
        1. Load historical odds sequence for the event
        2. Extract features for both home and away outcomes
        3. Predict line movement delta for each
        4. Bet on outcomes with significant predicted positive movement

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time (opening line)
            config: Backtest configuration
            session: Database session for loading historical sequences (required)

        Returns:
            List of BetOpportunity objects (may be empty if no edge found)
        """
        if not self.is_trained or self.model is None:
            logger.warning("model_not_trained", event_id=event.id)
            return []

        # LSTM requires database session to load historical sequences
        if session is None:
            logger.warning(
                "lstm_requires_session",
                event_id=event.id,
                message="LSTMLineMovementStrategy requires database session to load sequences",
            )
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

        market = self.params["market"]
        min_movement = self.params["min_predicted_movement"]
        movement_scale = self.params["movement_confidence_scale"]
        base_conf = self.params["base_confidence"]

        # Evaluate both home and away team
        for outcome_name in [event.home_team, event.away_team]:
            # Extract features for this outcome
            try:
                features_dict = self.feature_extractor.extract_features(
                    event=event,
                    odds_data=odds_sequences,
                    outcome=outcome_name,
                    market=market,
                )
            except Exception as e:
                logger.error(
                    "failed_to_extract_features",
                    event_id=event.id,
                    outcome=outcome_name,
                    error=str(e),
                )
                continue

            sequence = features_dict["sequence"]  # (timesteps, num_features)
            mask = features_dict["mask"]  # (timesteps,)

            # Skip if no valid timesteps
            if not mask.any():
                logger.debug("no_valid_timesteps", event_id=event.id, outcome=outcome_name)
                continue

            # Convert to PyTorch tensors and add batch dimension
            X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Run inference
            self.model.eval()
            with torch.no_grad():
                predicted_movement = self.model(X).item()

            # Only bet if significant positive movement is predicted
            # Positive movement = line expected to move in favor of this outcome
            if predicted_movement < min_movement:
                continue

            # Find best available odds for this outcome in the snapshot
            outcome_odds = [
                o
                for o in odds_snapshot
                if o.market_key == market and o.outcome_name == outcome_name
            ]

            if not outcome_odds:
                continue

            best_odd = max(outcome_odds, key=lambda o: o.price)

            # Convert predicted movement to confidence for Kelly sizing
            # Higher predicted movement = higher confidence
            confidence = base_conf + (predicted_movement - min_movement) * movement_scale
            confidence = min(max(confidence, 0.5), 0.95)  # Clamp to reasonable range

            opportunities.append(
                BetOpportunity(
                    event_id=event.id,
                    market=market,
                    outcome=outcome_name,
                    bookmaker=best_odd.bookmaker_key,
                    odds=best_odd.price,
                    line=best_odd.point,
                    confidence=confidence,
                    rationale=f"LSTM predicted movement: {predicted_movement:+.3f} "
                    f"({'prob' if market == 'h2h' else 'points'}) "
                    f"at {best_odd.bookmaker_key}",
                )
            )

        return opportunities

    def save_model(self, filepath: str | Path) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model weights and metadata

        Example:
            >>> strategy.save_model("models/lstm_line_movement_v1.pt")
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
            >>> strategy = LSTMLineMovementStrategy()
            >>> strategy.load_model("models/lstm_line_movement_v1.pt")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load saved data
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Restore parameters (in case they differ from constructor args)
        self.params.update(checkpoint["params"])
        self.input_size = checkpoint["input_size"]

        # Recreate feature extractor with loaded params
        self.feature_extractor = SequenceFeatureExtractor(
            lookback_hours=self.params["lookback_hours"],
            timesteps=self.params["timesteps"],
            sharp_bookmakers=self.params["sharp_bookmakers"],
            retail_bookmakers=self.params["retail_bookmakers"],
        )

        # Create model with loaded architecture
        self.model = self._create_model().to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore metadata
        self.is_trained = checkpoint.get("is_trained", True)
        self.training_history = checkpoint.get("training_history")

        logger.info("model_loaded", filepath=str(filepath), is_trained=self.is_trained)

    def train_from_config(
        self,
        config: MLTrainingConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train LSTM model using configuration object.

        Extracts hyperparameters from the config, resolves any search spaces
        to concrete values, validates parameters, and logs all settings for
        experiment tracking.

        Args:
            config: ML training configuration with model hyperparameters
            X_train: Training features (n_samples, timesteps, n_features)
            y_train: Training targets - line movement deltas (n_samples,)
            feature_names: List of feature names in order
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history dictionary with metrics

        Raises:
            ValueError: If config has invalid parameters or wrong strategy type
            TypeError: If model config is not LSTMConfig

        Example:
            >>> config = MLTrainingConfig.from_yaml("experiments/lstm_line_movement.yaml")
            >>> strategy = LSTMLineMovementStrategy()
            >>> history = strategy.train_from_config(config, X_train, y_train, feature_names)
        """
        from odds_analytics.training.config import LSTMConfig, resolve_search_spaces

        # Validate strategy type
        if config.training.strategy_type != "lstm_line_movement":
            raise ValueError(
                f"Invalid strategy_type '{config.training.strategy_type}' for "
                f"LSTMLineMovementStrategy. Expected 'lstm_line_movement'."
            )

        # Validate model config type
        if not isinstance(config.training.model, LSTMConfig):
            raise TypeError(
                f"Expected LSTMConfig, got {type(config.training.model).__name__}. "
                f"Ensure strategy_type matches model configuration."
            )

        model_config = config.training.model

        # Extract hyperparameters from config
        lstm_params = {
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_layers,
            "dropout": model_config.dropout,
            "epochs": model_config.epochs,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "loss_function": model_config.loss_function,
            "weight_decay": model_config.weight_decay,
        }

        # Override with search space midpoints if tuning config exists
        if config.tuning and config.tuning.search_spaces:
            lstm_params = resolve_search_spaces(lstm_params, config.tuning.search_spaces)

        # Update strategy params
        self.params["hidden_size"] = lstm_params["hidden_size"]
        self.params["num_layers"] = lstm_params["num_layers"]
        self.params["dropout"] = lstm_params["dropout"]
        self.params["loss_function"] = lstm_params["loss_function"]

        # Update training parameters from config
        training_config = config.training
        self.params["min_predicted_movement"] = training_config.min_predicted_movement
        self.params["movement_confidence_scale"] = training_config.movement_confidence_scale
        self.params["base_confidence"] = training_config.base_confidence

        # Log all hyperparameters for experiment tracking
        logger.info(
            "train_from_config",
            experiment_name=config.experiment.name,
            strategy_type=config.training.strategy_type,
            n_samples=len(X_train),
            n_features=len(feature_names),
            has_validation=X_val is not None,
            **lstm_params,
        )

        # Update input size from actual data
        if len(X_train.shape) == 3:
            self.input_size = X_train.shape[2]  # (samples, timesteps, features)
        else:
            self.input_size = len(feature_names)

        # Create model with configured architecture
        self.model = self._create_model().to(self.device)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=lstm_params["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        # Validation data if provided
        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Loss function and optimizer
        criterion = self._get_loss_function()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lstm_params["learning_rate"],
            weight_decay=lstm_params["weight_decay"],
        )

        # Training loop
        history: dict[str, Any] = {
            "train_mse": 0.0,
            "train_mae": 0.0,
            "train_r2": 0.0,
            "n_samples": len(X_train),
            "n_features": len(feature_names),
        }

        epochs = lstm_params["epochs"]

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_X)

                # Calculate loss
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    mae = torch.mean(torch.abs(predictions - batch_y)).item()
                    epoch_mae += mae
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_mae = epoch_mae / num_batches

            logger.debug("training_epoch", epoch=epoch + 1, loss=avg_loss, mae=avg_mae)

        # Calculate final training metrics
        self.model.eval()
        with torch.no_grad():
            train_predictions = self.model(X_tensor).cpu().numpy()
            y_train_np = y_train

            train_mse = float(np.mean((train_predictions - y_train_np) ** 2))
            train_mae = float(np.mean(np.abs(train_predictions - y_train_np)))

            # R² score
            ss_res = np.sum((y_train_np - train_predictions) ** 2)
            ss_tot = np.sum((y_train_np - np.mean(y_train_np)) ** 2)
            train_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            history["train_mse"] = train_mse
            history["train_mae"] = train_mae
            history["train_r2"] = train_r2

            # Validation metrics if provided
            if X_val_tensor is not None and y_val_tensor is not None:
                val_predictions = self.model(X_val_tensor).cpu().numpy()
                y_val_np = y_val

                val_mse = float(np.mean((val_predictions - y_val_np) ** 2))
                val_mae = float(np.mean(np.abs(val_predictions - y_val_np)))

                ss_res_val = np.sum((y_val_np - val_predictions) ** 2)
                ss_tot_val = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
                val_r2 = float(1 - ss_res_val / ss_tot_val) if ss_tot_val > 0 else 0.0

                history["val_mse"] = val_mse
                history["val_mae"] = val_mae
                history["val_r2"] = val_r2

        self.is_trained = True
        self.training_history = {"loss": [train_mse], "mae": [train_mae]}

        logger.info(
            "training_complete",
            experiment_name=config.experiment.name,
            model_type="LSTM",
            train_mse=train_mse,
            val_mse=history.get("val_mse"),
        )

        return history
