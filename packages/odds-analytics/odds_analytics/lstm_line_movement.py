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
    async with async_session_maker() as session:
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

    async with async_session_maker() as session:
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
    from odds_analytics.training.cross_validation import CVResult
    from odds_analytics.training.tracking import ExperimentTracker

from odds_lambda.fetch_tier import FetchTier

from odds_analytics.backtesting import (
    BacktestConfig,
    BacktestEvent,
    BetOpportunity,
    BettingStrategy,
)
from odds_analytics.feature_extraction import SequenceFeatureExtractor
from odds_analytics.feature_groups import prepare_training_data
from odds_analytics.sequence_loader import load_sequences_for_event
from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

__all__ = ["LSTMModel", "LSTMLineMovementStrategy"]


class LSTMModel(nn.Module):
    """
    Unified LSTM model supporting both classification and regression.

    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of hidden units in LSTM layers (default: 64)
        num_layers: Number of stacked LSTM layers (default: 2)
        dropout: Dropout rate between LSTM layers (default: 0.2)
        output_type: Type of output - "classification" or "regression" (default: "classification")

    Input:
        x: Tensor of shape (batch_size, seq_len, input_size)

    Output:
        Classification: Tuple of (logits, probs) where both are (batch_size,)
        Regression: Single tensor of predictions (batch_size,)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_type: str = "classification",
        static_size: int = 0,
    ):
        super().__init__()

        if output_type not in ("classification", "regression"):
            raise ValueError(
                f"output_type must be 'classification' or 'regression', got '{output_type}'"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_type = output_type
        self.static_size = static_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output layer: hidden_size + static_size when static branch is active
        fc_input_size = hidden_size + static_size
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass through LSTM with optional static feature branch.

        Args:
            x: Sequence input of shape ``(batch, seq_len, input_size)``.
            static_features: Optional static input of shape ``(batch, static_size)``.
                When ``static_size > 0`` and this is ``None``, zeros are used.
        """
        # output shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from last layer
        final_hidden = h_n[-1]

        # Concat static features when the static branch is active
        if self.static_size > 0:
            if static_features is None:
                static_features = torch.zeros(
                    final_hidden.size(0), self.static_size, device=final_hidden.device
                )
            final_hidden = torch.cat([final_hidden, static_features], dim=-1)

        # Fully connected layer
        output = self.fc(final_hidden)

        # Squeeze to (batch_size,)
        output = output.squeeze(-1)

        if self.output_type == "classification":
            probs = torch.sigmoid(output)
            return output, probs
        else:
            return output


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
            static_size=self.params.get("static_size", 0),
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

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: int | None = None,
        min_delta: float = 0.0001,
        masks: np.ndarray | None = None,
        static_train: np.ndarray | None = None,
        static_val: np.ndarray | None = None,
        tracker: ExperimentTracker | None = None,
        trial: Any | None = None,
    ) -> dict[str, Any]:
        """Core training loop shared by train() and train_from_config()."""
        from odds_analytics.training.utils import compute_regression_metrics

        # Create model
        self.model = self._create_model().to(self.device)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        # Build dataset tensors list
        dataset_tensors: list[torch.Tensor] = [X_tensor, y_tensor]
        if masks is not None:
            dataset_tensors.append(torch.BoolTensor(masks).to(self.device))
        if static_train is not None:
            dataset_tensors.append(torch.FloatTensor(static_train).to(self.device))

        dataset = torch.utils.data.TensorDataset(*dataset_tensors)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Precompute tensor indices for batch unpacking
        _has_masks = masks is not None
        _has_static = static_train is not None
        _static_idx = 2 + int(_has_masks)  # index of static tensor in batch tuple

        # Validation tensors
        X_val_tensor = None
        y_val_tensor = None
        static_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            if static_val is not None:
                static_val_tensor = torch.FloatTensor(static_val).to(self.device)

        # Loss function and optimizer
        criterion = self._get_loss_function()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Early stopping setup
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0

            for batch in dataloader:
                batch_X, batch_y = batch[0], batch[1]
                batch_static = batch[_static_idx] if _has_static else None
                optimizer.zero_grad()

                predictions = self.model(batch_X, static_features=batch_static)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    mae = torch.mean(torch.abs(predictions - batch_y)).item()
                    epoch_mae += mae
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_mae = epoch_mae / num_batches

            # Validation metrics
            intermediate_value = avg_loss
            val_loss = None
            val_mae = None
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor, static_features=static_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()
                    val_mae = torch.mean(torch.abs(val_predictions - y_val_tensor)).item()
                    intermediate_value = val_loss

            # Log per-epoch metrics to tracker
            if tracker:
                epoch_metrics: dict[str, Any] = {"train_loss": avg_loss, "train_mae": avg_mae}
                if val_loss is not None:
                    epoch_metrics["val_loss"] = val_loss
                    epoch_metrics["val_mae"] = val_mae
                tracker.log_metrics(epoch_metrics, step=epoch)

            # Report to Optuna trial for pruning
            if trial is not None:
                trial.report(intermediate_value, epoch)
                if trial.should_prune():
                    import optuna

                    logger.info(
                        "trial_pruned",
                        trial_number=trial.number,
                        epoch=epoch,
                        value=intermediate_value,
                    )
                    raise optuna.TrialPruned()

            # Early stopping check
            if patience is not None and X_val_tensor is not None and val_loss is not None:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                    logger.debug("new_best_model", epoch=epoch + 1, val_loss=val_loss)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(
                            "early_stopping_triggered",
                            epoch=epoch + 1,
                            best_val_loss=best_val_loss,
                            epochs_without_improvement=epochs_without_improvement,
                            patience=patience,
                        )
                        break

            logger.debug("training_epoch", epoch=epoch + 1, loss=avg_loss, mae=avg_mae)

        # Restore best model if early stopping was used
        if patience is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("restored_best_model", val_loss=best_val_loss)

        # Calculate final metrics
        static_train_tensor = (
            torch.FloatTensor(static_train).to(self.device) if static_train is not None else None
        )
        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_tensor, static_features=static_train_tensor).cpu().numpy()
            train_metrics = compute_regression_metrics(y_train, train_preds)

        history: dict[str, Any] = {
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "n_samples": len(X_train),
            "n_features": X_train.shape[-1],
        }

        if X_val_tensor is not None and y_val is not None:
            with torch.no_grad():
                val_preds = (
                    self.model(X_val_tensor, static_features=static_val_tensor).cpu().numpy()
                )
                val_metrics = compute_regression_metrics(y_val, val_preds)
            history.update(
                {
                    "val_mse": val_metrics["mse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                }
            )

        self.is_trained = True
        self.training_history = {"loss": [train_metrics["mse"]], "mae": [train_metrics["mae"]]}

        return history

    async def train(
        self,
        events: list[Event],
        session: AsyncSession,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        outcome: str = "home",
        closing_tier: FetchTier = FetchTier.CLOSING,
        validation_split: float = 0.2,
        patience: int | None = None,
        min_delta: float = 0.0001,
    ) -> dict[str, Any]:
        """
        Train LSTM model on historical events for line movement regression.

        Args:
            events: List of events with final scores (status=FINAL)
            session: Async database session
            epochs: Number of training epochs (default: 20)
            batch_size: Batch size for training (default: 32)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            outcome: What to predict - "home" or "away" (default: "home")
            closing_tier: Fetch tier for closing line (default: CLOSING)
            validation_split: Fraction of data for validation (default: 0.2)
            patience: Stop if no improvement for N epochs (default: None)
            min_delta: Minimum change to qualify as improvement (default: 0.0001)

        Returns:
            Training history dict with final scalar metrics (MSE, MAE, R²)

        Example:
            >>> history = await strategy.train(events, session, epochs=20, patience=5)
            >>> print(f"Final MSE: {history['train_mse']:.4f}")
        """
        logger.info(
            "training_lstm_line_movement",
            num_events=len(events),
            epochs=epochs,
            batch_size=batch_size,
            outcome=outcome,
            loss_function=self.params.get("loss_function", "mse"),
            patience=patience,
        )

        # Prepare training data with regression targets using composable feature groups
        feature_config = FeatureConfig(
            adapter="lstm",
            outcome=outcome,
            markets=[self.params["market"]],
            sharp_bookmakers=self.params["sharp_bookmakers"],
            retail_bookmakers=self.params["retail_bookmakers"],
            lookback_hours=self.params["lookback_hours"],
            timesteps=self.params["timesteps"],
            closing_tier=closing_tier,
            feature_groups=["tabular"],
        )
        result = await prepare_training_data(
            events=events,
            session=session,
            config=feature_config,
        )
        X, y, masks = result.X, result.y, result.masks

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

        # Split into train and validation sets
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        X_train = X[:n_train]
        y_train = y[:n_train]
        masks_train = masks[:n_train]

        X_val = X[n_train:]
        y_val = y[n_train:]

        logger.info(
            "data_split",
            n_train=n_train,
            n_val=n_val,
            validation_split=validation_split,
        )

        history = self._train_loop(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            min_delta=min_delta,
            masks=masks_train,
        )

        logger.info(
            "training_complete",
            train_mse=history["train_mse"],
            train_r2=history["train_r2"],
            val_mse=history.get("val_mse"),
            val_r2=history.get("val_r2"),
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
            "static_size": self.params.get("static_size", 0),
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
        self.params["static_size"] = checkpoint.get("static_size", 0)

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
        tracker: ExperimentTracker | None = None,
        trial: Any | None = None,
        static_train: np.ndarray | None = None,
        static_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train LSTM model using configuration object.

        Extracts hyperparameters from the config, validates parameters, and
        delegates to _train_loop() for the actual training.

        Args:
            config: ML training configuration with model hyperparameters
            X_train: Training features (n_samples, timesteps, n_features)
            y_train: Training targets - line movement deltas (n_samples,)
            feature_names: List of feature names in order
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            tracker: Optional experiment tracker for logging (optional)
            trial: Optional Optuna trial for pruning support (optional)

        Returns:
            Training history dictionary with metrics

        Raises:
            ValueError: If config has invalid parameters or wrong strategy type
            TypeError: If model config is not LSTMConfig
        """
        from odds_analytics.training.config import LSTMConfig

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
            "patience": model_config.patience,
            "min_delta": model_config.min_delta,
        }

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

        # Log configuration parameters to tracker if enabled
        if tracker:
            from odds_analytics.training.utils import flatten_config_for_tracking

            config_params = flatten_config_for_tracking(config, X_train, feature_names, X_val)
            tracker.log_params(config_params)

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

        # Determine static feature size from data
        static_size = static_train.shape[1] if static_train is not None else 0
        self.params["static_size"] = static_size

        history = self._train_loop(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=lstm_params["epochs"],
            batch_size=lstm_params["batch_size"],
            learning_rate=lstm_params["learning_rate"],
            weight_decay=lstm_params["weight_decay"],
            patience=lstm_params.get("patience"),
            min_delta=lstm_params.get("min_delta", 0.0001),
            static_train=static_train,
            static_val=static_val,
            tracker=tracker,
            trial=trial,
        )

        # Log final metrics to tracker if enabled
        if tracker:
            final_metrics: dict[str, Any] = {
                "final_train_mse": history["train_mse"],
                "final_train_mae": history["train_mae"],
                "final_train_r2": history["train_r2"],
            }
            if "val_mse" in history:
                final_metrics.update(
                    {
                        "final_val_mse": history["val_mse"],
                        "final_val_mae": history["val_mae"],
                        "final_val_r2": history["val_r2"],
                    }
                )
            tracker.log_metrics(final_metrics)

            if self.model is not None:
                tracker.log_model(self.model, artifact_path="model")

        logger.info(
            "training_complete",
            experiment_name=config.experiment.name,
            model_type="LSTM",
            train_mse=history["train_mse"],
            val_mse=history.get("val_mse"),
        )

        return history

    def train_with_cv(
        self,
        config: MLTrainingConfig,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        event_ids: np.ndarray | None = None,
        static_features: np.ndarray | None = None,
        static_test: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], CVResult]:
        from odds_analytics.training.cross_validation import (
            train_with_cv as _train_with_cv,
        )

        return _train_with_cv(
            self,
            config,
            X,
            y,
            feature_names,
            X_test,
            y_test,
            event_ids,
            static_features=static_features,
            static_test=static_test,
        )
