"""
XGBoost Line Movement Predictor Strategy.

This module implements an XGBoost-based regression model that predicts betting line
movement deltas (closing - opening) to identify value betting opportunities.

Key Features:
- XGBoost regressor for predicting continuous line movement values
- Uses TabularFeatureExtractor for feature engineering from opening odds
- Converts predicted movements to betting confidence for Kelly sizing
- Model persistence (save/load weights)
- Integration with backtesting framework

The core insight is that if a line is predicted to move significantly in one
direction (e.g., probability increasing for home team), it suggests the opening
line undervalues that outcome, creating a potential +EV betting opportunity.

Dependencies (install with uv):
    uv add xgboost scikit-learn numpy

Example:
    ```python
    # Load pre-trained model
    strategy = XGBoostLineMovementStrategy(model_path="models/xgb_line_movement.pkl")

    # Train a new model
    strategy = XGBoostLineMovementStrategy()
    strategy.train(X_train, y_train, feature_names)
    strategy.save_model("models/xgb_line_movement.pkl")

    # Use in backtesting
    from odds_analytics.backtesting import BacktestEngine, BacktestConfig
    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()
    ```
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
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
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.sequence_loader import (
    calculate_regression_target,
    extract_opening_closing_odds,
    load_sequences_for_event,
)

logger = structlog.get_logger()

__all__ = [
    "XGBoostLineMovementStrategy",
    "prepare_tabular_training_data",
]


class XGBoostLineMovementStrategy(BettingStrategy):
    """
    ML-based betting strategy using XGBoost regressor for line movement prediction.

    This strategy predicts how betting lines will move from opening to closing,
    then bets on outcomes where the predicted movement suggests undervaluation.

    The model predicts:
    - For h2h: Probability delta (closing_prob - opening_prob)
    - For spreads/totals: Point delta (closing_point - opening_point)

    A positive predicted delta for an outcome indicates the market is expected
    to move in favor of that outcome, suggesting it's currently undervalued.

    Key Parameters:
    - min_predicted_movement: Minimum absolute predicted delta to trigger a bet
    - movement_confidence_scale: Scaling factor to convert delta to confidence

    Example:
        ```python
        # Load pre-trained model
        strategy = XGBoostLineMovementStrategy(
            model_path="models/xgb_line_movement.pkl",
            min_predicted_movement=0.02,  # 2% probability movement
        )

        # Use in backtesting
        engine = BacktestEngine(strategy, config, reader)
        result = await engine.run()
        ```
    """

    def __init__(
        self,
        model_path: str | None = None,
        market: str = "h2h",
        min_predicted_movement: float = 0.02,
        movement_confidence_scale: float = 5.0,
        base_confidence: float = 0.52,
        bookmakers: list[str] | None = None,
        feature_names: list[str] | None = None,
        feature_extractor: TabularFeatureExtractor | None = None,
    ):
        """
        Initialize XGBoost Line Movement strategy.

        Args:
            model_path: Path to saved model file (loads on init if provided)
            market: Market to bet on (h2h, spreads, totals)
            min_predicted_movement: Minimum predicted movement to trigger bet
                For h2h: probability delta (e.g., 0.02 = 2%)
                For spreads/totals: point delta (e.g., 0.5 points)
            movement_confidence_scale: Scale factor to convert predicted movement
                to confidence. confidence = base + movement * scale (clamped 0.5-0.95)
            base_confidence: Base confidence when movement equals threshold
            bookmakers: List of bookmakers to consider (default: all major books)
            feature_names: List of feature names used by model (auto-set on load)
            feature_extractor: Feature extractor to use (default: TabularFeatureExtractor)
        """
        if bookmakers is None:
            bookmakers = [
                "pinnacle",
                "fanduel",
                "draftkings",
                "betmgm",
                "williamhill_us",
                "betrivers",
                "bovada",
                "circasports",
            ]

        super().__init__(
            name="XGBoostLineMovement",
            model_path=model_path,
            market=market,
            min_predicted_movement=min_predicted_movement,
            movement_confidence_scale=movement_confidence_scale,
            base_confidence=base_confidence,
            bookmakers=bookmakers,
        )

        self.model: Any = None  # XGBoost regressor
        self.feature_names: list[str] = feature_names or []
        self.feature_extractor = feature_extractor or TabularFeatureExtractor()

        if model_path:
            self.load_model(model_path)

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
        1. Extract features for both home and away outcomes
        2. Predict line movement delta for each
        3. Bet on outcomes with significant predicted positive movement

        Args:
            event: Event with final scores
            odds_snapshot: Odds at decision time (opening line)
            config: Backtest configuration
            session: Database session (optional)

        Returns:
            List of BetOpportunity objects
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        opportunities = []
        market = self.params["market"]
        bookmakers = self.params["bookmakers"]
        min_movement = self.params["min_predicted_movement"]
        movement_scale = self.params["movement_confidence_scale"]
        base_conf = self.params["base_confidence"]

        # Filter for target market and bookmakers
        market_odds = [
            o for o in odds_snapshot if o.market_key == market and o.bookmaker_key in bookmakers
        ]

        if not market_odds:
            return []

        # Evaluate both home and away team
        for outcome in [event.home_team, event.away_team]:
            # Extract features for this outcome
            features = self.feature_extractor.extract_features(
                event, odds_snapshot, market=market, outcome=outcome
            )

            if not features:
                continue

            # Convert to feature vector
            feature_vector = self.feature_extractor.create_feature_vector(
                features, self.feature_names if self.feature_names else None
            )

            # Handle NaN values (replace with 0 for prediction)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            # Predict line movement delta
            predicted_movement = self._predict_movement(feature_vector)

            # Only bet if significant positive movement is predicted
            # Positive movement = line expected to move in favor of this outcome
            if predicted_movement < min_movement:
                continue

            # Find best available odds for this outcome
            outcome_odds = [o for o in market_odds if o.outcome_name == outcome]
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
                    outcome=outcome,
                    bookmaker=best_odd.bookmaker_key,
                    odds=best_odd.price,
                    line=best_odd.point,
                    confidence=confidence,
                    rationale=f"Predicted movement: {predicted_movement:+.3f} "
                    f"({'prob' if market == 'h2h' else 'points'}) "
                    f"at {best_odd.bookmaker_key}",
                )
            )

        return opportunities

    def _predict_movement(self, feature_vector: np.ndarray) -> float:
        """
        Predict line movement delta using trained model.

        Args:
            feature_vector: Feature array for single prediction

        Returns:
            Predicted movement delta (positive = favorable movement expected)
        """
        prediction = self.model.predict(feature_vector.reshape(1, -1))
        return float(prediction[0])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **xgb_params,
    ) -> dict[str, Any]:
        """
        Train XGBoost regressor for line movement prediction.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets - line movement deltas (n_samples,)
            feature_names: List of feature names in order
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **xgb_params: Additional XGBoost parameters

        Returns:
            Training history dictionary

        Note:
            Requires xgboost package installed
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost not installed. Install with: uv add xgboost") from e

        self.feature_names = feature_names

        # Default parameters optimized for regression
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
        }

        # Merge with user-provided params
        params = {**default_params, **xgb_params}

        self.model = XGBRegressor(**params)

        # Prepare eval set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_mse = float(np.mean((train_predictions - y_train) ** 2))
        train_mae = float(np.mean(np.abs(train_predictions - y_train)))
        train_r2 = float(
            1
            - np.sum((y_train - train_predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
        )

        history = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "n_samples": len(X_train),
            "n_features": len(feature_names),
        }

        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_mse = float(np.mean((val_predictions - y_val) ** 2))
            val_mae = float(np.mean(np.abs(val_predictions - y_val)))
            val_r2 = float(
                1 - np.sum((y_val - val_predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
            )
            history.update(
                {
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                }
            )

        logger.info(
            "model_trained",
            train_mse=train_mse,
            train_mae=train_mae,
            train_r2=train_r2,
            n_samples=len(X_train),
        )

        return history

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
        Train XGBoost regressor using configuration object.

        Extracts hyperparameters from the config, resolves any search spaces
        to concrete values, validates parameters, and logs all settings for
        experiment tracking.

        Args:
            config: ML training configuration with model hyperparameters
            X_train: Training features (n_samples, n_features)
            y_train: Training targets - line movement deltas (n_samples,)
            feature_names: List of feature names in order
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history dictionary with metrics

        Raises:
            ValueError: If config has invalid parameters or wrong strategy type
            TypeError: If model config is not XGBoostConfig

        Example:
            >>> config = MLTrainingConfig.from_yaml("experiments/xgb_line_movement.yaml")
            >>> strategy = XGBoostLineMovementStrategy()
            >>> history = strategy.train_from_config(config, X_train, y_train, feature_names)
        """
        from odds_analytics.training.config import XGBoostConfig, resolve_search_spaces

        # Validate strategy type
        if config.training.strategy_type != "xgboost_line_movement":
            raise ValueError(
                f"Invalid strategy_type '{config.training.strategy_type}' for "
                f"XGBoostLineMovementStrategy. Expected 'xgboost_line_movement'."
            )

        # Validate model config type
        if not isinstance(config.training.model, XGBoostConfig):
            raise TypeError(
                f"Expected XGBoostConfig, got {type(config.training.model).__name__}. "
                f"Ensure strategy_type matches model configuration."
            )

        model_config = config.training.model

        # Extract hyperparameters from config
        xgb_params = {
            "n_estimators": model_config.n_estimators,
            "max_depth": model_config.max_depth,
            "min_child_weight": model_config.min_child_weight,
            "learning_rate": model_config.learning_rate,
            "gamma": model_config.gamma,
            "subsample": model_config.subsample,
            "colsample_bytree": model_config.colsample_bytree,
            "colsample_bylevel": model_config.colsample_bylevel,
            "colsample_bynode": model_config.colsample_bynode,
            "reg_alpha": model_config.reg_alpha,
            "reg_lambda": model_config.reg_lambda,
            "objective": model_config.objective,
            "random_state": model_config.random_state,
            "n_jobs": model_config.n_jobs,
        }

        # Handle early stopping if configured
        if model_config.early_stopping_rounds is not None:
            xgb_params["early_stopping_rounds"] = model_config.early_stopping_rounds

        # Override with search space midpoints if tuning config exists
        if config.tuning and config.tuning.search_spaces:
            xgb_params = resolve_search_spaces(xgb_params, config.tuning.search_spaces)

        # Log all hyperparameters for experiment tracking
        logger.info(
            "train_from_config",
            experiment_name=config.experiment.name,
            strategy_type=config.training.strategy_type,
            n_samples=len(X_train),
            n_features=len(feature_names),
            has_validation=X_val is not None,
            **xgb_params,
        )

        # Call existing train method
        history = self.train(
            X_train, y_train, feature_names, X_val=X_val, y_val=y_val, **xgb_params
        )

        # Log training completion
        logger.info(
            "training_complete",
            experiment_name=config.experiment.name,
            model_type="XGBoostRegressor",
            train_mse=history.get("train_mse"),
            val_mse=history.get("val_mse"),
        )

        return history

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (e.g., 'models/xgb_line_movement.pkl')
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("model_saved", filepath=filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

        # Update params if saved (for backward compatibility)
        if "params" in model_data:
            self.params.update(model_data["params"])

        logger.info(
            "model_loaded",
            filepath=filepath,
            n_features=len(self.feature_names),
        )

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Train or load model first.")

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores, strict=True))


async def prepare_tabular_training_data(
    events: list[Event],
    session: AsyncSession,
    outcome: str = "home",
    market: str = "h2h",
    opening_hours_before: float = 48.0,
    closing_hours_before: float = 0.5,
    sharp_bookmakers: list[str] | None = None,
    retail_bookmakers: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare training data for tabular XGBoost line movement model.

    This function:
    1. Loads historical odds sequences for each event
    2. Extracts features from opening line snapshot
    3. Calculates regression targets (closing - opening line delta)
    4. Returns arrays suitable for XGBoost training

    Args:
        events: List of Event objects with final scores (status=FINAL required)
        session: Async database session
        outcome: What to predict - "home" or "away"
        market: Market to analyze (h2h, spreads, totals)
        opening_hours_before: Hours before game for opening line (default: 48)
        closing_hours_before: Hours before game for closing line (default: 0.5)
        sharp_bookmakers: Sharp bookmakers for features (default: ["pinnacle"])
        retail_bookmakers: Retail bookmakers for features

    Returns:
        Tuple of (X, y, feature_names):
        - X: Feature array of shape (n_samples, n_features)
        - y: Target array of shape (n_samples,) - line movement deltas
        - feature_names: List of feature names

    Example:
        ```python
        from odds_lambda.storage.readers import OddsReader

        async with get_async_session() as session:
            reader = OddsReader(session)
            events = await reader.get_events_by_date_range(
                start_date=datetime(2024, 10, 1, tzinfo=UTC),
                end_date=datetime(2024, 12, 31, tzinfo=UTC),
                status=EventStatus.FINAL
            )

            X, y, feature_names = await prepare_tabular_training_data(
                events=events,
                session=session,
                outcome="home",
                market="h2h"
            )

            # Train model
            strategy = XGBoostLineMovementStrategy()
            strategy.train(X, y, feature_names)
        ```
    """
    if not events:
        logger.warning("no_events_provided")
        return np.array([]), np.array([]), []

    # Filter for events with final scores
    from odds_core.models import EventStatus

    valid_events = [
        e
        for e in events
        if e.status == EventStatus.FINAL and e.home_score is not None and e.away_score is not None
    ]

    if not valid_events:
        logger.warning("no_valid_events", total_events=len(events))
        return np.array([]), np.array([]), []

    # Initialize feature extractor
    extractor = TabularFeatureExtractor(
        sharp_bookmakers=sharp_bookmakers,
        retail_bookmakers=retail_bookmakers,
    )

    feature_names = extractor.get_feature_names()
    n_features = len(feature_names)

    # Pre-allocate arrays
    n_samples = len(valid_events)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    valid_sample_idx = 0
    skipped_events = 0

    for event in valid_events:
        # Load odds sequences for this event
        try:
            odds_sequences = await load_sequences_for_event(event.id, session)
        except Exception as e:
            logger.error(
                "failed_to_load_sequences",
                event_id=event.id,
                error=str(e),
            )
            skipped_events += 1
            continue

        if not odds_sequences:
            skipped_events += 1
            continue

        # Determine target outcome
        if outcome == "home":
            target_outcome = event.home_team
        elif outcome == "away":
            target_outcome = event.away_team
        else:
            target_outcome = outcome

        # Extract opening and closing odds for regression target
        opening_odds, closing_odds = extract_opening_closing_odds(
            odds_sequences,
            target_outcome,
            market,
            commence_time=event.commence_time,
            opening_hours_before=opening_hours_before,
            closing_hours_before=closing_hours_before,
        )

        # Calculate regression target
        regression_target = calculate_regression_target(opening_odds, closing_odds, market)

        if regression_target is None:
            logger.debug(
                "missing_regression_target",
                event_id=event.id,
            )
            skipped_events += 1
            continue

        # Find opening snapshot for feature extraction
        # Use the snapshot that matches our opening_odds timing
        opening_snapshot = None
        for snapshot in odds_sequences:
            # Filter for target market and outcome
            filtered = [
                o for o in snapshot if o.market_key == market and o.outcome_name == target_outcome
            ]
            if (
                filtered
                and opening_odds
                and filtered[0].odds_timestamp == opening_odds[0].odds_timestamp
            ):
                opening_snapshot = snapshot
                break

        if not opening_snapshot:
            # Fallback: use first snapshot with relevant data
            for snapshot in odds_sequences:
                filtered = [
                    o
                    for o in snapshot
                    if o.market_key == market and o.outcome_name == target_outcome
                ]
                if filtered:
                    opening_snapshot = snapshot
                    break

        if not opening_snapshot:
            skipped_events += 1
            continue

        # Create BacktestEvent for feature extraction
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        # Extract features from opening snapshot
        try:
            features = extractor.extract_features(
                event=backtest_event,
                odds_data=opening_snapshot,
                outcome=target_outcome,
                market=market,
            )
        except Exception as e:
            logger.error(
                "failed_to_extract_features",
                event_id=event.id,
                error=str(e),
            )
            skipped_events += 1
            continue

        # Convert to feature vector
        feature_vector = extractor.create_feature_vector(features)

        # Store in arrays
        X[valid_sample_idx] = feature_vector
        y[valid_sample_idx] = regression_target

        valid_sample_idx += 1

    # Trim arrays to actual number of valid samples
    if valid_sample_idx < n_samples:
        logger.info(
            "trimming_arrays",
            original_size=n_samples,
            valid_samples=valid_sample_idx,
            skipped=skipped_events,
        )
        X = X[:valid_sample_idx]
        y = y[:valid_sample_idx]

    # Replace NaN with 0 for XGBoost compatibility
    X = np.nan_to_num(X, nan=0.0)

    logger.info(
        "prepared_training_data",
        num_samples=valid_sample_idx,
        num_features=n_features,
        outcome=outcome,
        market=market,
        skipped_events=skipped_events,
        target_mean=float(np.mean(y)) if valid_sample_idx > 0 else 0.0,
        target_std=float(np.std(y)) if valid_sample_idx > 0 else 0.0,
    )

    return X, y, feature_names
