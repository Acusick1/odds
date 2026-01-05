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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import structlog
import yaml
from odds_core.models import Odds
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from odds_analytics.training.config import MLTrainingConfig
    from odds_analytics.training.tracking import ExperimentTracker

from odds_analytics.backtesting import (
    BacktestConfig,
    BacktestEvent,
    BetOpportunity,
    BettingStrategy,
)
from odds_analytics.feature_extraction import TabularFeatureExtractor

logger = structlog.get_logger()

__all__ = [
    "XGBoostLineMovementStrategy",
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
        tracker: ExperimentTracker | None = None,
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
            tracker: Optional experiment tracker for logging (optional)
            **xgb_params: Additional XGBoost parameters

        Returns:
            Training history dictionary

        Note:
            Requires xgboost package installed
            Per-round metrics are automatically logged via MLflow autolog when tracker is enabled
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
        tracker: ExperimentTracker | None = None,
        trial: Any | None = None,
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
            tracker: Optional experiment tracker for logging (optional)
            trial: Optional Optuna trial for pruning support (optional)

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
        from odds_analytics.training.config import XGBoostConfig

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
            **xgb_params,
        )

        # Add Optuna pruning callback if trial provided
        if trial is not None and X_val is not None:
            try:
                from optuna.integration import XGBoostPruningCallback

                # Create pruning callback that reports validation MSE
                pruning_callback = XGBoostPruningCallback(trial, "validation_1-rmse")

                # Add callback to xgb_params
                if "callbacks" not in xgb_params:
                    xgb_params["callbacks"] = []
                xgb_params["callbacks"].append(pruning_callback)

                logger.debug("optuna_pruning_enabled", trial_number=trial.number)
            except ImportError:
                logger.warning(
                    "optuna_integration_unavailable",
                    message="XGBoostPruningCallback not available, pruning disabled",
                )

        # Call train method with tracker for per-round metrics
        history = self.train(
            X_train,
            y_train,
            feature_names,
            X_val=X_val,
            y_val=y_val,
            tracker=tracker,
            **xgb_params,
        )

        # Log final test metrics to tracker if enabled
        if tracker:
            final_metrics = {
                "final_train_mse": history.get("train_mse"),
                "final_train_mae": history.get("train_mae"),
                "final_train_r2": history.get("train_r2"),
            }
            if X_val is not None:
                final_metrics.update(
                    {
                        "final_val_mse": history.get("val_mse"),
                        "final_val_mae": history.get("val_mae"),
                        "final_val_r2": history.get("val_r2"),
                    }
                )
            # Filter out None values before logging
            final_metrics_filtered: dict[str, float] = {
                k: float(v) for k, v in final_metrics.items() if v is not None
            }
            tracker.log_metrics(final_metrics_filtered)

            # Log model artifact
            if self.model is not None:
                tracker.log_model(self.model, artifact_path="model")

        # Log training completion
        logger.info(
            "training_complete",
            experiment_name=config.experiment.name,
            model_type="XGBoostRegressor",
            train_mse=history.get("train_mse"),
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
    ) -> tuple[dict[str, Any], Any]:
        """
        Train with K-Fold cross-validation, then train final model on all data.

        This method:
        1. Runs K-Fold CV to get robust performance estimates
        2. Trains a final model on all training data
        3. Returns both CV results and final model metrics

        Args:
            config: ML training configuration with kfold settings
            X: Full training feature matrix (n_samples, n_features)
            y: Full training target vector (n_samples,)
            feature_names: List of feature names
            X_test: Optional held-out test features for final evaluation
            y_test: Optional held-out test targets for final evaluation

        Returns:
            Tuple of (training_history, cv_result) where:
            - training_history: Dict with final model metrics + cv metrics
            - cv_result: CVResult object with per-fold details

        Example:
            >>> strategy = XGBoostLineMovementStrategy()
            >>> history, cv_result = strategy.train_with_cv(
            ...     config, X_train, y_train, feature_names, X_test, y_test
            ... )
            >>> print(f"CV R²: {cv_result.mean_val_r2:.4f} ± {cv_result.std_val_r2:.4f}")
            >>> print(f"Final test R²: {history['val_r2']:.4f}")
        """
        from odds_analytics.training.cross_validation import run_cv

        logger.info(
            "starting_train_with_cv",
            experiment_name=config.experiment.name,
            n_folds=config.training.data.n_folds,
            n_samples=len(X),
            n_features=len(feature_names),
        )

        # Step 1: Run cross-validation (time series or k-fold based on config)
        cv_result = run_cv(
            strategy=self,
            config=config,
            X=X,
            y=y,
            feature_names=feature_names,
        )

        logger.info(
            "cv_complete_training_final",
            cv_val_mse=f"{cv_result.mean_val_mse:.6f} ± {cv_result.std_val_mse:.6f}",
            cv_val_r2=f"{cv_result.mean_val_r2:.4f} ± {cv_result.std_val_r2:.4f}",
        )

        # Step 2: Train final model on all training data
        history = self.train_from_config(
            config=config,
            X_train=X,
            y_train=y,
            feature_names=feature_names,
            X_val=X_test,
            y_val=y_test,
        )

        # Step 3: Merge CV metrics into history
        history.update(cv_result.to_dict())

        logger.info(
            "train_with_cv_complete",
            experiment_name=config.experiment.name,
            cv_val_mse_mean=cv_result.mean_val_mse,
            final_train_mse=history.get("train_mse"),
            final_test_mse=history.get("val_mse"),
        )

        return history, cv_result

    def save_model(self, filepath: str | Path) -> None:
        """
        Save trained model to disk with configuration metadata.

        Saves model using joblib and configuration as YAML alongside it.
        For example, 'models/xgb.joblib' will also create 'models/xgb.yaml'.

        Args:
            filepath: Path to save model (e.g., 'models/xgb_line_movement.joblib')
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        filepath = Path(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata using joblib
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }

        joblib.dump(model_data, filepath)

        # Save configuration as YAML file alongside model
        config_filepath = filepath.with_suffix(".yaml")
        config_data = {
            "model_type": "XGBoostLineMovement",
            "saved_at": datetime.now(UTC).isoformat(),
            "params": self.params,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
        }

        with open(config_filepath, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(
            "model_saved",
            filepath=str(filepath),
            config_filepath=str(config_filepath),
        )

    def load_model(self, filepath: str | Path) -> None:
        """
        Load trained model from disk.

        Supports both new format (joblib + YAML config) and old format (pickle).
        If a YAML config file exists alongside the model, it will be loaded
        and logged for reproducibility tracking.

        Args:
            filepath: Path to saved model file
        """
        import pickle

        filepath = Path(filepath)

        # Try joblib first, fall back to pickle for backward compatibility
        try:
            model_data = joblib.load(filepath)
        except Exception:
            # Fall back to pickle for old models
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            logger.debug("loaded_legacy_pickle_format", filepath=str(filepath))

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

        # Update params if saved (for backward compatibility)
        if "params" in model_data:
            self.params.update(model_data["params"])

        # Try to load config file if it exists
        config_filepath = filepath.with_suffix(".yaml")
        config_loaded = False
        if config_filepath.exists():
            try:
                with open(config_filepath) as f:
                    config_data = yaml.safe_load(f)
                config_loaded = True
                logger.info(
                    "config_loaded",
                    config_filepath=str(config_filepath),
                    model_type=config_data.get("model_type"),
                    saved_at=config_data.get("saved_at"),
                )
            except Exception as e:
                logger.warning(
                    "config_load_failed",
                    config_filepath=str(config_filepath),
                    error=str(e),
                )

        logger.info(
            "model_loaded",
            filepath=str(filepath),
            n_features=len(self.feature_names),
            config_loaded=config_loaded,
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
