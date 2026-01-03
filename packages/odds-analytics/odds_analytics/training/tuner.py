"""
Hyperparameter Tuner for ML Models.

This module provides an abstraction layer for hyperparameter tuning with
concrete implementations using Optuna for Bayesian optimization.

Key Features:
- Abstract base class for tuner interface
- Optuna integration with PostgreSQL persistence
- Multiple sampling strategies (TPE, Random, CMA-ES)
- Pruning support (Median, Hyperband, or none)
- Automatic parameter mapping from search spaces
- Objective function factory for training integration

Example:
    ```python
    from odds_analytics.training import MLTrainingConfig, OptunaTuner
    from odds_analytics.training.tuner import create_objective

    # Load configuration with search spaces
    config = MLTrainingConfig.from_yaml("experiments/xgboost_tuning.yaml")

    # Create tuner with PostgreSQL persistence
    tuner = OptunaTuner(
        study_name="xgboost_h2h_optimization",
        direction="minimize",
        sampler="tpe",
        pruner="median",
        storage="postgresql://...",
    )

    # Create objective function for training
    objective = create_objective(
        config=config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    # Run optimization
    study = tuner.optimize(
        objective=objective,
        n_trials=100,
        timeout=3600,
    )

    # Get results
    results_df = tuner.get_results()
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

if TYPE_CHECKING:
    import optuna
    from numpy import ndarray
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.training.config import MLTrainingConfig

logger = structlog.get_logger()

__all__ = [
    "HyperparameterTuner",
    "OptunaTuner",
    "create_objective",
]


# =============================================================================
# Abstract Base Class
# =============================================================================


class HyperparameterTuner(ABC):
    """
    Abstract base class for hyperparameter tuning.

    Provides interface for study initialization, optimization execution,
    and result retrieval with support for different tuning backends.
    """

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> Any:
        """
        Run hyperparameter optimization.

        Args:
            objective: Objective function to minimize/maximize
            n_trials: Number of trials to run (optional)
            timeout: Maximum optimization time in seconds (optional)

        Returns:
            Study object with optimization results
        """
        pass

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        """
        Get optimization results as DataFrame.

        Returns:
            DataFrame with trial results including parameters and metrics
        """
        pass

    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """
        Get best hyperparameters found during optimization.

        Returns:
            Dictionary of best parameter values
        """
        pass


# =============================================================================
# Optuna Implementation
# =============================================================================


class OptunaTuner(HyperparameterTuner):
    """
    Optuna-based hyperparameter tuner with PostgreSQL persistence.

    Supports multiple sampling strategies (TPE, Random, CMA-ES) and pruning
    approaches (Median, Hyperband, or none). Automatically maps configuration
    search spaces to Optuna's suggestion API.

    Args:
        study_name: Unique name for the optimization study
        direction: Optimization direction ("minimize" or "maximize")
        sampler: Sampling strategy ("tpe", "random", "cmaes")
        pruner: Pruning strategy ("median", "hyperband", "none")
        storage: Storage URL for persistence (e.g., "postgresql://...")
            If None, uses in-memory storage

    Example:
        >>> tuner = OptunaTuner(
        ...     study_name="xgboost_optimization",
        ...     direction="minimize",
        ...     sampler="tpe",
        ...     pruner="median",
        ...     storage=DATABASE_URL,
        ... )
        >>> study = tuner.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        study_name: str,
        direction: str = "minimize",
        sampler: str = "tpe",
        pruner: str = "median",
        storage: str | None = None,
    ):
        """Initialize Optuna tuner with specified configuration."""
        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError("optuna not installed. Install with: uv add optuna") from e

        self.study_name = study_name
        self.direction = direction
        self.storage = storage

        # Create sampler
        self.sampler = self._create_sampler(sampler)

        # Create pruner
        self.pruner = self._create_pruner(pruner)

        # Study will be created on first optimize call
        self.study: optuna.Study | None = None

        logger.info(
            "tuner_initialized",
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            has_storage=storage is not None,
        )

    def _create_sampler(self, sampler_name: str) -> Any:
        """
        Create Optuna sampler from configuration.

        Args:
            sampler_name: Sampler type ("tpe", "random", "cmaes")

        Returns:
            Optuna sampler instance
        """
        import optuna

        sampler_name = sampler_name.lower()

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(seed=42)
        elif sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        else:
            logger.warning(
                "unknown_sampler",
                sampler=sampler_name,
                message=f"Unknown sampler '{sampler_name}', defaulting to TPE",
            )
            return optuna.samplers.TPESampler(seed=42)

    def _create_pruner(self, pruner_name: str) -> Any:
        """
        Create Optuna pruner from configuration.

        Args:
            pruner_name: Pruner type ("median", "hyperband", "none")

        Returns:
            Optuna pruner instance or None
        """
        import optuna

        pruner_name = pruner_name.lower()

        if pruner_name == "median":
            return optuna.pruners.MedianPruner()
        elif pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif pruner_name == "none":
            return optuna.pruners.NopPruner()
        else:
            logger.warning(
                "unknown_pruner",
                pruner=pruner_name,
                message=f"Unknown pruner '{pruner_name}', defaulting to Median",
            )
            return optuna.pruners.MedianPruner()

    def optimize(
        self,
        objective: Callable[[Any], float],
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> Any:
        """
        Run Optuna optimization.

        Creates study if it doesn't exist and runs optimization with
        the specified number of trials or timeout.

        Args:
            objective: Objective function accepting Optuna trial
            n_trials: Number of trials to run (optional)
            timeout: Maximum time in seconds (optional)

        Returns:
            Optuna Study object with results

        Example:
            >>> def objective(trial):
            ...     x = trial.suggest_float("x", -10, 10)
            ...     return x ** 2
            >>> study = tuner.optimize(objective, n_trials=100)
        """
        import optuna

        # Create study if not exists
        if self.study is None:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                load_if_exists=True,
            )

            logger.info(
                "study_created",
                study_name=self.study_name,
                direction=self.direction,
            )

        # Run optimization
        logger.info(
            "optimization_started",
            study_name=self.study_name,
            n_trials=n_trials,
            timeout=timeout,
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(
            "optimization_completed",
            study_name=self.study_name,
            n_trials=len(self.study.trials),
            best_value=self.study.best_value,
        )

        return self.study

    def get_results(self) -> pd.DataFrame:
        """
        Get optimization results as DataFrame.

        Returns:
            DataFrame with columns: trial_number, value, params_*, state

        Raises:
            ValueError: If no study has been run yet
        """
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        return self.study.trials_dataframe()

    def get_best_params(self) -> dict[str, Any]:
        """
        Get best hyperparameters from optimization.

        Returns:
            Dictionary of best parameter values

        Raises:
            ValueError: If no study has been run yet
        """
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        return self.study.best_params


# =============================================================================
# Objective Function Factory
# =============================================================================


def suggest_params_from_search_space(
    trial: Any,
    search_spaces: dict[str, Any],
) -> dict[str, Any]:
    """
    Suggest parameters from search space definitions.

    Maps SearchSpace configurations to Optuna trial suggestions,
    automatically handling int, float, and categorical parameters
    with appropriate scaling and step sizes.

    Args:
        trial: Optuna trial object
        search_spaces: Dictionary mapping parameter names to SearchSpace objects

    Returns:
        Dictionary of suggested parameter values

    Example:
        >>> from odds_analytics.training.config import SearchSpace
        >>> search_spaces = {
        ...     "n_estimators": SearchSpace(type="int", low=50, high=500, step=50),
        ...     "learning_rate": SearchSpace(type="float", low=0.001, high=0.3, log=True),
        ... }
        >>> params = suggest_params_from_search_space(trial, search_spaces)
    """
    suggested_params = {}

    for param_name, space in search_spaces.items():
        if space.type == "int":
            suggested_params[param_name] = trial.suggest_int(
                param_name,
                low=int(space.low),
                high=int(space.high),
                step=int(space.step) if space.step else 1,
            )
        elif space.type == "float":
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                low=space.low,
                high=space.high,
                step=space.step,
                log=space.log,
            )
        elif space.type == "categorical":
            suggested_params[param_name] = trial.suggest_categorical(
                param_name,
                choices=space.choices,
            )
        else:
            logger.warning(
                "unknown_search_space_type",
                param_name=param_name,
                type=space.type,
            )

    return suggested_params


def create_objective(
    config: MLTrainingConfig,
    X_train: ndarray,
    y_train: ndarray,
    feature_names: list[str],
    X_val: ndarray | None = None,
    y_val: ndarray | None = None,
    session: AsyncSession | None = None,
) -> Callable[[Any], float]:
    """
    Create objective function for hyperparameter optimization.

    Factory function that creates a trial objective integrating with the
    training pipeline. The objective function:
    1. Extracts suggested parameters from trial
    2. Distinguishes between model parameters and feature parameters
    3. Invokes appropriate training methods
    4. Returns validation metric for optimization

    Args:
        config: ML training configuration with search spaces
        X_train: Training features
        y_train: Training targets
        feature_names: List of feature names
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        session: Database session for data extraction (optional)

    Returns:
        Objective function accepting Optuna trial and returning metric

    Raises:
        ValueError: If config has no search spaces or invalid strategy type

    Example:
        >>> config = MLTrainingConfig.from_yaml("experiments/tune.yaml")
        >>> objective = create_objective(config, X_train, y_train, feature_names, X_val, y_val)
        >>> study = tuner.optimize(objective, n_trials=50)
    """
    if config.tuning is None or not config.tuning.search_spaces:
        raise ValueError("No search spaces defined in tuning configuration")

    strategy_type = config.training.strategy_type
    search_spaces = config.tuning.search_spaces
    metric_name = config.tuning.metric

    # Determine which parameters are model vs feature parameters
    from odds_analytics.training.config import FeatureConfig

    feature_param_names = set(FeatureConfig.model_fields.keys())
    model_param_names = {name for name in search_spaces.keys() if name not in feature_param_names}

    logger.info(
        "objective_created",
        strategy_type=strategy_type,
        metric=metric_name,
        search_space_size=len(search_spaces),
        model_params=list(model_param_names),
        feature_params=list(feature_param_names & search_spaces.keys()),
    )

    def objective(trial: Any) -> float:
        """
        Objective function for single trial.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize (lower is better for minimize)
        """
        # Suggest parameters from search spaces
        suggested_params = suggest_params_from_search_space(trial, search_spaces)

        # Split into model and feature parameters
        model_params = {k: v for k, v in suggested_params.items() if k in model_param_names}
        feature_params = {k: v for k, v in suggested_params.items() if k in feature_param_names}

        logger.info(
            "trial_started",
            trial_number=trial.number,
            model_params=model_params,
            feature_params=feature_params,
        )

        # Create modified config with suggested parameters
        modified_config = config.model_copy(deep=True)

        # Update model parameters
        for param_name, param_value in model_params.items():
            if hasattr(modified_config.training.model, param_name):
                setattr(modified_config.training.model, param_name, param_value)

        # Update feature parameters
        for param_name, param_value in feature_params.items():
            if hasattr(modified_config.training.features, param_name):
                setattr(modified_config.training.features, param_name, param_value)

        # If feature parameters changed, we need to re-extract features
        # For now, we assume features are pre-extracted and only model params are tuned
        # TODO: Support feature parameter tuning with re-extraction

        # Import strategy classes
        if strategy_type in ("xgboost", "xgboost_line_movement"):
            from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy

            strategy = XGBoostLineMovementStrategy()
        elif strategy_type in ("lstm", "lstm_line_movement"):
            from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy

            strategy = LSTMLineMovementStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Train with modified configuration
        try:
            history = strategy.train_from_config(
                config=modified_config,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                X_val=X_val,
                y_val=y_val,
            )
        except Exception as e:
            logger.error(
                "trial_failed",
                trial_number=trial.number,
                error=str(e),
            )
            # Return worst possible value for failed trials
            return float("inf") if config.tuning.direction == "minimize" else float("-inf")

        # Extract metric value
        metric_value = history.get(metric_name)
        if metric_value is None:
            logger.warning(
                "metric_not_found",
                trial_number=trial.number,
                metric=metric_name,
                available_metrics=list(history.keys()),
            )
            return float("inf") if config.tuning.direction == "minimize" else float("-inf")

        logger.info(
            "trial_completed",
            trial_number=trial.number,
            metric=metric_name,
            value=metric_value,
        )

        return float(metric_value)

    return objective
