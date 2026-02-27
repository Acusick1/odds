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

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    import optuna
    from numpy import ndarray

    from odds_analytics.training.config import MLTrainingConfig, TrackingConfig
    from odds_analytics.training.data_preparation import TrainingDataResult

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
    Optuna-based hyperparameter tuner with PostgreSQL persistence and MLflow integration.

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
        tracking_config: Optional TrackingConfig for MLflow experiment tracking
            If provided, each trial is logged as a nested run with the parent run
            containing study metadata and the best model

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
        tracking_config: TrackingConfig | None = None,
    ):
        """Initialize Optuna tuner with specified configuration."""
        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError("optuna not installed. Install with: uv add optuna") from e

        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.tracking_config = tracking_config

        # Create sampler
        self.sampler = self._create_sampler(sampler)

        # Create pruner
        self.pruner = self._create_pruner(pruner)

        # Study will be created on first optimize call
        self.study: optuna.Study | None = None

        # MLflow tracker and parent run info
        self._tracker = None
        self._parent_run_id = None

        logger.info(
            "tuner_initialized",
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            has_storage=storage is not None,
            tracking_enabled=tracking_config is not None and tracking_config.enabled,
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
        Run Optuna optimization with optional MLflow tracking.

        Creates study if it doesn't exist and runs optimization with
        the specified number of trials or timeout.

        When tracking is enabled (via tracking_config):
        - Creates a parent MLflow run for study metadata
        - Uses Optuna's MLflowCallback to log each trial as a nested run
        - Logs study-level information (trial count, direction, sampler, pruner)
        - After optimization, logs best parameters and value to parent run

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

        # Initialize MLflow tracking if enabled
        callbacks = []
        if self.tracking_config and self.tracking_config.enabled:
            try:
                from odds_analytics.training.tracking import create_tracker

                # Create tracker and start parent run
                self._tracker = create_tracker(self.tracking_config)
                self._tracker.start_run(
                    run_name=f"{self.study_name}_parent",
                    tags={
                        "mlflow.runName": f"{self.study_name}_parent",
                        "study_name": self.study_name,
                        "type": "hyperparameter_tuning",
                    },
                )

                # Log study configuration
                self._tracker.log_params(
                    {
                        "study_name": self.study_name,
                        "direction": self.direction,
                        "sampler": self.sampler.__class__.__name__,
                        "pruner": self.pruner.__class__.__name__,
                        "n_trials": n_trials or "unlimited",
                        "timeout": timeout or "unlimited",
                    }
                )

                # Get parent run ID for MLflowCallback
                import mlflow

                self._parent_run_id = mlflow.active_run().info.run_id

                # Create MLflow callback for nested trial runs
                mlflow_callback = optuna.integration.MLflowCallback(
                    tracking_uri=self.tracking_config.tracking_uri,
                    metric_name="value",
                    create_experiment=False,
                    mlflow_kwargs={
                        "experiment_id": mlflow.active_run().info.experiment_id,
                        "nested": True,
                    },
                )
                callbacks.append(mlflow_callback)

                logger.info(
                    "mlflow_tracking_initialized",
                    parent_run_id=self._parent_run_id,
                    experiment_id=mlflow.active_run().info.experiment_id,
                )

            except Exception as e:
                logger.warning(
                    "mlflow_tracking_setup_failed",
                    error=str(e),
                    message="Continuing without tracking",
                )
                self._tracker = None
                self._parent_run_id = None

        # Run optimization
        logger.info(
            "optimization_started",
            study_name=self.study_name,
            n_trials=n_trials,
            timeout=timeout,
            tracking_enabled=bool(self._tracker),
        )

        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                callbacks=callbacks if callbacks else None,
            )
        finally:
            # Log final results to parent run if tracking enabled
            if self._tracker:
                run_status = "FINISHED"
                if self.study.best_trial:
                    try:
                        # Log best trial results
                        self._tracker.log_params(
                            {f"best_{k}": v for k, v in self.study.best_params.items()}
                        )
                        self._tracker.log_metrics(
                            {
                                "best_value": self.study.best_value,
                                "n_completed_trials": len(
                                    [
                                        t
                                        for t in self.study.trials
                                        if t.state == optuna.trial.TrialState.COMPLETE
                                    ]
                                ),
                                "n_pruned_trials": len(
                                    [
                                        t
                                        for t in self.study.trials
                                        if t.state == optuna.trial.TrialState.PRUNED
                                    ]
                                ),
                                "n_failed_trials": len(
                                    [
                                        t
                                        for t in self.study.trials
                                        if t.state == optuna.trial.TrialState.FAIL
                                    ]
                                ),
                            }
                        )
                        logger.info(
                            "mlflow_parent_run_updated",
                            best_value=self.study.best_value,
                            n_trials=len(self.study.trials),
                        )
                    except Exception as e:
                        logger.warning(
                            "mlflow_parent_run_logging_failed",
                            error=str(e),
                        )
                else:
                    # No successful trials - mark as failed
                    run_status = "FAILED"
                    logger.warning(
                        "optimization_no_successful_trials",
                        study_name=self.study_name,
                        n_trials=len(self.study.trials),
                    )

                # Always end parent run
                self._tracker.end_run(status=run_status)

        # Log feature selection results to MLflow if available
        if self._tracker and self._parent_run_id:
            try:
                ranking_names = self.study.user_attrs.get("feature_ranking_names")
                ranking_scores = self.study.user_attrs.get("feature_ranking_scores")
                ranking_method = self.study.user_attrs.get("feature_ranking_method")
                ranking_metadata = self.study.user_attrs.get("feature_ranking_metadata")

                if ranking_names:
                    import mlflow

                    with mlflow.start_run(run_id=self._parent_run_id):
                        # Log feature ranking parameters
                        self._tracker.log_params(
                            {
                                "feature_selection_method": ranking_method,
                                "n_ranked_features": len(ranking_names),
                            }
                        )

                        # Log top features as parameters (limit to 10)
                        for i, (name, score) in enumerate(
                            zip(ranking_names[:10], ranking_scores[:10], strict=False)
                        ):
                            self._tracker.log_params(
                                {
                                    f"top_feature_{i + 1}_name": name,
                                    f"top_feature_{i + 1}_score": score,
                                }
                            )

                        # Log complete rankings as artifact (JSON)
                        import json
                        import tempfile
                        from pathlib import Path

                        with tempfile.TemporaryDirectory() as tmpdir:
                            ranking_file = Path(tmpdir) / "feature_ranking.json"
                            with open(ranking_file, "w") as f:
                                json.dump(
                                    {
                                        "feature_names": ranking_names,
                                        "scores": ranking_scores,
                                        "method": ranking_method,
                                        "metadata": ranking_metadata,
                                    },
                                    f,
                                    indent=2,
                                )
                            mlflow.log_artifact(str(ranking_file), "feature_selection")

                        logger.info(
                            "feature_ranking_logged_to_mlflow",
                            n_features=len(ranking_names),
                            method=ranking_method,
                        )

            except Exception as e:
                logger.warning(
                    "feature_ranking_mlflow_logging_failed",
                    error=str(e),
                )

        # Guard best_value access - only log if we have a best trial
        log_data = {
            "study_name": self.study_name,
            "n_trials": len(self.study.trials),
        }
        if self.study.best_trial:
            log_data["best_value"] = self.study.best_value
            log_data["best_trial_number"] = self.study.best_trial.number

            # Include mean/std CV metrics from best trial if available
            if hasattr(self.study.best_trial, "user_attrs"):
                for key, value in self.study.best_trial.user_attrs.items():
                    if isinstance(value, int | float) and key.startswith(("mean_", "std_")):
                        log_data[key] = value

        logger.info("optimization_completed", **log_data)

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

    def log_best_model(self, model: Any, artifact_path: str = "best_model") -> None:
        """
        Log the best model to the parent MLflow run.

        This should be called after training the final model with best parameters
        to persist it alongside the hyperparameter tuning study metadata.

        Args:
            model: The trained model object (XGBoost, PyTorch, etc.)
            artifact_path: Path within artifact store for the model

        Raises:
            ValueError: If tracking is not enabled or parent run doesn't exist

        Example:
            >>> study = tuner.optimize(objective, n_trials=100)
            >>> # Train final model with best params
            >>> final_model = train_with_best_params(study.best_params)
            >>> tuner.log_best_model(final_model)
        """
        if not self._tracker or not self._parent_run_id:
            raise ValueError(
                "Cannot log model: tracking is not enabled or parent run doesn't exist. "
                "Ensure tracking_config is provided and optimize() has been called."
            )

        try:
            # Temporarily reactivate parent run to log model
            import mlflow

            with mlflow.start_run(run_id=self._parent_run_id):
                self._tracker.log_model(model, artifact_path=artifact_path)
                logger.info(
                    "best_model_logged",
                    run_id=self._parent_run_id,
                    artifact_path=artifact_path,
                )
        except Exception as e:
            logger.error(
                "best_model_logging_failed",
                error=str(e),
                run_id=self._parent_run_id,
            )
            raise


# =============================================================================
# Objective Function Factory
# =============================================================================


def _compute_feature_config_hash(feature_params: dict[str, Any]) -> str:
    """
    Compute a stable hash of feature configuration parameters.

    Args:
        feature_params: Dictionary of feature configuration parameters

    Returns:
        Hex string hash of the parameters

    Example:
        >>> params = {"normalize": True, "movement_threshold": 0.01}
        >>> hash1 = _compute_feature_config_hash(params)
        >>> hash2 = _compute_feature_config_hash(params)
        >>> assert hash1 == hash2  # Stable hash
    """
    import hashlib
    import json

    # Sort keys for stable hashing
    sorted_params = json.dumps(feature_params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


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
            # Convert unhashable choices (lists) to tuples for Optuna
            choices = [tuple(c) if isinstance(c, list) else c for c in space.choices]
            suggested_params[param_name] = trial.suggest_categorical(param_name, choices=choices)
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
    precomputed_features: dict[tuple[str, ...], TrainingDataResult] | None = None,
    static_train: ndarray | None = None,
    static_val: ndarray | None = None,
) -> Callable[[Any], float]:
    """
    Create objective function for hyperparameter optimization.

    Factory function that creates a trial objective integrating with the
    training pipeline. The objective function:
    1. Extracts suggested parameters from trial
    2. Looks up pre-computed features for feature_groups parameter
    3. Runs feature selection once on first trial (if enabled and top_k_features in search space)
    4. Subsets features based on top_k_features parameter (if in search space)
    5. Invokes appropriate training methods
    6. Returns validation metric for optimization

    Args:
        config: ML training configuration with search spaces
        X_train: Training features (default features)
        y_train: Training targets (default targets)
        feature_names: List of feature names (default feature names)
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        precomputed_features: Pre-computed features for each feature_groups choice.
            Dict mapping feature_groups tuple to TrainingDataResult.
            Pre-compute these before calling create_objective to avoid async calls during trials.

    Returns:
        Objective function accepting Optuna trial and returning metric

    Raises:
        ValueError: If config has no search spaces or invalid strategy type

    Example:
        >>> config = MLTrainingConfig.from_yaml("experiments/tune.yaml")
        >>> # Pre-compute features for all feature_groups choices
        >>> precomputed = {("tabular",): data_result1, ("tabular", "polymarket"): data_result2}
        >>> objective = create_objective(config, X_train, y_train, feature_names, X_val, y_val, precomputed)
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

    # Check if feature_groups is being tuned and we have pre-computed features
    tuning_feature_groups = "feature_groups" in search_spaces
    if tuning_feature_groups and precomputed_features is None:
        logger.warning(
            "feature_groups_without_precomputed",
            message="feature_groups in search space but no precomputed_features provided. "
            "Will use default features for all trials.",
        )

    # Check if top_k_features is being tuned
    tuning_top_k_features = "top_k_features" in search_spaces

    # Check if this is an LSTM strategy (feature selection not supported for 3D sequences)
    is_lstm_strategy = strategy_type == "lstm_line_movement"

    # Warn if LSTM + feature selection combination detected
    if is_lstm_strategy and tuning_top_k_features:
        logger.warning(
            "feature_selection_not_supported_for_lstm",
            strategy_type=strategy_type,
            message="top_k_features tuning is not supported for LSTM strategies. "
            "LSTM uses 3D sequences while feature selectors expect 2D data. "
            "LSTM gating mechanisms handle implicit feature weighting. "
            "Ignoring top_k_features parameter.",
        )

    logger.info(
        "objective_created",
        strategy_type=strategy_type,
        metric=metric_name,
        search_space_size=len(search_spaces),
        model_params=list(model_param_names),
        feature_params=list(feature_param_names & search_spaces.keys()),
        tuning_feature_groups=tuning_feature_groups,
        tuning_top_k_features=tuning_top_k_features,
        has_precomputed=precomputed_features is not None,
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

        # Split into model and feature parameters (excluding top_k_features)
        model_params = {k: v for k, v in suggested_params.items() if k in model_param_names}
        feature_params = {
            k: v
            for k, v in suggested_params.items()
            if k in feature_param_names and k != "top_k_features"
        }
        top_k = suggested_params.get("top_k_features")

        logger.info(
            "trial_started",
            trial_number=trial.number,
            model_params=model_params,
            feature_params=feature_params,
            top_k_features=top_k,
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

        # Select features: use pre-computed if available for feature_groups, otherwise default
        trial_X_train, trial_y_train = X_train, y_train
        trial_X_val, trial_y_val = X_val, y_val
        trial_feature_names = feature_names
        trial_static_train, trial_static_val = static_train, static_val

        if "feature_groups" in feature_params and precomputed_features is not None:
            fg_key = feature_params["feature_groups"]
            if fg_key in precomputed_features:
                data = precomputed_features[fg_key]
                trial_X_train = data.X_train
                trial_y_train = data.y_train
                trial_X_val = data.X_val if data.num_val_samples > 0 else data.X_test
                trial_y_val = data.y_val if data.num_val_samples > 0 else data.y_test
                trial_feature_names = data.feature_names
                trial_static_train = data.static_train
                trial_static_val = data.static_val if data.num_val_samples > 0 else data.static_test
                logger.debug(
                    "using_precomputed_features",
                    trial_number=trial.number,
                    feature_groups=fg_key,
                    n_features=len(trial_feature_names),
                )

        # Feature selection integration for top_k_features tuning
        # Skip for LSTM strategies - they use 3D sequences incompatible with 2D selectors
        if (
            tuning_top_k_features
            and config.training.feature_selection.enabled
            and not is_lstm_strategy
        ):
            # Run feature selection once on first trial and store rankings
            if trial.number == 0:
                from odds_analytics.training.feature_selection import get_feature_selector

                logger.info(
                    "running_feature_selection",
                    trial_number=trial.number,
                    method=config.training.feature_selection.method,
                    n_features=len(trial_feature_names),
                )

                # Run feature selection
                selector = get_feature_selector(config.training.feature_selection)
                ranking = selector.select(trial_X_train, trial_y_train, trial_feature_names)

                # Store rankings in study user attributes (as JSON)
                trial.study.set_user_attr("feature_ranking_names", ranking.feature_names)
                trial.study.set_user_attr("feature_ranking_scores", ranking.scores)
                trial.study.set_user_attr("feature_ranking_method", ranking.method)
                trial.study.set_user_attr("feature_ranking_metadata", ranking.metadata)

                logger.info(
                    "feature_selection_completed",
                    trial_number=trial.number,
                    method=ranking.method,
                    n_features=len(ranking.feature_names),
                    top_5_features=ranking.feature_names[:5],
                    top_5_scores=ranking.scores[:5],
                )

            # Retrieve stored rankings (available in all trials after first)
            ranking_names = trial.study.user_attrs.get("feature_ranking_names")

            if ranking_names and top_k:
                # Subset to top_k features
                selected_features = ranking_names[: int(top_k)]
                selected_indices = [trial_feature_names.index(f) for f in selected_features]

                # Apply subsetting to training and validation data
                trial_X_train = trial_X_train[:, selected_indices]
                if trial_X_val is not None:
                    trial_X_val = trial_X_val[:, selected_indices]
                trial_feature_names = selected_features

                logger.debug(
                    "applied_feature_subsetting",
                    trial_number=trial.number,
                    top_k=top_k,
                    n_features=len(trial_feature_names),
                    selected_features=selected_features[:10],  # Log first 10
                )

        # Import strategy classes
        if strategy_type in ("xgboost", "xgboost_line_movement"):
            from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy

            strategy = XGBoostLineMovementStrategy()
        elif strategy_type in ("lstm", "lstm_line_movement"):
            from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy

            strategy = LSTMLineMovementStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Check if we should use cross-validation
        use_cv = modified_config.training.data.use_kfold

        # Train with modified configuration
        try:
            if use_cv:
                # Use cross-validation for tuning
                from odds_analytics.training.cross_validation import run_cv

                # Combine train and validation sets for CV
                if trial_X_val is not None and trial_y_val is not None:
                    X_full = np.vstack([trial_X_train, trial_X_val])
                    y_full = np.concatenate([trial_y_train, trial_y_val])
                    static_full = (
                        np.vstack([trial_static_train, trial_static_val])
                        if trial_static_train is not None and trial_static_val is not None
                        else trial_static_train
                    )
                else:
                    X_full = trial_X_train
                    y_full = trial_y_train
                    static_full = trial_static_train

                logger.debug(
                    "using_cv_for_tuning",
                    trial_number=trial.number,
                    n_samples=len(X_full),
                    n_folds=modified_config.training.data.n_folds,
                )

                # Run cross-validation
                cv_result = run_cv(
                    strategy=strategy,
                    config=modified_config,
                    X=X_full,
                    y=y_full,
                    feature_names=trial_feature_names,
                    static_features=static_full,
                )

                # Log mean and std CV metrics if trial has set_user_attr (for MLflow logging)
                if hasattr(trial, "set_user_attr"):
                    trial.set_user_attr("mean_val_mse", cv_result.mean_val_mse)
                    trial.set_user_attr("std_val_mse", cv_result.std_val_mse)
                    trial.set_user_attr("mean_val_mae", cv_result.mean_val_mae)
                    trial.set_user_attr("std_val_mae", cv_result.std_val_mae)
                    trial.set_user_attr("mean_val_r2", cv_result.mean_val_r2)
                    trial.set_user_attr("std_val_r2", cv_result.std_val_r2)

                # Use mean CV metric as optimization target
                if metric_name == "val_mse":
                    metric_value = cv_result.mean_val_mse
                elif metric_name == "val_mae":
                    metric_value = cv_result.mean_val_mae
                elif metric_name == "val_r2":
                    metric_value = cv_result.mean_val_r2
                elif metric_name.startswith("cv_"):
                    # Direct CV metric access (e.g., cv_val_mse_mean)
                    cv_dict = cv_result.to_dict()
                    metric_value = cv_dict.get(metric_name)
                    if metric_value is None:
                        logger.warning(
                            "cv_metric_not_found",
                            trial_number=trial.number,
                            metric=metric_name,
                            available_metrics=list(cv_dict.keys()),
                        )
                        return (
                            float("inf") if config.tuning.direction == "minimize" else float("-inf")
                        )
                else:
                    logger.warning(
                        "unsupported_cv_metric",
                        trial_number=trial.number,
                        metric=metric_name,
                        message=f"Metric '{metric_name}' not directly available from CV. Using mean_val_mse as fallback.",
                    )
                    metric_value = cv_result.mean_val_mse

                logger.info(
                    "trial_completed_with_cv",
                    trial_number=trial.number,
                    metric=metric_name,
                    value=metric_value,
                    cv_std=cv_result.std_val_mse if metric_name == "val_mse" else None,
                )
            else:
                # Use simple train/validation split (backward compatibility)
                history = strategy.train_from_config(
                    config=modified_config,
                    X_train=trial_X_train,
                    y_train=trial_y_train,
                    feature_names=trial_feature_names,
                    X_val=trial_X_val,
                    y_val=trial_y_val,
                    trial=trial,  # Pass trial for pruning support
                    static_train=trial_static_train,
                    static_val=trial_static_val,
                )

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

        except ImportError:
            # Re-raise TrialPruned exception to let Optuna handle it
            import optuna

            if hasattr(optuna, "TrialPruned"):
                raise
            # If it's a different ImportError, treat as failure
            logger.error(
                "trial_failed",
                trial_number=trial.number,
                error="Import error during training",
            )
            return float("inf") if config.tuning.direction == "minimize" else float("-inf")
        except Exception as e:
            # Check if it's a TrialPruned exception (might not be ImportError)
            if e.__class__.__name__ == "TrialPruned":
                raise  # Re-raise for Optuna to handle
            logger.error(
                "trial_failed",
                trial_number=trial.number,
                error=str(e),
            )
            # Return worst possible value for failed trials
            return float("inf") if config.tuning.direction == "minimize" else float("-inf")

        return float(metric_value)

    return objective
