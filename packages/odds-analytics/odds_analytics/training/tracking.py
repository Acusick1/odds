"""
Experiment Tracking Abstraction Layer.

This module provides a backend-agnostic interface for experiment tracking,
with initial support for MLflow and a no-op fallback for development/testing.

Key Features:
- Abstract base class defining the tracking interface
- MLflow 2.x implementation with autologging support
- NullTracker for graceful degradation when tracking is disabled
- Factory function for configuration-based tracker creation
- Context manager support for automatic run lifecycle management
- Thread-safe design for parallel hyperparameter tuning

Example usage:
    ```python
    from odds_analytics.training import TrackingConfig
    from odds_analytics.training.tracking import create_tracker

    # Create tracker from config
    config = TrackingConfig(
        enabled=True,
        tracking_uri="mlruns",
        experiment_name="xgboost_line_movement",
    )

    tracker = create_tracker(config)

    # Use as context manager
    with tracker.start_run(run_name="experiment_v1"):
        tracker.log_params({"learning_rate": 0.1, "n_estimators": 100})

        for epoch in range(10):
            loss = train_epoch()
            tracker.log_metrics({"loss": loss}, step=epoch)

        tracker.log_model(model, artifact_path="model")

    # Or manual lifecycle management
    tracker.start_run(run_name="experiment_v2")
    try:
        # training code
        tracker.end_run(status="FINISHED")
    except Exception:
        tracker.end_run(status="FAILED")
        raise
    ```
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

from odds_analytics.training.config import TrackingConfig

logger = structlog.get_logger()

__all__ = [
    "ExperimentTracker",
    "MLflowTracker",
    "NullTracker",
    "create_tracker",
]


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking backends.

    Defines the interface for logging parameters, metrics, artifacts,
    and models during ML experiment runs. Implementations should be
    thread-safe for parallel hyperparameter tuning scenarios.

    Supports context manager protocol for automatic run lifecycle management.
    """

    @abstractmethod
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> ExperimentTracker:
        """
        Initialize an experiment tracking run.

        Args:
            run_name: Optional name for this run
            tags: Optional tags for categorization
            nested: If True, create a nested run under current active run

        Returns:
            Self for chaining or context manager usage
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log configuration parameters for the experiment.

        Args:
            params: Dictionary of parameter names to values.
                    Values will be converted to strings for storage.
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for time-series metrics (e.g., epoch)
        """
        pass

    @abstractmethod
    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log a file as an artifact.

        Args:
            local_path: Path to the local file to log
            artifact_path: Optional destination path within the artifact store
        """
        pass

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> None:
        """
        Log a trained model.

        Args:
            model: The trained model object
            artifact_path: Path within artifact store for the model
            registered_name: Optional name for model registry
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Run status - "FINISHED", "FAILED", or "KILLED"
        """
        pass

    def __enter__(self) -> ExperimentTracker:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit with automatic status handling."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")


class MLflowTracker(ExperimentTracker):
    """
    MLflow implementation of experiment tracking.

    Uses MLflow 2.x APIs for logging parameters, metrics, artifacts,
    and models. Supports nested runs for hyperparameter tuning.

    Thread-safe for parallel trial execution via MLflow's internal locking.

    Attributes:
        tracking_uri: MLflow tracking server URI or local directory
        experiment_name: Name of the MLflow experiment
        log_model_enabled: Whether to log models
        log_params_enabled: Whether to log parameters
        log_metrics_enabled: Whether to log metrics
    """

    def __init__(self, config: TrackingConfig) -> None:
        """
        Initialize MLflow tracker from configuration.

        Args:
            config: TrackingConfig with MLflow settings

        Raises:
            ImportError: If mlflow is not installed
        """
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow is required for MLflowTracker. Install with: pip install mlflow"
            ) from e

        self.tracking_uri = config.tracking_uri
        self.experiment_name = config.experiment_name
        self.log_model_enabled = config.log_model
        self.log_params_enabled = config.log_params
        self.log_metrics_enabled = config.log_metrics
        self.artifact_path = config.artifact_path

        # Thread lock for thread-safety
        self._lock = threading.Lock()
        self._active_run = None

        # Configure MLflow
        self._mlflow.set_tracking_uri(self.tracking_uri)

        logger.info(
            "mlflow_tracker_initialized",
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> MLflowTracker:
        """
        Start an MLflow run.

        Args:
            run_name: Optional name for this run
            tags: Optional tags for categorization
            nested: If True, create nested run under current active run

        Returns:
            Self for chaining
        """
        with self._lock:
            # Set experiment if specified
            if self.experiment_name:
                self._mlflow.set_experiment(self.experiment_name)

            # Start the run
            self._active_run = self._mlflow.start_run(
                run_name=run_name,
                tags=tags,
                nested=nested,
            )

            logger.info(
                "mlflow_run_started",
                run_id=self._active_run.info.run_id,
                run_name=run_name,
                experiment_name=self.experiment_name,
                nested=nested,
            )

        return self

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Converts all values to strings as required by MLflow.
        Handles nested dictionaries by flattening with dot notation.

        Args:
            params: Dictionary of parameter names to values
        """
        if not self.log_params_enabled:
            return

        with self._lock:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)

            # MLflow has a limit on param name/value length
            # Truncate if necessary
            truncated_params = {}
            for key, value in flat_params.items():
                key_str = str(key)[:250]  # MLflow key limit
                value_str = str(value)[:500]  # MLflow value limit
                truncated_params[key_str] = value_str

            self._mlflow.log_params(truncated_params)

            logger.debug(
                "mlflow_params_logged",
                param_count=len(truncated_params),
            )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to float values
            step: Optional step number for time-series tracking
        """
        if not self.log_metrics_enabled:
            return

        with self._lock:
            self._mlflow.log_metrics(metrics, step=step)

            logger.debug(
                "mlflow_metrics_logged",
                metric_count=len(metrics),
                step=step,
            )

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log a file artifact to MLflow.

        Args:
            local_path: Path to the local file
            artifact_path: Optional destination path in artifact store
        """
        with self._lock:
            local_path = Path(local_path)

            if not local_path.exists():
                logger.warning(
                    "mlflow_artifact_not_found",
                    local_path=str(local_path),
                )
                return

            self._mlflow.log_artifact(str(local_path), artifact_path)

            logger.debug(
                "mlflow_artifact_logged",
                local_path=str(local_path),
                artifact_path=artifact_path,
            )

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> None:
        """
        Log a trained model to MLflow.

        Automatically detects model type and uses appropriate flavor:
        - XGBoost models: mlflow.xgboost
        - PyTorch models: mlflow.pytorch
        - Scikit-learn models: mlflow.sklearn

        Args:
            model: The trained model object
            artifact_path: Path within artifact store
            registered_name: Optional name for model registry
        """
        if not self.log_model_enabled:
            return

        with self._lock:
            # Use custom artifact path if configured
            if self.artifact_path:
                artifact_path = self.artifact_path

            # Detect model type and use appropriate flavor
            model_type = type(model).__module__

            if "xgboost" in model_type:
                try:
                    import mlflow.xgboost

                    mlflow.xgboost.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_name,
                    )
                    logger.info(
                        "mlflow_xgboost_model_logged",
                        artifact_path=artifact_path,
                        registered_name=registered_name,
                    )
                except ImportError:
                    self._log_model_generic(model, artifact_path, registered_name)

            elif "torch" in model_type:
                try:
                    import mlflow.pytorch

                    mlflow.pytorch.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_name,
                    )
                    logger.info(
                        "mlflow_pytorch_model_logged",
                        artifact_path=artifact_path,
                        registered_name=registered_name,
                    )
                except ImportError:
                    self._log_model_generic(model, artifact_path, registered_name)

            elif "sklearn" in model_type:
                try:
                    import mlflow.sklearn

                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_name,
                    )
                    logger.info(
                        "mlflow_sklearn_model_logged",
                        artifact_path=artifact_path,
                        registered_name=registered_name,
                    )
                except ImportError:
                    self._log_model_generic(model, artifact_path, registered_name)

            else:
                self._log_model_generic(model, artifact_path, registered_name)

    def _log_model_generic(
        self,
        model: Any,
        artifact_path: str,
        registered_name: str | None,
    ) -> None:
        """
        Log model using generic pyfunc flavor.

        Fallback when specific flavor is not available.
        """
        import mlflow.pyfunc

        # Create a simple wrapper for generic models
        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model: Any) -> None:
                self.model = model

            def predict(self, context: Any, model_input: Any) -> Any:
                if hasattr(self.model, "predict"):
                    return self.model.predict(model_input)
                return None

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=ModelWrapper(model),
            registered_model_name=registered_name,
        )

        logger.info(
            "mlflow_generic_model_logged",
            artifact_path=artifact_path,
            registered_name=registered_name,
            model_type=type(model).__name__,
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Run status - "FINISHED", "FAILED", or "KILLED"
        """
        with self._lock:
            if self._active_run:
                run_id = self._active_run.info.run_id
                self._mlflow.end_run(status=status)
                self._active_run = None

                logger.info(
                    "mlflow_run_ended",
                    run_id=run_id,
                    status=status,
                )

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """
        Flatten a nested dictionary using dot notation.

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for keys
            sep: Separator between nested keys

        Returns:
            Flattened dictionary
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class NullTracker(ExperimentTracker):
    """
    No-operation tracker for when tracking is disabled.

    Implements all interface methods as no-ops, allowing code to use
    the tracking interface without conditional checks everywhere.
    Thread-safe by design (no shared state).

    Useful for:
    - Development/debugging without tracking overhead
    - Testing without external dependencies
    - Graceful degradation when MLflow is unavailable
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        """
        Initialize NullTracker.

        Args:
            config: Optional config (ignored, for interface compatibility)
        """
        logger.debug("null_tracker_initialized")

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> NullTracker:
        """No-op start run."""
        logger.debug(
            "null_tracker_start_run",
            run_name=run_name,
            nested=nested,
        )
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        """No-op log params."""
        logger.debug(
            "null_tracker_log_params",
            param_count=len(params),
        )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """No-op log metrics."""
        logger.debug(
            "null_tracker_log_metrics",
            metric_count=len(metrics),
            step=step,
        )

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """No-op log artifact."""
        logger.debug(
            "null_tracker_log_artifact",
            local_path=str(local_path),
        )

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> None:
        """No-op log model."""
        logger.debug(
            "null_tracker_log_model",
            model_type=type(model).__name__,
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """No-op end run."""
        logger.debug(
            "null_tracker_end_run",
            status=status,
        )


def create_tracker(config: TrackingConfig | None = None) -> ExperimentTracker:
    """
    Factory function to create the appropriate tracker based on configuration.

    Creates an MLflowTracker if tracking is enabled and MLflow is available,
    otherwise returns a NullTracker for graceful degradation.

    Args:
        config: TrackingConfig specifying tracker settings.
                If None or tracking disabled, returns NullTracker.

    Returns:
        ExperimentTracker instance (MLflowTracker or NullTracker)

    Example:
        ```python
        from odds_analytics.training.config import TrackingConfig
        from odds_analytics.training.tracking import create_tracker

        # With tracking enabled
        config = TrackingConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="my_experiment",
        )
        tracker = create_tracker(config)  # Returns MLflowTracker

        # With tracking disabled
        config = TrackingConfig(enabled=False)
        tracker = create_tracker(config)  # Returns NullTracker

        # No config provided
        tracker = create_tracker()  # Returns NullTracker
        ```
    """
    # Return NullTracker if no config or tracking disabled
    if config is None or not config.enabled:
        logger.info(
            "creating_null_tracker",
            reason="tracking_disabled" if config else "no_config",
        )
        return NullTracker(config)

    # Try to create MLflowTracker
    try:
        tracker = MLflowTracker(config)
        logger.info(
            "creating_mlflow_tracker",
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
        )
        return tracker
    except ImportError as e:
        # Graceful degradation when MLflow not available
        logger.warning(
            "mlflow_not_available",
            error=str(e),
            fallback="null_tracker",
        )
        return NullTracker(config)
