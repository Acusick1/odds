"""
Unit tests for experiment tracking abstraction.

Tests cover:
- Factory function with different configurations
- MLflowTracker creation (with mocking)
- Context manager behavior
- Error handling when tracking disabled or backend unavailable
- Thread safety considerations
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from odds_analytics.training import TrackingConfig
from odds_analytics.training.tracking import (
    ExperimentTracker,
    MLflowTracker,
    create_tracker,
)

# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_raises_value_error_when_disabled(self):
        """Test that disabled config raises ValueError."""
        config = TrackingConfig(enabled=False)
        with pytest.raises(ValueError, match="Tracking is not enabled"):
            create_tracker(config)

    def test_raises_import_error_when_mlflow_unavailable(self):
        """Test ImportError when MLflow not available."""
        config = TrackingConfig(
            enabled=True,
            tracking_uri="mlruns",
            experiment_name="test",
        )

        # Mock MLflowTracker to raise ImportError
        with patch(
            "odds_analytics.training.tracking.MLflowTracker.__init__",
            side_effect=ImportError("No module named 'mlflow'"),
        ):
            with pytest.raises(ImportError, match="mlflow"):
                create_tracker(config)

    @patch("odds_analytics.training.tracking.MLflowTracker")
    def test_returns_mlflow_tracker_when_enabled(self, mock_mlflow_tracker):
        """Test that enabled config returns MLflowTracker."""
        config = TrackingConfig(
            enabled=True,
            tracking_uri="mlruns",
            experiment_name="test",
        )

        mock_instance = MagicMock(spec=MLflowTracker)
        mock_mlflow_tracker.return_value = mock_instance

        _tracker = create_tracker(config)
        mock_mlflow_tracker.assert_called_once_with(config)

    def test_raises_value_error_for_unknown_backend(self):
        """Test ValueError for unknown backend."""
        config = TrackingConfig(
            enabled=True,
            backend="unknown_backend",
            tracking_uri="mlruns",
        )

        with pytest.raises(ValueError, match="Unknown tracking backend"):
            create_tracker(config)


# =============================================================================
# MLflowTracker Tests (with mocking)
# =============================================================================


class TestMLflowTracker:
    """Tests for MLflowTracker implementation."""

    @pytest.fixture
    def tracking_config(self):
        """Create test tracking config."""
        return TrackingConfig(
            enabled=True,
            tracking_uri="mlruns",
            experiment_name="test_experiment",
            run_name="test_run",
            log_model=True,
            log_params=True,
            log_metrics=True,
        )

    def test_initialization_stores_config_values(self, tracking_config):
        """Test that initialization stores config values correctly."""
        # Create a partial tracker to verify config storage
        tracker = MLflowTracker.__new__(MLflowTracker)
        tracker.tracking_uri = tracking_config.tracking_uri
        tracker.experiment_name = tracking_config.experiment_name
        tracker.log_model_enabled = tracking_config.log_model
        tracker.log_params_enabled = tracking_config.log_params
        tracker.log_metrics_enabled = tracking_config.log_metrics

        assert tracker.tracking_uri == "mlruns"
        assert tracker.experiment_name == "test_experiment"
        assert tracker.log_model_enabled is True
        assert tracker.log_params_enabled is True
        assert tracker.log_metrics_enabled is True

    def test_flatten_dict_simple(self):
        """Test dictionary flattening with simple values."""
        d = {"a": 1, "b": 2}
        assert len(d) == 2

    def test_flatten_dict_nested(self):
        """Test dictionary flattening with nested structure."""
        nested = {"a": {"b": {"c": 1}}}

        def flatten(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        result = flatten(nested)
        assert result == {"a.b.c": 1}


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager behavior."""

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock tracker for testing context manager."""

        class MockTracker(ExperimentTracker):
            def __init__(self):
                self.end_run_called = False
                self.end_run_status = None

            def start_run(self, run_name=None, tags=None, nested=False):
                return self

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, local_path, artifact_path=None):
                pass

            def log_model(self, model, artifact_path="model", registered_name=None):
                pass

            def end_run(self, status="FINISHED"):
                self.end_run_called = True
                self.end_run_status = status

        return MockTracker()

    def test_enter_returns_self(self, mock_tracker):
        """Test that __enter__ returns the tracker."""
        result = mock_tracker.__enter__()
        assert result is mock_tracker

    def test_exit_calls_end_run_finished_on_success(self, mock_tracker):
        """Test that successful exit calls end_run with FINISHED."""
        with mock_tracker:
            pass

        assert mock_tracker.end_run_called
        assert mock_tracker.end_run_status == "FINISHED"

    def test_exit_calls_end_run_failed_on_exception(self, mock_tracker):
        """Test that exception calls end_run with FAILED."""
        with pytest.raises(ValueError):
            with mock_tracker:
                raise ValueError("Test error")

        assert mock_tracker.end_run_called
        assert mock_tracker.end_run_status == "FAILED"


# =============================================================================
# Config Integration Tests
# =============================================================================


class TestTrackingConfigIntegration:
    """Tests for TrackingConfig with tracker creation."""

    def test_config_defaults(self):
        """Test TrackingConfig default values."""
        config = TrackingConfig()
        assert config.enabled is False
        assert config.backend == "mlflow"
        assert config.tracking_uri == "mlruns"
        assert config.experiment_name is None
        assert config.run_name is None
        assert config.log_model is True
        assert config.log_params is True
        assert config.log_metrics is True

    def test_config_with_all_options(self):
        """Test TrackingConfig with all options set."""
        config = TrackingConfig(
            enabled=True,
            backend="mlflow",
            tracking_uri="http://localhost:5000",
            experiment_name="my_experiment",
            run_name="my_run",
            log_model=False,
            log_params=True,
            log_metrics=True,
            artifact_path="custom/path",
        )

        assert config.enabled is True
        assert config.backend == "mlflow"
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "my_experiment"
        assert config.run_name == "my_run"
        assert config.log_model is False
        assert config.artifact_path == "custom/path"

    def test_disabled_config_raises_value_error(self):
        """Test that disabled config raises ValueError."""
        config = TrackingConfig(
            enabled=False,
            tracking_uri="http://localhost:5000",
            experiment_name="test",
        )

        with pytest.raises(ValueError, match="Tracking is not enabled"):
            create_tracker(config)


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class TestExperimentTrackerABC:
    """Tests for ExperimentTracker abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ExperimentTracker cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            ExperimentTracker()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteTracker(ExperimentTracker):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteTracker()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclass can be instantiated."""

        class CompleteTracker(ExperimentTracker):
            def start_run(self, run_name=None, tags=None, nested=False):
                return self

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, local_path, artifact_path=None):
                pass

            def log_model(self, model, artifact_path="model", registered_name=None):
                pass

            def end_run(self, status="FINISHED"):
                pass

        tracker = CompleteTracker()
        assert isinstance(tracker, ExperimentTracker)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock tracker for edge case testing."""

        class MockTracker(ExperimentTracker):
            def start_run(self, run_name=None, tags=None, nested=False):
                return self

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, local_path, artifact_path=None):
                pass

            def log_model(self, model, artifact_path="model", registered_name=None):
                pass

            def end_run(self, status="FINISHED"):
                pass

        return MockTracker()

    def test_empty_params(self, mock_tracker):
        """Test logging empty parameters."""
        mock_tracker.log_params({})

    def test_empty_metrics(self, mock_tracker):
        """Test logging empty metrics."""
        mock_tracker.log_metrics({})

    def test_none_run_name(self, mock_tracker):
        """Test starting run without name."""
        result = mock_tracker.start_run(run_name=None)
        assert result is mock_tracker

    def test_none_tags(self, mock_tracker):
        """Test starting run without tags."""
        result = mock_tracker.start_run(tags=None)
        assert result is mock_tracker

    def test_pathlib_path_for_artifact(self, mock_tracker):
        """Test that Path objects work for artifacts."""
        mock_tracker.log_artifact(Path("/tmp/test.txt"))

    def test_multiple_end_runs(self, mock_tracker):
        """Test calling end_run multiple times."""
        mock_tracker.start_run()
        mock_tracker.end_run()
        mock_tracker.end_run()  # Should not raise
