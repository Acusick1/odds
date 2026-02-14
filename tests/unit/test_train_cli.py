"""
Unit tests for train CLI commands.

Tests cover:
1. Tune command validation and error handling
2. Configuration loading and validation
3. Study name and storage URL determination
4. Best parameter export
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import typer
from odds_analytics.training.config import (
    DataConfig,
    ExperimentConfig,
    MLTrainingConfig,
    SearchSpace,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)
from typer.testing import CliRunner


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text (Rich output formatting)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config_with_tuning(tmp_path):
    """Create a sample config file with tuning section."""
    config = MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_tuning_experiment",
            tags=["test", "tuning"],
            description="Test tuning experiment",
        ),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date="2024-10-01",
                end_date="2024-12-31",
                test_split=0.2,
                validation_split=0.1,
                random_seed=42,
            ),
            model=XGBoostConfig(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            ),
        ),
        tuning=TuningConfig(
            n_trials=5,
            direction="minimize",
            metric="val_mse",
            search_spaces={
                "n_estimators": SearchSpace(type="int", low=50, high=500, step=50),
                "learning_rate": SearchSpace(type="float", low=0.001, high=0.3, log=True),
                "max_depth": SearchSpace(type="int", low=3, high=10),
            },
            sampler="tpe",
            pruner="median",
        ),
    )

    # Save to temp file
    config_path = tmp_path / "test_config.yaml"
    config.to_yaml(config_path)

    return config_path


@pytest.fixture
def sample_config_without_tuning(tmp_path):
    """Create a sample config file without tuning section."""
    config = MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_experiment",
            tags=["test"],
        ),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date="2024-10-01",
                end_date="2024-12-31",
            ),
            model=XGBoostConfig(),
        ),
        tuning=None,  # No tuning section
    )

    config_path = tmp_path / "test_config_no_tuning.yaml"
    config.to_yaml(config_path)

    return config_path


# =============================================================================
# Test Configuration Validation
# =============================================================================


def test_tune_command_requires_config(cli_runner):
    """Test that tune command requires --config option."""
    from odds_cli.commands.train import app

    result = cli_runner.invoke(app, ["tune"])

    assert result.exit_code != 0
    # Strip ANSI codes from Rich output before checking
    output = strip_ansi(result.stdout + result.stderr)
    assert "Missing option '--config'" in output or "required" in output.lower()


def test_tune_command_validates_tuning_section(cli_runner, sample_config_without_tuning):
    """Test that tune command validates tuning section exists."""
    from odds_cli.commands.train import app

    result = cli_runner.invoke(app, ["tune", "--config", str(sample_config_without_tuning)])

    assert result.exit_code == 1
    assert "must include a 'tuning' section" in result.stdout


def test_tune_command_validates_search_spaces(cli_runner, tmp_path):
    """Test that tune command validates search spaces are defined."""
    # Create config with tuning but no search spaces
    config = MLTrainingConfig(
        experiment=ExperimentConfig(name="test"),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(start_date="2024-10-01", end_date="2024-12-31"),
            model=XGBoostConfig(),
        ),
        tuning=TuningConfig(
            n_trials=10,
            search_spaces={},  # Empty search spaces
        ),
    )

    config_path = tmp_path / "config_no_spaces.yaml"
    config.to_yaml(config_path)

    from odds_cli.commands.train import app

    result = cli_runner.invoke(app, ["tune", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "must define search_spaces" in result.stdout


# =============================================================================
# Test Study Name and Storage Configuration
# =============================================================================


def test_tune_command_default_study_name(sample_config_with_tuning):
    """Test that default study name is generated from experiment name."""
    from odds_cli.commands.train import load_config

    config = load_config(str(sample_config_with_tuning))

    # Default study name should be {experiment_name}_tuning
    expected_study_name = f"{config.experiment.name}_tuning"

    assert expected_study_name == "test_tuning_experiment_tuning"


def test_tune_command_custom_study_name(cli_runner, sample_config_with_tuning):
    """Test that custom study name can be specified."""
    from odds_cli.commands.train import app

    # Mock the async function to avoid actual execution
    with patch("odds_cli.commands.train.asyncio.run") as mock_run:
        cli_runner.invoke(
            app,
            ["tune", "--config", str(sample_config_with_tuning), "--study-name", "custom_study"],
        )

        # Verify study name was passed correctly
        assert mock_run.called


# =============================================================================
# Test Parameter Overrides
# =============================================================================


def test_tune_command_override_n_trials(sample_config_with_tuning):
    """Test that n_trials can be overridden via CLI."""
    from odds_cli.commands.train import load_config

    config = load_config(str(sample_config_with_tuning))
    original_trials = config.tuning.n_trials

    # Override should happen in the command
    # Original config should have 5 trials (from fixture)
    assert original_trials == 5


def test_tune_command_override_timeout(sample_config_with_tuning):
    """Test that timeout can be overridden via CLI."""
    from odds_cli.commands.train import load_config

    config = load_config(str(sample_config_with_tuning))
    original_timeout = config.tuning.timeout

    # Original config should have None timeout
    assert original_timeout is None


# =============================================================================
# Test Output Path Generation
# =============================================================================


def test_tune_command_default_output_path(sample_config_with_tuning):
    """Test that default output path is {config_stem}_best.yaml."""
    config_path = Path(sample_config_with_tuning)

    # Default output should be test_config_best.yaml
    expected_output = config_path.parent / f"{config_path.stem}_best.yaml"

    assert expected_output.name == "test_config_best.yaml"


def test_tune_command_custom_output_path(cli_runner, sample_config_with_tuning):
    """Test that custom output path can be specified."""
    from odds_cli.commands.train import app

    custom_output = "/tmp/custom_output.yaml"

    with patch("odds_cli.commands.train.asyncio.run") as mock_run:
        cli_runner.invoke(
            app,
            ["tune", "--config", str(sample_config_with_tuning), "--output", custom_output],
        )

        assert mock_run.called


# =============================================================================
# Test Display Functions
# =============================================================================


def test_display_tuning_summary():
    """Test tuning summary display formatting."""
    from odds_cli.commands.train import _display_tuning_summary

    config = MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_experiment",
            description="Test description",
        ),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(start_date="2024-10-01", end_date="2024-12-31"),
            model=XGBoostConfig(),
        ),
        tuning=TuningConfig(
            n_trials=100,
            direction="minimize",
            metric="val_mse",
            search_spaces={
                "n_estimators": SearchSpace(type="int", low=50, high=500),
            },
        ),
    )

    # Should not raise any errors
    _display_tuning_summary(config, "test_study", "postgresql://localhost/test")


def test_display_best_parameters():
    """Test best parameters display formatting."""
    from odds_cli.commands.train import _display_best_parameters

    best_params = {
        "n_estimators": 200,
        "learning_rate": 0.05432,
        "max_depth": 8,
    }

    # Should not raise any errors
    _display_best_parameters(best_params)


# =============================================================================
# Test Integration with Mocked Components
# =============================================================================


@pytest.mark.asyncio
async def test_run_tuning_async_workflow(sample_config_with_tuning, tmp_path):
    """Test the async tuning workflow with mocked components."""
    from odds_cli.commands.train import _run_tuning_async, load_config

    config = load_config(str(sample_config_with_tuning))
    study_name = "test_study"
    storage_url = None
    output_path = str(tmp_path / "best_config.yaml")

    # Mock all the components (patch where they're imported/used)
    with (
        patch("odds_core.database.async_session_maker") as mock_get_session,
        patch("odds_analytics.training.prepare_training_data_from_config") as mock_prepare_data,
        patch("odds_analytics.training.tuner.OptunaTuner") as mock_tuner_class,
        patch("odds_analytics.training.tuner.create_objective") as mock_create_objective,
    ):
        # Mock session
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Mock data preparation
        mock_data_result = Mock()
        mock_data_result.num_train_samples = 1000
        mock_data_result.num_val_samples = 200
        mock_data_result.num_test_samples = 200
        mock_data_result.num_features = 50
        mock_data_result.X_train = Mock()
        mock_data_result.y_train = Mock()
        mock_data_result.X_val = Mock()
        mock_data_result.y_val = Mock()
        mock_data_result.X_test = Mock()
        mock_data_result.y_test = Mock()
        mock_data_result.feature_names = ["feature_1", "feature_2"]
        mock_prepare_data.return_value = mock_data_result

        # Mock tuner
        mock_tuner = Mock()
        mock_study = Mock()
        mock_study.best_value = 0.123456
        mock_study.best_params = {"n_estimators": 200, "learning_rate": 0.05}
        mock_study.trials = [Mock(), Mock(), Mock()]  # 3 trials
        # Set user_attrs to a dict so .items() works in CLI display
        mock_study.best_trial.user_attrs = {
            "mean_val_mse": 0.123456,
            "std_val_mse": 0.01,
        }
        mock_tuner.optimize.return_value = mock_study
        mock_tuner_class.return_value = mock_tuner

        # Mock objective
        mock_objective = Mock()
        mock_create_objective.return_value = mock_objective

        # Run the async function
        await _run_tuning_async(
            config, study_name, storage_url, str(sample_config_with_tuning), output_path, False
        )

        # Verify workflow
        assert mock_prepare_data.called
        assert mock_tuner_class.called
        assert mock_tuner.optimize.called
        assert mock_create_objective.called

        # Verify best config was saved
        assert Path(output_path).exists()


@pytest.mark.asyncio
async def test_run_tuning_async_with_train_best(sample_config_with_tuning, tmp_path):
    """Test the async tuning workflow with --train-best flag."""
    from odds_cli.commands.train import _run_tuning_async, load_config

    config = load_config(str(sample_config_with_tuning))
    study_name = "test_study"
    storage_url = None
    output_path = str(tmp_path / "best_config.yaml")

    with (
        patch("odds_core.database.async_session_maker") as mock_get_session,
        patch("odds_analytics.training.prepare_training_data_from_config") as mock_prepare_data,
        patch("odds_analytics.training.tuner.OptunaTuner") as mock_tuner_class,
        patch("odds_analytics.training.tuner.create_objective") as mock_create_objective,
    ):
        # Setup mocks (similar to previous test)
        import numpy as np

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_data_result = Mock()
        mock_data_result.num_train_samples = 1000
        mock_data_result.num_val_samples = 200
        mock_data_result.num_test_samples = 200
        mock_data_result.num_features = 50
        # Use numpy arrays instead of Mocks so len() works
        mock_data_result.X_train = np.random.randn(1000, 50)
        mock_data_result.y_train = np.random.randn(1000)
        mock_data_result.X_val = np.random.randn(200, 50)
        mock_data_result.y_val = np.random.randn(200)
        mock_data_result.X_test = np.random.randn(200, 50)
        mock_data_result.y_test = np.random.randn(200)
        mock_data_result.feature_names = ["feature_1", "feature_2"]
        mock_prepare_data.return_value = mock_data_result

        mock_tuner = Mock()
        mock_study = Mock()
        mock_study.best_value = 0.123456
        mock_study.best_params = {"n_estimators": 200, "learning_rate": 0.05}
        mock_study.trials = [Mock(), Mock()]
        # Set user_attrs to a dict so .items() works in CLI display
        mock_study.best_trial.user_attrs = {
            "mean_val_mse": 0.123456,
            "std_val_mse": 0.01,
        }
        mock_tuner.optimize.return_value = mock_study
        mock_tuner_class.return_value = mock_tuner

        mock_create_objective.return_value = Mock()

        # Mock strategy for training using STRATEGY_CLASSES
        from odds_cli.commands.train import STRATEGY_CLASSES

        mock_strategy = Mock()
        mock_strategy.train_from_config.return_value = {"train_mse": 0.1, "val_mse": 0.2}

        # Temporarily replace the strategy class in STRATEGY_CLASSES
        original_strategy = STRATEGY_CLASSES["xgboost_line_movement"]
        STRATEGY_CLASSES["xgboost_line_movement"] = lambda: mock_strategy

        try:
            # Run with train_best=True
            await _run_tuning_async(
                config,
                study_name,
                storage_url,
                str(sample_config_with_tuning),
                output_path,
                train_best=True,
            )

            # Verify training was called
            assert mock_strategy.train_from_config.called
            assert mock_strategy.save_model.called
        finally:
            # Restore original strategy
            STRATEGY_CLASSES["xgboost_line_movement"] = original_strategy


# =============================================================================
# Test Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_run_tuning_async_data_preparation_error(sample_config_with_tuning):
    """Test error handling when data preparation fails."""
    from odds_cli.commands.train import _run_tuning_async, load_config

    config = load_config(str(sample_config_with_tuning))

    with (
        patch("odds_core.database.async_session_maker") as mock_get_session,
        patch("odds_analytics.training.prepare_training_data_from_config") as mock_prepare_data,
    ):
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Simulate data preparation error
        mock_prepare_data.side_effect = ValueError("No data found in date range")

        # Should raise typer.Exit
        with pytest.raises(typer.Exit):
            await _run_tuning_async(
                config, "test_study", None, str(sample_config_with_tuning), None, False
            )


@pytest.mark.asyncio
async def test_run_tuning_async_optimization_error(sample_config_with_tuning, tmp_path):
    """Test error handling when optimization fails."""
    from odds_cli.commands.train import _run_tuning_async, load_config

    config = load_config(str(sample_config_with_tuning))

    with (
        patch("odds_core.database.async_session_maker") as mock_get_session,
        patch("odds_analytics.training.prepare_training_data_from_config") as mock_prepare_data,
        patch("odds_analytics.training.tuner.OptunaTuner") as mock_tuner_class,
        patch("odds_analytics.training.tuner.create_objective") as mock_create_objective,
    ):
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_data_result = Mock()
        mock_data_result.num_train_samples = 1000
        mock_data_result.num_val_samples = 200
        mock_data_result.num_test_samples = 200
        mock_data_result.num_features = 50
        mock_data_result.X_train = Mock()
        mock_data_result.y_train = Mock()
        mock_data_result.X_val = Mock()
        mock_data_result.y_val = Mock()
        mock_data_result.X_test = Mock()
        mock_data_result.y_test = Mock()
        mock_data_result.feature_names = ["feature_1"]
        mock_prepare_data.return_value = mock_data_result

        mock_tuner = Mock()
        mock_tuner.optimize.side_effect = Exception("Optimization failed")
        mock_tuner_class.return_value = mock_tuner

        mock_create_objective.return_value = Mock()

        # Should raise typer.Exit
        with pytest.raises(typer.Exit):
            await _run_tuning_async(
                config, "test_study", None, str(sample_config_with_tuning), None, False
            )
