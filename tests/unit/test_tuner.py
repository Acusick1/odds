"""
Unit tests for hyperparameter tuner.

Tests cover:
1. Search space parameter mapping (int, float, categorical)
2. PostgreSQL persistence
3. Proper training integration
4. Feature parameter handling
5. Pruning functionality
6. Graceful failure handling
7. Sampler and pruner creation
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from odds_analytics.training.config import (
    DataConfig,
    ExperimentConfig,
    MLTrainingConfig,
    SearchSpace,
    TrainingConfig,
    TuningConfig,
    XGBoostConfig,
)
from odds_analytics.training.tuner import (
    HyperparameterTuner,
    OptunaTuner,
    create_objective,
    suggest_params_from_search_space,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_search_spaces():
    """Create sample search spaces for testing."""
    return {
        "n_estimators": SearchSpace(type="int", low=50, high=500, step=50),
        "learning_rate": SearchSpace(type="float", low=0.001, high=0.3, log=True),
        "max_depth": SearchSpace(type="int", low=3, high=10),
        "objective": SearchSpace(
            type="categorical",
            choices=["reg:squarederror", "reg:absoluteerror"],
        ),
    }


@pytest.fixture
def sample_config(sample_search_spaces):
    """Create sample training configuration with search spaces."""
    return MLTrainingConfig(
        experiment=ExperimentConfig(
            name="test_experiment",
            tags=["test"],
            description="Test experiment",
        ),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date="2024-10-01",
                end_date="2024-12-31",
                test_split=0.2,
                random_seed=42,
            ),
            model=XGBoostConfig(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            ),
        ),
        tuning=TuningConfig(
            n_trials=10,
            direction="minimize",
            metric="val_mse",
            search_spaces=sample_search_spaces,
            sampler="tpe",
            pruner="median",
        ),
    )


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randn(20)
    feature_names = [f"feature_{i}" for i in range(10)]
    return X_train, y_train, X_val, y_val, feature_names


# =============================================================================
# Test Parameter Mapping
# =============================================================================


def test_suggest_params_int(sample_search_spaces):
    """Test integer parameter suggestion."""
    mock_trial = Mock()
    mock_trial.suggest_int.return_value = 200

    params = suggest_params_from_search_space(mock_trial, sample_search_spaces)

    # Should call suggest_int for n_estimators and max_depth
    assert mock_trial.suggest_int.call_count == 2
    assert "n_estimators" in params
    assert "max_depth" in params


def test_suggest_params_float(sample_search_spaces):
    """Test float parameter suggestion with log scale."""
    mock_trial = Mock()
    mock_trial.suggest_float.return_value = 0.05

    suggest_params_from_search_space(mock_trial, sample_search_spaces)

    # Should call suggest_float for learning_rate
    mock_trial.suggest_float.assert_called_once()
    call_args = mock_trial.suggest_float.call_args
    assert call_args[0][0] == "learning_rate"
    assert call_args[1]["log"] is True


def test_suggest_params_categorical(sample_search_spaces):
    """Test categorical parameter suggestion."""
    mock_trial = Mock()
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    suggest_params_from_search_space(mock_trial, sample_search_spaces)

    # Should call suggest_categorical for objective
    mock_trial.suggest_categorical.assert_called_once()
    call_args = mock_trial.suggest_categorical.call_args
    assert call_args[0][0] == "objective"
    assert "reg:squarederror" in call_args[1]["choices"]


def test_suggest_params_all_types(sample_search_spaces):
    """Test parameter suggestion for all types together."""
    mock_trial = Mock()
    mock_trial.suggest_int.return_value = 100
    mock_trial.suggest_float.return_value = 0.1
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    params = suggest_params_from_search_space(mock_trial, sample_search_spaces)

    # Should have all parameters
    assert len(params) == 4
    assert "n_estimators" in params
    assert "learning_rate" in params
    assert "max_depth" in params
    assert "objective" in params


# =============================================================================
# Test OptunaTuner
# =============================================================================


def test_optuna_tuner_initialization():
    """Test OptunaTuner initialization."""
    tuner = OptunaTuner(
        study_name="test_study",
        direction="minimize",
        sampler="tpe",
        pruner="median",
    )

    assert tuner.study_name == "test_study"
    assert tuner.direction == "minimize"
    assert tuner.study is None  # Study created on optimize


def test_optuna_tuner_sampler_creation():
    """Test sampler creation for different types."""
    # TPE sampler
    tuner_tpe = OptunaTuner(study_name="test", sampler="tpe")
    assert tuner_tpe.sampler is not None

    # Random sampler
    tuner_random = OptunaTuner(study_name="test", sampler="random")
    assert tuner_random.sampler is not None

    # CMA-ES sampler
    tuner_cmaes = OptunaTuner(study_name="test", sampler="cmaes")
    assert tuner_cmaes.sampler is not None

    # Unknown sampler (should default to TPE)
    tuner_unknown = OptunaTuner(study_name="test", sampler="unknown")
    assert tuner_unknown.sampler is not None


def test_optuna_tuner_pruner_creation():
    """Test pruner creation for different types."""
    # Median pruner
    tuner_median = OptunaTuner(study_name="test", pruner="median")
    assert tuner_median.pruner is not None

    # Hyperband pruner
    tuner_hyperband = OptunaTuner(study_name="test", pruner="hyperband")
    assert tuner_hyperband.pruner is not None

    # No pruner
    tuner_none = OptunaTuner(study_name="test", pruner="none")
    assert tuner_none.pruner is not None

    # Unknown pruner (should default to median)
    tuner_unknown = OptunaTuner(study_name="test", pruner="unknown")
    assert tuner_unknown.pruner is not None


def test_optuna_tuner_optimize():
    """Test optimization process."""
    tuner = OptunaTuner(
        study_name="test_optimize",
        direction="minimize",
        sampler="random",  # Use random for reproducibility
    )

    # Simple objective function
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return x**2

    # Run optimization
    study = tuner.optimize(objective, n_trials=5)

    assert study is not None
    assert len(study.trials) == 5
    assert study.best_value >= 0  # x^2 is always positive


def test_optuna_tuner_get_results():
    """Test getting results as DataFrame."""
    tuner = OptunaTuner(study_name="test_results")

    # Should raise error before optimization
    with pytest.raises(ValueError, match="No study found"):
        tuner.get_results()

    # Run optimization
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x

    tuner.optimize(objective, n_trials=3)

    # Should return DataFrame
    df = tuner.get_results()
    assert df is not None
    assert len(df) == 3
    assert "value" in df.columns


def test_optuna_tuner_get_best_params():
    """Test getting best parameters."""
    tuner = OptunaTuner(study_name="test_best_params", direction="minimize")

    # Should raise error before optimization
    with pytest.raises(ValueError, match="No study found"):
        tuner.get_best_params()

    # Run optimization
    def objective(trial):
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_int("y", 0, 10)
        return x**2 + y

    tuner.optimize(objective, n_trials=10)

    # Should return best params
    best_params = tuner.get_best_params()
    assert "x" in best_params
    assert "y" in best_params
    assert isinstance(best_params["y"], int)


def test_optuna_tuner_storage_persistence():
    """Test PostgreSQL storage persistence (mocked)."""
    storage_url = "postgresql://user:pass@localhost/test_db"

    tuner = OptunaTuner(
        study_name="test_storage",
        direction="minimize",
        storage=storage_url,
    )

    assert tuner.storage == storage_url


# =============================================================================
# Test Objective Function Factory
# =============================================================================


def test_create_objective_no_search_spaces():
    """Test objective creation fails without search spaces."""
    config = MLTrainingConfig(
        experiment=ExperimentConfig(name="test"),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(start_date="2024-10-01", end_date="2024-12-31"),
            model=XGBoostConfig(),
        ),
        tuning=None,  # No tuning config
    )

    X_train = np.random.randn(10, 5)
    y_train = np.random.randn(10)

    with pytest.raises(ValueError, match="No search spaces defined"):
        create_objective(config, X_train, y_train, ["f1", "f2", "f3", "f4", "f5"])


def test_create_objective_with_search_spaces(sample_config, sample_training_data):
    """Test objective function creation with search spaces."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    objective = create_objective(
        config=sample_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    assert callable(objective)


@patch("odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy")
def test_objective_function_execution(mock_strategy_class, sample_config, sample_training_data):
    """Test objective function execution with mocked training."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Mock strategy instance
    mock_strategy = MagicMock()
    mock_strategy.train_from_config.return_value = {
        "val_mse": 0.5,
        "train_mse": 0.4,
    }
    mock_strategy_class.return_value = mock_strategy

    objective = create_objective(
        config=sample_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    # Mock trial
    mock_trial = Mock()
    mock_trial.number = 0
    mock_trial.suggest_int.side_effect = [200, 6]  # n_estimators, max_depth
    mock_trial.suggest_float.return_value = 0.05  # learning_rate
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    # Execute objective
    result = objective(mock_trial)

    # Should return the metric value
    assert result == 0.5
    mock_strategy.train_from_config.assert_called_once()


@patch("odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy")
def test_objective_function_handles_training_failure(
    mock_strategy_class, sample_config, sample_training_data
):
    """Test objective function handles training failures gracefully."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Mock strategy that raises exception
    mock_strategy = MagicMock()
    mock_strategy.train_from_config.side_effect = Exception("Training failed")
    mock_strategy_class.return_value = mock_strategy

    objective = create_objective(
        config=sample_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    # Mock trial
    mock_trial = Mock()
    mock_trial.number = 0
    mock_trial.suggest_int.side_effect = [200, 6]
    mock_trial.suggest_float.return_value = 0.05
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    # Execute objective - should return inf for minimize direction
    result = objective(mock_trial)

    assert result == float("inf")  # Bad value for failed trial


@patch("odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy")
def test_objective_function_metric_not_found(
    mock_strategy_class, sample_config, sample_training_data
):
    """Test objective function when metric not found in history."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Mock strategy that returns history without the expected metric
    mock_strategy = MagicMock()
    mock_strategy.train_from_config.return_value = {
        "train_mse": 0.4,
        # Missing val_mse
    }
    mock_strategy_class.return_value = mock_strategy

    objective = create_objective(
        config=sample_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    # Mock trial
    mock_trial = Mock()
    mock_trial.number = 0
    mock_trial.suggest_int.side_effect = [200, 6]
    mock_trial.suggest_float.return_value = 0.05
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    # Execute objective - should return inf when metric not found
    result = objective(mock_trial)

    assert result == float("inf")


# =============================================================================
# Test Integration
# =============================================================================


def test_full_optimization_workflow(sample_config, sample_training_data):
    """Test full optimization workflow (integration test)."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Create tuner
    tuner = OptunaTuner(
        study_name="test_full_workflow",
        direction="minimize",
        sampler="random",
    )

    # Patch XGBoost training to avoid actual model training
    with patch("odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy") as mock_class:
        mock_strategy = MagicMock()
        mock_strategy.train_from_config.return_value = {
            "val_mse": np.random.uniform(0.1, 1.0),
            "train_mse": 0.3,
        }
        mock_class.return_value = mock_strategy

        # Create objective
        objective = create_objective(
            config=sample_config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_val,
            y_val=y_val,
        )

        # Run optimization
        study = tuner.optimize(objective, n_trials=5)

        # Verify results
        assert study is not None
        assert len(study.trials) == 5

        # Get best params
        best_params = tuner.get_best_params()
        assert "n_estimators" in best_params
        assert "learning_rate" in best_params

        # Get results DataFrame
        df = tuner.get_results()
        assert len(df) == 5


# =============================================================================
# Test Abstract Base Class
# =============================================================================


def test_hyperparameter_tuner_is_abstract():
    """Test that HyperparameterTuner cannot be instantiated."""
    with pytest.raises(TypeError):
        HyperparameterTuner()


def test_optuna_tuner_implements_interface():
    """Test that OptunaTuner implements all abstract methods."""
    tuner = OptunaTuner(study_name="test")

    # Should have all required methods
    assert hasattr(tuner, "optimize")
    assert hasattr(tuner, "get_results")
    assert hasattr(tuner, "get_best_params")
    assert callable(tuner.optimize)
    assert callable(tuner.get_results)
    assert callable(tuner.get_best_params)


# =============================================================================
# Test Error Handling
# =============================================================================


def test_optuna_not_installed():
    """Test graceful handling when Optuna is not installed."""
    with patch.dict("sys.modules", {"optuna": None}):
        with pytest.raises(ImportError, match="optuna not installed"):
            OptunaTuner(study_name="test")


# =============================================================================
# Test Pruning Integration
# =============================================================================


@patch("odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy")
def test_pruning_trial_passed_to_training(mock_strategy_class, sample_config, sample_training_data):
    """Test that trial object is passed to training for pruning."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Mock strategy instance
    mock_strategy = MagicMock()
    mock_strategy.train_from_config.return_value = {
        "val_mse": 0.5,
        "train_mse": 0.4,
    }
    mock_strategy_class.return_value = mock_strategy

    objective = create_objective(
        config=sample_config,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        X_val=X_val,
        y_val=y_val,
    )

    # Mock trial
    mock_trial = Mock()
    mock_trial.number = 0
    mock_trial.suggest_int.side_effect = [200, 6]
    mock_trial.suggest_float.return_value = 0.05
    mock_trial.suggest_categorical.return_value = "reg:squarederror"

    # Execute objective
    objective(mock_trial)

    # Verify trial was passed to train_from_config
    call_kwargs = mock_strategy.train_from_config.call_args[1]
    assert "trial" in call_kwargs
    assert call_kwargs["trial"] == mock_trial


# =============================================================================
# Test Feature Parameter Re-extraction
# =============================================================================


def test_feature_config_hash_stability():
    """Test that feature config hash is stable for same parameters."""
    from odds_analytics.training.tuner import _compute_feature_config_hash

    params1 = {"normalize": True, "movement_threshold": 0.01}
    params2 = {"movement_threshold": 0.01, "normalize": True}  # Different order

    hash1 = _compute_feature_config_hash(params1)
    hash2 = _compute_feature_config_hash(params2)

    # Hashes should be identical regardless of parameter order
    assert hash1 == hash2


def test_feature_config_hash_uniqueness():
    """Test that different feature configs produce different hashes."""
    from odds_analytics.training.tuner import _compute_feature_config_hash

    params1 = {"normalize": True, "movement_threshold": 0.01}
    params2 = {"normalize": True, "movement_threshold": 0.02}  # Different value

    hash1 = _compute_feature_config_hash(params1)
    hash2 = _compute_feature_config_hash(params2)

    # Hashes should be different for different configurations
    assert hash1 != hash2


def test_feature_groups_warning_without_precomputed(sample_config, sample_training_data):
    """Test warning when feature_groups in search space but no precomputed_features provided."""
    X_train, y_train, X_val, y_val, feature_names = sample_training_data

    # Add feature_groups to search space
    from odds_analytics.training.config import SearchSpace

    sample_config.tuning.search_spaces["feature_groups"] = SearchSpace(
        type="categorical", choices=[["tabular"], ["tabular", "trajectory"]]
    )

    # Create objective without precomputed_features
    with patch("odds_analytics.training.tuner.logger") as mock_logger:
        create_objective(
            config=sample_config,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            X_val=X_val,
            y_val=y_val,
            precomputed_features=None,  # No precomputed features
        )

        # Should have logged a warning
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert call_args[0] == "feature_groups_without_precomputed"
