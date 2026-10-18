"""Tests that training and tuning fail loudly instead of silently degrading (#437)."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import optuna
import pytest
import typer
import yaml
from odds_analytics.feature_groups import PreparedFeatureData
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrainingConfig,
    XGBoostConfig,
    prepare_training_data_from_config,
    run_cv,
)
from odds_analytics.training.config import (
    SearchSpace,
    StrategyType,
    TuningConfig,
)
from odds_analytics.training.cross_validation import train_with_cv
from odds_analytics.training.data_preparation import EvalData, TrainingDataResult
from odds_analytics.training.tuner import create_objective
from odds_core.models import Event, EventStatus
from pydantic import ValidationError


def make_config(
    *,
    search_spaces: dict[str, SearchSpace] | None = None,
    metric: str = "val_mse",
    direction: str | None = None,
    use_kfold: bool = False,
    cv_method: str = "timeseries",
    test_split: float = 0.2,
    strategy_type: str = "xgboost_line_movement",
) -> MLTrainingConfig:
    data_kwargs: dict = {}
    if cv_method == "walk_forward":
        data_kwargs.update(min_train_events=4, val_step_events=2)
    elif use_kfold:
        data_kwargs.update(n_folds=3)
    tuning_kwargs: dict = {"metric": metric, "search_spaces": search_spaces or {}}
    if direction is not None:
        tuning_kwargs["direction"] = direction
    return MLTrainingConfig(
        experiment=ExperimentConfig(name="t"),
        training=TrainingConfig(
            strategy_type=strategy_type,
            data=DataConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                use_kfold=use_kfold,
                cv_method=cv_method,
                test_split=test_split,
                **data_kwargs,
            ),
            model=XGBoostConfig(n_estimators=10),
        ),
        tuning=TuningConfig(**tuning_kwargs),
    )


N_ESTIMATORS = {"n_estimators": SearchSpace(type="int", low=10, high=50)}


class TestStudyStorage:
    """`odds train tune` storage comes from settings; failures are not swallowed."""

    def test_defaults_to_database_url_with_sync_driver(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        settings = SimpleNamespace(
            database=SimpleNamespace(url="postgresql+asyncpg://u:p@host:5432/db")
        )
        with patch("odds_core.config.get_settings", return_value=settings):
            assert _resolve_study_storage(None, False) == "postgresql+psycopg://u:p@host:5432/db"

    def test_settings_failure_propagates(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        with patch("odds_core.config.get_settings", side_effect=RuntimeError("no settings")):
            with pytest.raises(RuntimeError, match="no settings"):
                _resolve_study_storage(None, False)

    def test_unsupported_scheme_is_rejected(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        settings = SimpleNamespace(database=SimpleNamespace(url="sqlite:///x.db"))
        with patch("odds_core.config.get_settings", return_value=settings):
            with pytest.raises(typer.BadParameter):
                _resolve_study_storage(None, False)

    def test_in_memory_is_explicit(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        assert _resolve_study_storage(None, True) is None

    def test_in_memory_conflicts_with_storage(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        with pytest.raises(typer.BadParameter):
            _resolve_study_storage("postgresql://x", True)

    def test_explicit_storage_wins(self) -> None:
        from odds_cli.commands.train import _resolve_study_storage

        assert _resolve_study_storage("sqlite:///s.db", False) == "sqlite:///s.db"


class TestObjectiveErrors:
    """Errors in a trial fail the trial; pruning still works."""

    @staticmethod
    def _run(strategy: MagicMock, n_trials: int = 3) -> optuna.Study:
        config = make_config(search_spaces=N_ESTIMATORS)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        objective = create_objective(
            config=config,
            X_train=X,
            y_train=y,
            feature_names=["a", "b", "c"],
            X_val=X[:5],
            y_val=y[:5],
        )
        study = optuna.create_study()
        with patch(
            "odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy",
            return_value=strategy,
        ):
            study.optimize(objective, n_trials=n_trials)
        return study

    def test_error_marks_trial_failed_and_aborts_study(self) -> None:
        strategy = MagicMock()
        strategy.train_from_config.side_effect = RuntimeError("boom")
        study = optuna.create_study()
        config = make_config(search_spaces=N_ESTIMATORS)
        objective = create_objective(
            config=config,
            X_train=np.zeros((4, 1)),
            y_train=np.zeros(4),
            feature_names=["a"],
            X_val=np.zeros((2, 1)),
            y_val=np.zeros(2),
        )
        with patch(
            "odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy",
            return_value=strategy,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                study.optimize(objective, n_trials=3)
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.FAIL]

    def test_trial_pruned_is_recorded_as_pruned(self) -> None:
        strategy = MagicMock()
        strategy.train_from_config.side_effect = optuna.TrialPruned()
        study = self._run(strategy, n_trials=2)
        assert {t.state for t in study.trials} == {optuna.trial.TrialState.PRUNED}

    def test_import_error_is_not_swallowed(self) -> None:
        strategy = MagicMock()
        strategy.train_from_config.side_effect = ImportError("xgboost missing")
        with pytest.raises(ImportError):
            self._run(strategy, n_trials=1)


class TestTestSplitZero:
    """`test_split: 0.0` holds out nothing."""

    @staticmethod
    def _events(n: int) -> list[MagicMock]:
        events = []
        for i in range(n):
            event = MagicMock(spec=Event)
            event.id = f"e{i}"
            event.status = EventStatus.FINAL
            event.commence_time = datetime(2024, 10, 15, tzinfo=UTC)
            events.append(event)
        return events

    async def _prepare(self, test_split: float, with_event_ids: bool) -> TrainingDataResult:
        config = make_config(test_split=test_split)
        n_events = 10
        X = np.random.randn(n_events * 2, 3)
        y = np.random.randn(n_events * 2)
        event_ids = np.repeat([f"e{i}" for i in range(n_events)], 2) if with_event_ids else None
        prepared = PreparedFeatureData(X=X, y=y, feature_names=["a", "b", "c"], event_ids=event_ids)
        with (
            patch(
                "odds_analytics.training.data_preparation.filter_events_by_date_range",
                return_value=self._events(n_events),
            ),
            patch("odds_analytics.feature_groups.prepare_training_data", return_value=prepared),
        ):
            return await prepare_training_data_from_config(config, AsyncMock())

    @pytest.mark.asyncio
    async def test_event_level_split_holds_out_nothing(self) -> None:
        result = await self._prepare(0.0, with_event_ids=True)
        assert result.num_train_samples == 20
        assert result.num_test_samples == 0
        assert result.eval_data is None

    @pytest.mark.asyncio
    async def test_row_level_split_holds_out_nothing(self) -> None:
        result = await self._prepare(0.0, with_event_ids=False)
        assert result.num_train_samples == 20
        assert result.num_test_samples == 0

    @pytest.mark.asyncio
    async def test_small_positive_split_still_holds_out_one_event(self) -> None:
        result = await self._prepare(0.05, with_event_ids=True)
        assert result.num_test_samples == 2

    def test_eval_data_prefers_validation_then_test(self) -> None:
        val = np.ones((2, 1))
        test = np.zeros((3, 1))
        result = TrainingDataResult(
            X_train=np.zeros((4, 1)),
            X_test=test,
            y_train=np.zeros(4),
            y_test=np.zeros(3),
            feature_names=["a"],
            strategy_type="xgboost_line_movement",
            X_val=val,
            y_val=np.ones(2),
        )
        assert isinstance(result.eval_data, EvalData)
        assert result.eval_data.X is val
        result.X_val = None
        assert result.eval_data is not None and result.eval_data.X is test

    def test_objective_cv_trial_with_nothing_held_out(self) -> None:
        config = make_config(
            search_spaces=N_ESTIMATORS, use_kfold=True, cv_method="walk_forward", test_split=0.0
        )
        strategy = MagicMock()
        strategy.train_from_config.return_value = dict.fromkeys(
            ("train_mse", "train_mae", "train_r2", "val_mse", "val_mae", "val_r2"), 0.1
        )
        objective = create_objective(
            config=config,
            X_train=np.zeros((10, 1)),
            y_train=np.zeros(10),
            feature_names=["a"],
            X_val=None,
            y_val=None,
            event_ids_train=np.array([f"e{i}" for i in range(10)]),
        )
        trial = optuna.create_study().ask()
        with patch(
            "odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy",
            return_value=strategy,
        ):
            assert objective(trial) == pytest.approx(0.1)

    def test_train_with_cv_without_test_set_fits_final_model_unvalidated(self) -> None:
        config = make_config(use_kfold=True, cv_method="walk_forward", test_split=0.0)
        history = dict.fromkeys(
            ("train_mse", "train_mae", "train_r2", "val_mse", "val_mae", "val_r2"), 0.1
        )
        strategy = MagicMock()
        strategy.train_from_config.return_value = history
        X = np.zeros((10, 1))
        train_with_cv(
            strategy,
            config,
            X,
            np.zeros(10),
            ["a"],
            X_test=np.zeros((0, 1)),
            y_test=np.zeros(0),
            event_ids=np.array([f"e{i}" for i in range(10)]),
            static_test=np.zeros((0, 1)),
            masks_test=np.zeros((0, 1)),
        )
        final_call = strategy.train_from_config.call_args_list[-1]
        assert final_call.kwargs["X_val"] is None
        assert final_call.kwargs["y_val"] is None
        assert final_call.kwargs["static_val"] is None
        assert "masks_val" not in final_call.kwargs


class TestWalkForwardRequiresEventIds:
    def test_raises_instead_of_switching_to_timeseries(self) -> None:
        config = make_config(use_kfold=True, cv_method="walk_forward")
        strategy = MagicMock()
        with pytest.raises(ValueError, match="event_ids"):
            run_cv(strategy, config, np.zeros((10, 1)), np.zeros(10), ["a"], event_ids=None)
        strategy.train_from_config.assert_not_called()


class TestTuningMetric:
    @pytest.mark.parametrize(
        ("metric", "direction"),
        [("val_mse", "minimize"), ("val_mae", "minimize"), ("val_r2", "maximize")],
    )
    def test_direction_derived_from_metric(self, metric: str, direction: str) -> None:
        assert TuningConfig(metric=metric).direction == direction

    def test_contradicting_direction_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires direction 'maximize'"):
            TuningConfig(metric="val_r2", direction="minimize")

    def test_agreeing_direction_is_accepted(self) -> None:
        assert TuningConfig(metric="val_mse", direction="minimize").direction == "minimize"

    @pytest.mark.parametrize("metric", ["cv_val_mse_mean", "accuracy"])
    def test_unsupported_metric_is_rejected(self, metric: str) -> None:
        with pytest.raises(ValidationError):
            TuningConfig(metric=metric)

    def test_cv_objective_returns_requested_metric(self) -> None:
        config = make_config(
            search_spaces=N_ESTIMATORS, metric="val_r2", use_kfold=True, cv_method="walk_forward"
        )
        strategy = MagicMock()
        strategy.train_from_config.return_value = {
            "train_mse": 0.1,
            "train_mae": 0.1,
            "train_r2": 0.5,
            "val_mse": 0.2,
            "val_mae": 0.3,
            "val_r2": 0.4,
        }
        objective = create_objective(
            config=config,
            X_train=np.zeros((10, 1)),
            y_train=np.zeros(10),
            feature_names=["a"],
            event_ids_train=np.array([f"e{i}" for i in range(10)]),
        )
        with patch(
            "odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy",
            return_value=strategy,
        ):
            assert objective(optuna.create_study().ask()) == pytest.approx(0.4)


class TestSearchSpaceKeys:
    def test_unknown_key_fails_validation(self) -> None:
        with pytest.raises(ValidationError, match="no_such_param"):
            make_config(search_spaces={"no_such_param": SearchSpace(type="int", low=1, high=3)})

    def test_model_feature_and_top_k_keys_are_accepted(self) -> None:
        make_config(
            search_spaces={
                "n_estimators": SearchSpace(type="int", low=1, high=3),
                "form_window": SearchSpace(type="int", low=1, high=3),
                "top_k_features": SearchSpace(type="int", low=1, high=3),
            }
        )

    def test_lstm_key_on_xgboost_model_fails(self) -> None:
        with pytest.raises(ValidationError, match="hidden_size"):
            make_config(search_spaces={"hidden_size": SearchSpace(type="int", low=1, high=3)})

    def test_create_objective_rejects_mutated_config(self) -> None:
        config = make_config(search_spaces=N_ESTIMATORS)
        assert config.tuning is not None
        config.tuning.search_spaces["no_such_param"] = SearchSpace(type="int", low=1, high=3)
        with pytest.raises(ValueError, match="no_such_param"):
            create_objective(config, np.zeros((4, 1)), np.zeros(4), ["a"])


class TestStrategyTypes:
    def test_accepted_types_are_exactly_the_implemented_strategies(self) -> None:
        from odds_cli.commands.train import STRATEGY_CLASSES

        assert set(get_args(StrategyType)) == set(STRATEGY_CLASSES)

    @pytest.mark.parametrize("name", ["xgboost", "lstm"])
    def test_unimplemented_alias_is_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            make_config(strategy_type=name)


class TestTrainBestSavesFullConfig:
    def test_saved_yaml_is_readable_by_the_model_loader(self, tmp_path: Path) -> None:
        from odds_cli.commands.train import _save_model_with_config
        from odds_lambda.model_loader import _load_feature_config

        config = make_config()
        config.training.features.sharp_bookmakers = ["pinnacle", "betfair_exchange"]
        config.training.data.sport_key = "soccer_epl"
        strategy = Mock()
        strategy.save_model.side_effect = lambda path: Path(path).write_bytes(b"model")
        model_path = tmp_path / "model.pkl"

        _save_model_with_config(strategy, config, str(model_path))

        s3 = Mock()
        s3.download_file.side_effect = lambda bucket, key, dest: shutil.copy(
            model_path.with_suffix(".yaml"), dest
        )
        feature_config = _load_feature_config(s3, "bucket", "model")
        assert isinstance(feature_config, FeatureConfig)
        assert feature_config.sharp_bookmakers == ["pinnacle", "betfair_exchange"]
        assert "features" in yaml.safe_load(model_path.with_suffix(".yaml").read_text())["training"]

    def test_config_write_failure_fails_the_command(self, tmp_path: Path) -> None:
        from odds_cli.commands.train import _save_model_with_config

        config = Mock()
        config.to_yaml.side_effect = OSError("disk full")
        with pytest.raises(typer.Exit):
            _save_model_with_config(Mock(), config, str(tmp_path / "m.pkl"))
