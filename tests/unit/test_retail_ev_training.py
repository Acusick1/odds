"""Unit tests for the retail_ev target type and sample weights in training and CV."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from odds_analytics.retail_ev import DECISION_WINDOW
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrainingConfig,
    XGBoostConfig,
    run_cv,
    train_with_cv,
)
from odds_analytics.training.config import SearchSpace, TuningConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.tuner import create_objective
from odds_analytics.training.utils import compute_regression_metrics
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from pydantic import ValidationError


def _features(**overrides: object) -> FeatureConfig:
    fields: dict[str, object] = {
        "target_type": "retail_ev",
        "markets": ["1x2"],
        "feature_groups": (),
    }
    fields.update(overrides)
    return FeatureConfig(**fields)


@pytest.fixture
def config() -> MLTrainingConfig:
    return MLTrainingConfig(
        experiment=ExperimentConfig(name="retail_ev_unit"),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date=date(2025, 8, 1),
                end_date=date(2026, 6, 1),
                test_split=0.0,
                validation_split=0.0,
                cv_method="walk_forward",
                min_train_events=4,
                val_step_events=2,
            ),
            model=XGBoostConfig(n_estimators=30, learning_rate=0.3, max_depth=2, subsample=1.0),
            features=_features(),
        ),
    )


class TestRetailEvConfig:
    def test_accepts_defaults(self) -> None:
        assert _features().target_type == "retail_ev"

    def test_round_trips_through_a_full_dump(self) -> None:
        assert FeatureConfig(**_features().model_dump()).target_type == "retail_ev"

    def test_accepts_as_of_t_groups(self) -> None:
        groups = ("polymarket_at_t", "dynamics", "schedule_as_of_t", "standings_as_of_t")
        assert _features(feature_groups=groups).feature_groups == groups

    def test_other_targets_reject_as_of_t_groups(self) -> None:
        with pytest.raises(ValidationError, match="retail_ev feature groups"):
            FeatureConfig(
                target_type="devigged_bookmaker",
                target_bookmaker="bet365",
                markets=["1x2"],
                feature_groups=("tabular", "dynamics"),
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"feature_groups": ("tabular",)},
            {"feature_groups": ("dynamics", "standings")},
            {"feature_groups": ("standings",)},
            {"feature_groups": ("match_stats",)},
            {"feature_groups": ("epl_schedule",)},
            {"outcome": "away"},
            {"sharp_bookmakers": ["betfair_exchange"]},
            {"retail_bookmakers": ["bet365"]},
            {"closing_source_priority": ["football_data_uk"]},
            {"sampling": {"strategy": "tier"}},
            {"target_bookmaker": "bet365"},
            {"markets": ["totals"], "outcome": "over"},
            {"adapter": "lstm"},
        ],
    )
    def test_rejects(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            _features(**overrides)


class TestRetailEvProtocol:
    @pytest.mark.parametrize(
        "data_overrides",
        [
            {"test_split": 0.2},
            {"use_kfold": False, "validation_split": 0.2},
            {"use_kfold": False},
            {"cv_method": "kfold", "n_folds": 2},
            {"cv_method": "timeseries", "n_folds": 2},
        ],
    )
    def test_rejects_holdout_and_non_walk_forward(
        self, config: MLTrainingConfig, data_overrides: dict[str, object]
    ) -> None:
        raw = config.model_dump()
        raw["training"]["data"].update(data_overrides)
        with pytest.raises(ValidationError, match="retail_ev"):
            MLTrainingConfig(**raw)

    def test_accepts_walk_forward(self, config: MLTrainingConfig) -> None:
        assert MLTrainingConfig(**config.model_dump()).training.data.cv_method == "walk_forward"


class TestWeightedMetrics:
    def test_weights_change_every_metric(self) -> None:
        y = np.array([0.0, 1.0, 2.0])
        pred = np.array([0.0, 0.0, 0.0])
        w = np.array([1.0, 0.0, 1.0])
        weighted = compute_regression_metrics(y, pred, w)
        assert weighted["mse"] == pytest.approx(2.0)
        assert weighted["mae"] == pytest.approx(1.0)
        assert weighted["r2"] == pytest.approx(1 - 4 / 2)
        assert compute_regression_metrics(y, pred)["mse"] == pytest.approx(5 / 3)


class TestXGBoostSampleWeights:
    def test_weights_reach_fit(self) -> None:
        X = np.zeros((200, 2))
        y = np.r_[np.zeros(100), np.ones(100)]
        w = np.r_[np.ones(100), np.zeros(100)]
        strategy = XGBoostLineMovementStrategy()
        strategy.train(X, y, ["a", "b"], sample_weight_train=w, n_estimators=50, learning_rate=0.5)
        assert strategy.model is not None
        assert abs(float(strategy.model.predict(X[:1])[0])) < 0.05

    def test_weights_reach_eval_set_and_history(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = X[:, 0] + rng.normal(scale=0.5, size=300)
        w = rng.uniform(0.1, 2.0, size=300)
        strategy = XGBoostLineMovementStrategy()
        history = strategy.train(
            X[:200],
            y[:200],
            ["a", "b", "c"],
            X_val=X[200:],
            y_val=y[200:],
            sample_weight_train=w[:200],
            sample_weight_val=w[200:],
            n_estimators=20,
        )
        assert strategy.model is not None
        pred = strategy.model.predict(X[200:])
        expected = compute_regression_metrics(y[200:], pred, w[200:])
        assert history["val_mse"] == pytest.approx(expected["mse"], rel=1e-5)
        eval_rmse = strategy.model.evals_result()["validation_1"]["rmse"][-1]
        assert eval_rmse == pytest.approx(np.sqrt(expected["mse"]), rel=1e-4)


KICKOFF0 = np.datetime64("2025-08-16T14:00", "us")


def _kickoffs(event_ids: np.ndarray, spacing_hours: float) -> np.ndarray:
    return KICKOFF0 + (event_ids * spacing_hours * 3600 * 1e6).astype("timedelta64[us]")


class TestRunCvSampleWeights:
    def test_fold_weights_follow_fold_rows(self, config: MLTrainingConfig) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        X = np.arange(16, dtype=float).reshape(-1, 1)
        y = np.zeros(16)
        weights = np.linspace(0.1, 1.6, 16)
        strategy = _stub_strategy()

        run_cv(
            strategy,
            config,
            X,
            y,
            ["x"],
            event_ids=event_ids,
            sample_weights=weights,
            commence_times=_kickoffs(event_ids, 24 * 7),
        )

        assert strategy.train_from_config.call_count == 2
        for call in strategy.train_from_config.call_args_list:
            kwargs = call.kwargs
            np.testing.assert_array_equal(
                kwargs["sample_weight_train"], weights[kwargs["X_train"][:, 0].astype(int)]
            )
            np.testing.assert_array_equal(
                kwargs["sample_weight_val"], weights[kwargs["X_val"][:, 0].astype(int)]
            )

    def test_weighted_cv_metric_end_to_end(self, config: MLTrainingConfig) -> None:
        rng = np.random.default_rng(5)
        event_ids = np.repeat(np.arange(8), 25)
        kickoffs = _kickoffs(event_ids, 24 * 7)
        X = rng.normal(size=(200, 2))
        y = X[:, 0] + rng.normal(size=200)
        weights = rng.uniform(0.1, 3.0, size=200)

        def mean_val_mse(w: np.ndarray) -> float:
            result = run_cv(
                XGBoostLineMovementStrategy(),
                config,
                X,
                y,
                ["a", "b"],
                event_ids,
                sample_weights=w,
                commence_times=kickoffs,
            )
            return result.mean_val_mse

        assert mean_val_mse(weights) != pytest.approx(mean_val_mse(np.ones(200)))


def _stub_strategy() -> MagicMock:
    strategy = MagicMock()
    strategy.train_from_config.return_value = dict.fromkeys(
        ("train_mse", "train_mae", "train_r2", "val_mse", "val_mae", "val_r2"), 0.0
    )
    return strategy


class TestRetailEvWalkForwardGap:
    def test_drops_training_events_within_the_decision_window(self) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        splits = make_walk_forward_splits(
            event_ids,
            min_train_events=4,
            val_step_events=2,
            commence_times=_kickoffs(event_ids, 84),
            gap=DECISION_WINDOW,
        )
        train_events = [sorted(set(event_ids[train].tolist())) for train, _ in splits]
        val_events = [sorted(set(event_ids[val].tolist())) for _, val in splits]
        assert val_events == [[4, 5], [6, 7]]
        assert train_events == [[0, 1, 2], [0, 1, 2, 3, 4]]

    def test_run_cv_applies_the_gap_for_retail_ev(self, config: MLTrainingConfig) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        strategy = _stub_strategy()
        run_cv(
            strategy,
            config,
            np.zeros((16, 1)),
            np.zeros(16),
            ["x"],
            event_ids=event_ids,
            sample_weights=np.ones(16),
            commence_times=_kickoffs(event_ids, 84),
        )
        first_fold = strategy.train_from_config.call_args_list[0].kwargs
        assert len(first_fold["X_train"]) == 3 * 2

    def test_gap_needs_commence_times(self) -> None:
        with pytest.raises(ValueError, match="commence_times"):
            make_walk_forward_splits(np.arange(8), 4, 2, gap=DECISION_WINDOW)


class TestRetailEvRequiresWeights:
    @pytest.mark.parametrize("missing", ["sample_weights", "commence_times"])
    def test_run_cv(self, config: MLTrainingConfig, missing: str) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        inputs = {"sample_weights": np.ones(16), "commence_times": _kickoffs(event_ids, 168)}
        inputs[missing] = None
        with pytest.raises(ValueError, match="retail_ev"):
            run_cv(
                _stub_strategy(),
                config,
                np.zeros((16, 1)),
                np.zeros(16),
                ["x"],
                event_ids,
                **inputs,
            )

    def test_train_with_cv(self, config: MLTrainingConfig) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        with pytest.raises(ValueError, match="retail_ev"):
            train_with_cv(
                _stub_strategy(),
                config,
                np.zeros((16, 1)),
                np.zeros(16),
                ["x"],
                event_ids=event_ids,
            )

    def test_rejects_non_walk_forward(self, config: MLTrainingConfig) -> None:
        kfold = config.model_copy(deep=True)
        kfold.training.data.cv_method = "kfold"
        kfold.training.data.n_folds = 2
        event_ids = np.repeat(np.arange(8), 2)
        with pytest.raises(ValueError, match="walk_forward"):
            run_cv(
                _stub_strategy(),
                kfold,
                np.zeros((16, 1)),
                np.zeros(16),
                ["x"],
                event_ids,
                sample_weights=np.ones(16),
                commence_times=_kickoffs(event_ids, 168),
            )

    def test_create_objective(self, config: MLTrainingConfig) -> None:
        tuned = config.model_copy(deep=True)
        tuned.tuning = TuningConfig(
            n_trials=1,
            search_spaces={"max_depth": SearchSpace(type="int", low=2, high=3)},
        )
        with pytest.raises(ValueError, match="retail_ev"):
            create_objective(tuned, np.zeros((16, 1)), np.zeros(16), ["x"])
