"""Unit tests for the retail_ev target type and sample weights in training and CV."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrainingConfig,
    XGBoostConfig,
    run_cv,
)
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
    def test_accepts_context_groups(self) -> None:
        assert _features(feature_groups=("standings", "match_stats")).target_type == "retail_ev"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"feature_groups": ("tabular",)},
            {"target_bookmaker": "bet365"},
            {"markets": ["totals"], "outcome": "over"},
            {"adapter": "lstm"},
        ],
    )
    def test_rejects(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            _features(**overrides)


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


class TestRunCvSampleWeights:
    def test_fold_weights_follow_fold_rows(self, config: MLTrainingConfig) -> None:
        event_ids = np.repeat(np.arange(8), 2)
        X = np.arange(16, dtype=float).reshape(-1, 1)
        y = np.zeros(16)
        weights = np.linspace(0.1, 1.6, 16)
        strategy = MagicMock()
        strategy.train_from_config.return_value = dict.fromkeys(
            ("train_mse", "train_mae", "train_r2", "val_mse", "val_mae", "val_r2"), 0.0
        )

        run_cv(strategy, config, X, y, ["x"], event_ids=event_ids, sample_weights=weights)

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
        X = rng.normal(size=(200, 2))
        y = X[:, 0] + rng.normal(size=200)
        weights = rng.uniform(0.1, 3.0, size=200)
        weighted = run_cv(
            XGBoostLineMovementStrategy(),
            config,
            X,
            y,
            ["a", "b"],
            event_ids,
            sample_weights=weights,
        )
        unweighted = run_cv(XGBoostLineMovementStrategy(), config, X, y, ["a", "b"], event_ids)
        assert weighted.mean_val_mse != pytest.approx(unweighted.mean_val_mse)
