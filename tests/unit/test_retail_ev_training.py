"""Unit tests for the retail_ev target type and sample weights in training and CV."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest
from odds_analytics.retail_ev import BASELINE_FEATURE, DECISION_WINDOW
from odds_analytics.retail_ev_metrics import weighted_mse
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
from odds_analytics.training.config import RetailEvArm, SearchSpace, TuningConfig
from odds_analytics.training.cross_validation import make_walk_forward_splits
from odds_analytics.training.tuner import create_objective
from odds_analytics.training.utils import compute_regression_metrics
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from pydantic import ValidationError
from xgboost import XGBRegressor


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

    def test_accepts_the_residual_arm(self) -> None:
        assert _features(retail_ev_arm="residual").retail_ev_arm is RetailEvArm.RESIDUAL

    def test_other_targets_reject_the_residual_arm(self) -> None:
        with pytest.raises(ValidationError, match="retail_ev_arm"):
            FeatureConfig(
                target_type="raw",
                markets=["1x2"],
                retail_ev_arm="residual",
            )

    def test_accepts_as_of_t_groups(self) -> None:
        groups = ("polymarket_at_t", "dynamics", "schedule_as_of_t", "standings_as_of_t")
        assert _features(feature_groups=groups).feature_groups == groups

    def test_other_targets_reject_as_of_t_groups(self) -> None:
        with pytest.raises(ValidationError, match="retail_ev feature groups"):
            FeatureConfig(
                target_type="raw",
                markets=["1x2"],
                feature_groups=("tabular", "dynamics"),
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"feature_groups": ("tabular",)},
            {"outcome": "away"},
            {"sharp_bookmakers": ["betfair_exchange"]},
            {"retail_bookmakers": ["bet365"]},
            {"sampling": {"strategy": "tier"}},
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

    def test_residual_arm_rejects_top_k_feature_selection(self, config: MLTrainingConfig) -> None:
        raw = config.model_dump()
        raw["training"]["features"]["retail_ev_arm"] = "residual"
        raw["tuning"] = {
            "n_trials": 1,
            "search_spaces": {"top_k_features": {"type": "int", "low": 5, "high": 10}},
        }
        with pytest.raises(ValidationError, match="top_k_features"):
            MLTrainingConfig(**raw)

    def test_residual_arm_rejects_mlflow_tracking(self, config: MLTrainingConfig) -> None:
        raw = config.model_dump()
        raw["training"]["features"]["retail_ev_arm"] = "residual"
        raw["tracking"] = {"enabled": True}
        with pytest.raises(ValidationError, match="MLflow"):
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


FEATURES = ["a", BASELINE_FEATURE, "b"]
N_EVENTS = 8
ROWS_PER_EVENT = 25


@pytest.fixture
def residual_config(config: MLTrainingConfig) -> MLTrainingConfig:
    residual = config.model_copy(deep=True)
    residual.training.features.retail_ev_arm = RetailEvArm.RESIDUAL
    return residual


@pytest.fixture
def rows() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    n = N_EVENTS * ROWS_PER_EVENT
    X = rng.normal(size=(n, 3)).astype(np.float32)
    X[:, 1] = rng.normal(scale=0.1, size=n)
    y = (X[:, 1] + 0.05 * X[:, 0] + 0.02 + rng.normal(scale=0.02, size=n)).astype(np.float32)
    event_ids = np.repeat(np.arange(N_EVENTS), ROWS_PER_EVENT)
    return {
        "X": X,
        "y": y,
        "w": rng.uniform(0.5, 2.0, size=n),
        "event_ids": event_ids,
        "kickoffs": _kickoffs(event_ids, 24 * 7),
    }


def _learns_nothing(config: MLTrainingConfig) -> MLTrainingConfig:
    frozen = config.model_copy(deep=True)
    frozen.training.model.min_child_weight = 10**9
    return frozen


class TestResidualArm:
    def test_matches_fitting_the_target_minus_the_baseline_with_no_intercept(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        X, y, w = rows["X"], rows["y"], rows["w"]
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(residual_config, X, y, FEATURES, sample_weight_train=w)
        assert strategy.model is not None

        baseline = X[:, 1]
        script = XGBRegressor(**{**strategy.model.get_params(), "base_score": 0.0})
        script.fit(X, y - baseline, sample_weight=w)
        np.testing.assert_allclose(
            strategy.predict(X), script.predict(X) + baseline, rtol=1e-5, atol=1e-6
        )

    def test_a_model_that_learns_nothing_predicts_the_baseline(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        X = rows["X"]
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            _learns_nothing(residual_config), X, rows["y"], FEATURES, sample_weight_train=rows["w"]
        )
        np.testing.assert_array_equal(strategy.predict(X), X[:, 1])

    def test_direct_arm_has_no_margin(
        self, config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            _learns_nothing(config), rows["X"], rows["y"], FEATURES, sample_weight_train=rows["w"]
        )
        assert strategy.margin_feature is None
        assert np.unique(strategy.predict(rows["X"])).size == 1

    def test_cv_scores_predictions_on_the_target_scale(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        X, y, w = rows["X"], rows["y"], rows["w"]
        result = run_cv(
            XGBoostLineMovementStrategy(),
            _learns_nothing(residual_config),
            X,
            y,
            FEATURES,
            rows["event_ids"],
            sample_weights=w,
            commence_times=rows["kickoffs"],
        )
        splits = make_walk_forward_splits(
            rows["event_ids"],
            min_train_events=4,
            val_step_events=2,
            commence_times=rows["kickoffs"],
            gap=DECISION_WINDOW,
        )
        expected = [weighted_mse(y[v], X[v, 1], w[v]) for _, v in splits]
        assert [f.val_mse for f in result.fold_results] == pytest.approx(expected, rel=1e-6)

    def test_tuning_objective_scores_the_residual_arm(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        tuned = _learns_nothing(residual_config)
        tuned.tuning = TuningConfig(
            n_trials=1,
            search_spaces={"max_depth": SearchSpace(type="int", low=2, high=3)},
        )
        objective = create_objective(
            tuned,
            rows["X"],
            rows["y"],
            FEATURES,
            event_ids_train=rows["event_ids"],
            sample_weights_train=rows["w"],
            commence_times_train=rows["kickoffs"],
        )
        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

        reference = run_cv(
            XGBoostLineMovementStrategy(),
            tuned,
            rows["X"],
            rows["y"],
            FEATURES,
            rows["event_ids"],
            sample_weights=rows["w"],
            commence_times=rows["kickoffs"],
        )
        assert study.best_value == pytest.approx(reference.mean_val_mse, rel=1e-6)

    def test_save_and_load_keep_the_margin(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray], tmp_path: Path
    ) -> None:
        strategy = XGBoostLineMovementStrategy()
        strategy.train_from_config(
            residual_config, rows["X"], rows["y"], FEATURES, sample_weight_train=rows["w"]
        )
        path = tmp_path / "residual.joblib"
        strategy.save_model(path)
        loaded = XGBoostLineMovementStrategy(model_path=str(path))
        assert loaded.margin_feature == BASELINE_FEATURE
        np.testing.assert_array_equal(loaded.predict(rows["X"]), strategy.predict(rows["X"]))

    def test_needs_the_baseline_column(
        self, residual_config: MLTrainingConfig, rows: dict[str, np.ndarray]
    ) -> None:
        with pytest.raises(ValueError, match=BASELINE_FEATURE):
            XGBoostLineMovementStrategy().train_from_config(
                residual_config, rows["X"], rows["y"], ["a", "c", "b"]
            )
