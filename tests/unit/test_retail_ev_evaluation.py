"""The one-look retail-EV evaluation: spec, window guard, blocks, verdict and output guard."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from odds_analytics.retail_ev import BASELINE_FEATURE, RetailEvFeatures, event_weights
from odds_analytics.retail_ev_evaluation import (
    BootstrapSpec,
    EvaluationData,
    EvaluationSpec,
    KeepBar,
    OutputExistsError,
    RepeatedLookError,
    Verdict,
    WindowEndError,
    check_one_look,
    check_output_dir,
    check_window_end,
    evaluate,
    evaluation_blocks,
    incomplete_events,
    xgboost_fitter,
)
from odds_analytics.retail_ev_metrics import BootstrapEstimate
from odds_analytics.training.config import RetailEvArm, XGBoostConfig
from odds_core.market_prices import market_outcomes

TRAINING_START = datetime(2024, 8, 1, tzinfo=UTC)
WINDOW_START = datetime(2025, 8, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 31, tzinfo=UTC)
FIRST_KICKOFF = datetime(2024, 8, 10, 15, tzinfo=UTC)


def _spec(**overrides: Any) -> EvaluationSpec:
    fields: dict[str, Any] = {
        "training_start": TRAINING_START,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "refit_every_events": 4,
        "feature_names": RetailEvFeatures(market_outcomes("1x2")).names,
        "bootstrap": {"resamples": 500, "seed": 0},
    }
    return EvaluationSpec.model_validate(fields | overrides)


def _rows(kickoffs: list[datetime], rows_per_event: int = 2) -> pd.DataFrame:
    event_ids = np.repeat([f"e{i:03d}" for i in range(len(kickoffs))], rows_per_event)
    times = np.repeat([k.replace(tzinfo=None) for k in kickoffs], rows_per_event)
    n = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids,
            "commence_time": pd.to_datetime(times),
            "horizon": "1-3d",
            "retail_source": "fduk_opening",
            "close_source": "football_data_uk",
            "event_type": "fduk_only",
            "outcome": np.resize(["home", "draw", "away"], n),
            "y": 0.0,
            "baseline": 0.0,
            "w_event": event_weights(event_ids),
            "w_loss": 1.0,
        }
    )


def _weekly(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(days=7 * i) for i in range(n)]


class TestEvaluationSpec:
    def test_rejects_a_window_starting_before_training(self) -> None:
        with pytest.raises(ValueError, match="training_start < window.start"):
            _spec(training_start=WINDOW_START + timedelta(days=1))

    def test_rejects_a_window_ending_before_it_starts(self) -> None:
        with pytest.raises(ValueError, match="window.start <= window.end"):
            _spec(window={"start": WINDOW_END, "end": WINDOW_START})

    def test_rejects_a_gap_shorter_than_the_decision_window(self) -> None:
        with pytest.raises(ValueError, match="shorter than the decision window"):
            _spec(gap=timedelta(days=6))

    def test_rejects_a_naive_window(self) -> None:
        with pytest.raises(ValueError):
            _spec(window={"start": datetime(2025, 8, 1), "end": datetime(2026, 5, 31)})

    def test_rejects_a_second_seed_in_the_model(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            _spec(model={"random_state": 7})

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValueError):
            _spec(window_end=WINDOW_END)

    def test_trial_parameters_replace_the_model_and_seed_sets_random_state(
        self, tmp_path: Path
    ) -> None:
        summary = {
            "variants": {
                "price_only": {
                    "runs": [
                        {"run": "direct_s42", "best_params": {"max_depth": 2}},
                        {
                            "run": "residual_s42",
                            "best_params": {"max_depth": 4, "n_estimators": 30},
                        },
                    ]
                }
            }
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary))
        spec = _spec(
            model={"n_jobs": 1, "max_depth": 9},
            trial={"path": "summary.json", "variant": "price_only", "run": "residual_s42"},
            seed=11,
        )

        model = spec.resolved_model(tmp_path)

        assert (model.max_depth, model.n_estimators, model.n_jobs) == (4, 30, 1)
        assert model.random_state == 11

    def test_trial_from_a_summary_without_variants(self, tmp_path: Path) -> None:
        summary = {"runs": [{"run": "residual_s42", "best_params": {"learning_rate": 0.02}}]}
        (tmp_path / "summary.json").write_text(json.dumps(summary))
        spec = _spec(trial={"path": "summary.json", "run": "residual_s42"})

        assert spec.resolved_model(tmp_path).learning_rate == 0.02

    def test_missing_trial_run_raises(self, tmp_path: Path) -> None:
        (tmp_path / "summary.json").write_text(json.dumps({"runs": []}))
        spec = _spec(trial={"path": "summary.json", "run": "residual_s7"})

        with pytest.raises(ValueError, match="residual_s7"):
            spec.resolved_model(tmp_path)

    def test_hash_follows_the_spec(self) -> None:
        assert _spec().sha256() == _spec().sha256()
        assert _spec().sha256() != _spec(refit_every_events=5).sha256()


class TestWindowEndGuard:
    def test_accepts_kickoffs_up_to_the_end(self) -> None:
        check_window_end([WINDOW_START, WINDOW_END], WINDOW_END)

    def test_refuses_a_kickoff_after_the_end(self) -> None:
        with pytest.raises(WindowEndError, match="after the window end"):
            check_window_end([WINDOW_START, WINDOW_END + timedelta(seconds=1)], WINDOW_END)


class TestEvaluationBlocks:
    def test_every_window_event_is_scored_once_with_a_partial_last_block(self) -> None:
        rows = _rows(_weekly(FIRST_KICKOFF, 40) + _weekly(WINDOW_START + timedelta(days=1), 10))

        blocks = evaluation_blocks(rows, _spec())

        assert [len(set(rows["event_id"].iloc[b.eval_rows])) for b in blocks] == [4, 4, 2]
        scored = np.concatenate([b.eval_rows for b in blocks])
        window = np.flatnonzero(rows["commence_time"] >= pd.Timestamp(2025, 8, 1))
        np.testing.assert_array_equal(np.sort(scored), window)

    def test_training_events_kick_off_a_gap_before_the_block(self) -> None:
        rows = _rows(_weekly(FIRST_KICKOFF, 40) + _weekly(WINDOW_START + timedelta(days=1), 10))

        for block in evaluation_blocks(rows, _spec(gap=timedelta(days=10))):
            first_eval = rows["commence_time"].iloc[block.eval_rows].min()
            train = rows.iloc[block.train_rows]
            assert train["commence_time"].max() <= first_eval - timedelta(days=10)
            assert (rows["commence_time"] <= first_eval - timedelta(days=10)).sum() == len(train)

    def test_later_blocks_train_on_earlier_window_events(self) -> None:
        rows = _rows(_weekly(FIRST_KICKOFF, 40) + _weekly(WINDOW_START + timedelta(days=1), 10))

        blocks = evaluation_blocks(rows, _spec())

        assert len(blocks[2].train_rows) > len(blocks[0].train_rows)
        trained_in_window = rows["commence_time"].iloc[blocks[2].train_rows] >= pd.Timestamp(
            2025, 8, 1
        )
        assert trained_in_window.any()


def _synthetic_data(signal: float, seed: int = 0) -> EvaluationData:
    """FDUK-like rows whose baseline never picks; ``signal`` is the EV a flagged row adds."""
    rng = np.random.default_rng(seed)
    kickoffs = [FIRST_KICKOFF + timedelta(days=2 * i) for i in range(160)] + [
        WINDOW_START + timedelta(days=1, hours=6 * i) for i in range(120)
    ]
    rows = _rows(kickoffs, rows_per_event=6)
    n = len(rows)
    baseline = rng.normal(-0.04, 0.01, n)
    flag = rng.integers(0, 2, n).astype(float)
    noise = rng.normal(0.0, 0.02, n)
    rows["baseline"] = baseline
    rows["y"] = baseline + signal * flag + noise
    X = np.column_stack([baseline, flag, rng.normal(size=n)]).astype(np.float32)
    return EvaluationData(
        rows=rows,
        X=X,
        feature_names=[BASELINE_FEATURE, "flag", "noise"],
        window_events=rows.drop_duplicates("event_id")[["event_id", "commence_time"]],
        window_drops={},
    )


def _known_answer_spec() -> EvaluationSpec:
    return _spec(
        refit_every_events=40,
        model=XGBoostConfig(n_estimators=60, max_depth=2, learning_rate=0.2, n_jobs=1),
        bootstrap=BootstrapSpec(resamples=1000, seed=0),
    )


class TestKeepBarVerdict:
    @pytest.mark.parametrize(
        ("mse_low", "pick_low", "verdict"),
        [
            (0.1, 0.1, Verdict.KEEP),
            (0.1, 0.0, Verdict.RETIRE),
            (0.0, 0.1, Verdict.RETIRE),
            (-0.1, -0.1, Verdict.RETIRE),
        ],
    )
    def test_keeps_only_when_both_cis_lie_above_zero(
        self, mse_low: float, pick_low: float, verdict: Verdict
    ) -> None:
        bar = KeepBar(
            mse_gain=BootstrapEstimate(1.0, mse_low, 2.0),
            pick_ev_gain=BootstrapEstimate(1.0, pick_low, 2.0),
        )

        assert bar.verdict is verdict


class TestKnownAnswerVerdict:
    def test_a_model_that_finds_value_the_baseline_misses_is_kept(self) -> None:
        data = _synthetic_data(signal=0.10)
        spec = _known_answer_spec()
        model = spec.resolved_model(Path("."))

        scored, blocks, bar = evaluate(
            data, spec, xgboost_fitter(data.feature_names, model, RetailEvArm.RESIDUAL)
        )

        assert len(blocks) == 3
        assert len(scored) == 120 * 6
        assert bar.mse_gain.ci_low > 0
        assert bar.pick_ev_gain.ci_low > 0
        assert bar.verdict is Verdict.KEEP

    def test_a_model_on_noise_is_retired(self) -> None:
        data = _synthetic_data(signal=0.0)
        spec = _known_answer_spec()
        model = spec.resolved_model(Path("."))

        _, _, bar = evaluate(
            data, spec, xgboost_fitter(data.feature_names, model, RetailEvArm.RESIDUAL)
        )

        assert bar.mse_gain.ci_low <= 0
        assert bar.verdict is Verdict.RETIRE

    def test_identical_runs_give_identical_predictions(self) -> None:
        data = _synthetic_data(signal=0.10)
        spec = _known_answer_spec()
        model = spec.resolved_model(Path("."))
        fitter = xgboost_fitter(data.feature_names, model, RetailEvArm.RESIDUAL)

        first, _, _ = evaluate(data, spec, fitter)
        second, _, _ = evaluate(data, spec, fitter)

        np.testing.assert_array_equal(first["model"].to_numpy(), second["model"].to_numpy())


class TestOutputDirectory:
    def test_creates_nothing(self, tmp_path: Path) -> None:
        check_output_dir(tmp_path / "out", force=False)

        assert not (tmp_path / "out").exists()

    def test_accepts_an_empty_directory(self, tmp_path: Path) -> None:
        check_output_dir(tmp_path, force=False)

    def test_refuses_a_directory_holding_an_evaluation(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")

        with pytest.raises(OutputExistsError):
            check_output_dir(tmp_path, force=False)

    def test_force_overrides_the_refusal(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")

        check_output_dir(tmp_path, force=True)


class TestReadiness:
    def test_flags_events_that_are_not_final_or_have_no_fduk_close(self) -> None:
        window = pd.DataFrame(
            {
                "event_id": ["done", "scheduled", "no_close"],
                "status": ["final", "scheduled", "final"],
                "has_fduk_closing": [True, True, False],
            }
        )

        assert incomplete_events(window) == ["scheduled", "no_close"]


class TestOneLook:
    def _write(self, root: Path, name: str, spec_sha256: str) -> Path:
        (root / name).mkdir()
        path = root / name / "manifest.json"
        path.write_text(json.dumps({"spec_sha256": spec_sha256, "predictions_sha256": "abc"}))
        return path

    def test_refuses_a_spec_already_evaluated(self, tmp_path: Path) -> None:
        self._write(tmp_path, "earlier", "s1")

        with pytest.raises(RepeatedLookError, match="already evaluated"):
            check_one_look(tmp_path, "s1", force=False)

    def test_other_specs_do_not_block(self, tmp_path: Path) -> None:
        self._write(tmp_path, "earlier", "s1")

        assert check_one_look(tmp_path, "s2", force=False) == []

    def test_skips_unreadable_manifests(self, tmp_path: Path) -> None:
        (tmp_path / "broken").mkdir()
        (tmp_path / "broken" / "manifest.json").write_text("{not json")
        earlier = self._write(tmp_path, "earlier", "s1")

        assert check_one_look(tmp_path, "s1", force=True) == [earlier]

    def test_force_returns_the_earlier_runs(self, tmp_path: Path) -> None:
        earlier = self._write(tmp_path, "earlier", "s1")

        assert check_one_look(tmp_path, "s1", force=True) == [earlier]

    def test_forced_overwrite_returns_the_overwritten_predictions_hash(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path, "out", "s1")

        assert check_output_dir(tmp_path / "out", force=True) == "abc"
