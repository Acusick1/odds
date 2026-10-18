"""Unit tests for the ADR-0012 keep-bar metrics on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from odds_analytics.retail_ev_metrics import (
    event_weights,
    mse_gain,
    pick_ev_gain,
    weighted_mse,
)


class TestEventWeights:
    def test_one_over_rows_in_event(self) -> None:
        ids = np.array(["a", "a", "b", "c", "c", "c", "c"])
        np.testing.assert_allclose(event_weights(ids), [0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25])


class TestWeightedMse:
    def test_known_value(self) -> None:
        y = np.array([0.0, 1.0])
        pred = np.array([1.0, 1.0])
        assert weighted_mse(y, pred, np.array([3.0, 1.0])) == pytest.approx(0.75)


class TestMseGain:
    def test_constant_gain_has_degenerate_ci(self) -> None:
        y = np.array([0.0, 0.5, 1.0, -0.5])
        ids = np.array(["a", "b", "c", "d"])
        result = mse_gain(y, y, y + 0.1, np.ones(4), ids, n_boot=200)
        assert result.point == pytest.approx(0.01)
        assert result.ci_low == pytest.approx(0.01)
        assert result.ci_high == pytest.approx(0.01)
        assert result.excludes_zero

    def test_weighted_point_estimate(self) -> None:
        y = np.zeros(3)
        model = np.array([0.1, 0.0, 0.3])
        baseline = np.array([0.2, 0.2, 0.0])
        w = np.array([1.0, 1.0, 2.0])
        ids = np.array(["a", "a", "b"])
        expected = ((1 * 0.04 + 1 * 0.04 + 0) - (1 * 0.01 + 0 + 2 * 0.09)) / 4
        assert mse_gain(y, model, baseline, w, ids).point == pytest.approx(expected)

    def test_identical_predictions_give_zero(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.normal(size=50)
        pred = rng.normal(size=50)
        ids = np.repeat(np.arange(10), 5)
        result = mse_gain(y, pred, pred, np.ones(50), ids)
        assert (result.point, result.ci_low, result.ci_high) == (0.0, 0.0, 0.0)

    def test_ci_brackets_point_and_resamples_events(self) -> None:
        rng = np.random.default_rng(2)
        n_events, per_event = 200, 6
        ids = np.repeat(np.arange(n_events), per_event)
        event_noise = np.repeat(rng.normal(size=n_events), per_event)
        y = event_noise + rng.normal(scale=0.1, size=ids.size)
        model = event_noise
        baseline = np.zeros(ids.size)
        result = mse_gain(y, model, baseline, np.ones(ids.size), ids)
        assert result.ci_low < result.point < result.ci_high
        assert result.excludes_zero

    def test_clustered_ci_is_wider_than_row_level(self) -> None:
        rng = np.random.default_rng(3)
        n_events, per_event = 100, 20
        clustered = np.repeat(np.arange(n_events), per_event)
        shared_error = np.repeat(rng.normal(size=n_events), per_event)
        y = np.zeros(clustered.size)
        model = shared_error
        baseline = rng.normal(size=clustered.size)
        w = np.ones(clustered.size)
        by_event = mse_gain(y, model, baseline, w, clustered)
        by_row = mse_gain(y, model, baseline, w, np.arange(clustered.size))
        assert by_event.point == pytest.approx(by_row.point)
        assert by_event.ci_high - by_event.ci_low > by_row.ci_high - by_row.ci_low


class TestPickEvGain:
    def test_totals_of_differing_pick_sets(self) -> None:
        y = np.array([0.1, -0.2, 0.3])
        model = np.array([1.0, -1.0, 1.0])
        baseline = np.array([1.0, 1.0, -1.0])
        ids = np.array(["a", "b", "c"])
        result = pick_ev_gain(y, model, baseline, np.ones(3), ids, n_boot=100)
        assert result.point == pytest.approx((0.1 + 0.3) - (0.1 - 0.2))

    def test_empty_model_picks_total_zero(self) -> None:
        y = np.array([0.1, -0.2, 0.3])
        ids = np.array(["a", "b", "c"])
        result = pick_ev_gain(y, -np.ones(3), np.ones(3), np.ones(3), ids)
        assert result.point == pytest.approx(-0.2)

    def test_event_weights_scale_each_pick(self) -> None:
        ids = np.array(["a", "a", "b"])
        y = np.array([0.2, 0.4, -0.1])
        model = np.ones(3)
        baseline = -np.ones(3)
        result = pick_ev_gain(y, model, baseline, event_weights(ids), ids)
        assert result.point == pytest.approx(0.5 * 0.2 + 0.5 * 0.4 - 0.1)

    def test_consistent_edge_excludes_zero(self) -> None:
        rng = np.random.default_rng(4)
        n = 400
        ids = np.arange(n)
        y = rng.normal(scale=0.05, size=n)
        model = y
        baseline = rng.normal(size=n)
        result = pick_ev_gain(y, model, baseline, np.ones(n), ids)
        assert result.ci_low > 0
