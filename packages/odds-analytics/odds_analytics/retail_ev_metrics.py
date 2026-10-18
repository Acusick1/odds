"""ADR-0012 keep-bar metrics, each with an event-clustered bootstrap 95% CI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 2000
CI_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapEstimate:
    point: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def event_weights(event_ids: np.ndarray) -> np.ndarray:
    """``w_event = 1 / (rows in the event)`` for each row."""
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def weighted_mse(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (y - pred) ** 2) / np.sum(weights))


def _event_sums(event_ids: np.ndarray, *values: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(event_ids, return_inverse=True)
    return [np.bincount(inverse, weights=v) for v in values]


def _resample_counts(n_events: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=n_boot)


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    tail = (1 - CI_LEVEL) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def mse_gain(
    y: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    loss_weights: np.ndarray,
    event_ids: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> BootstrapEstimate:
    """Baseline weighted MSE minus the model's; positive means the model is better."""
    se_model, se_base, w = _event_sums(
        event_ids,
        loss_weights * (y - pred_model) ** 2,
        loss_weights * (y - pred_baseline) ** 2,
        loss_weights,
    )
    point = (se_base.sum() - se_model.sum()) / w.sum()
    counts = _resample_counts(len(w), n_boot, seed)
    samples = (counts @ se_base - counts @ se_model) / (counts @ w)
    low, high = _percentile_ci(samples)
    return BootstrapEstimate(float(point), low, high)


def pick_ev_gain(
    y: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    w_event: np.ndarray,
    event_ids: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> BootstrapEstimate:
    """Total ``w_event``-weighted realised EV of rows predicted EV > 0, model minus baseline.

    ``y`` is the realised EV against the close. An empty pick set totals zero.
    """
    (diff,) = _event_sums(
        event_ids, w_event * y * ((pred_model > 0).astype(float) - (pred_baseline > 0))
    )
    counts = _resample_counts(len(diff), n_boot, seed)
    low, high = _percentile_ci(counts @ diff)
    return BootstrapEstimate(float(diff.sum()), low, high)
