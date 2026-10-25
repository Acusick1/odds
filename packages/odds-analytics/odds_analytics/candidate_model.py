"""The retail-EV model behind a model candidate rule, tuned and refitted walk-forward in a replay.

A replay leaves the first ``training_matches`` of its range to training only. Hyperparameters
are tuned once on those matches alone, by expanding walk-forward folds in kick-off order, and
then fixed. The remaining matches are scored in blocks of ``refit_every_matches``; before each
block the model is refitted on every range match that kicked off before the block's first
decision time.

Rows, and the inputs the model predicts from, come from co-timed snapshots only: each is the
market state ``odds_lambda.candidate_rules.build_market_state`` builds at a snapshot's own
time, so a training row is exactly what the rule would have seen at that time. A row's label
is the live Betfair close, which exists once the match kicks off; a match closed only by FDUK,
whose closing prices are published days later, gives no rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from odds_core.betfair_close import event_close
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event, OddsSnapshot
from odds_core.snapshot_sources import SnapshotKind
from odds_lambda.candidate_rules import (
    MarketState,
    ModelRuleSpec,
    RetailEvModelParams,
    RuleSpec,
    build_market_state,
    decision_times,
)
from odds_lambda.candidate_rules.market_state import sport_market
from odds_lambda.storage.readers import OddsReader

from odds_analytics.retail_ev import (
    DECISION_CUTOFF,
    DECISION_WINDOW,
    RetailEvDataset,
    RetailEvFeatures,
    RetailEvInput,
    RetailEvRow,
    fit_retail_ev,
    quote_inputs,
)
from odds_analytics.retail_ev_metrics import weighted_mse
from odds_analytics.training.config import RetailEvArm, SearchSpace

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy

# The price-only retail-EV search space (experiments/configs/xgboost_epl_retail_ev_price_only.yaml),
# whose min_child_weight range spans about 0.13 to 13 events at the fixed loss-weight scale.
SEARCH_SPACES: dict[str, SearchSpace] = {
    "n_estimators": SearchSpace(type="int", low=100, high=600, step=50),
    "max_depth": SearchSpace(type="int", low=2, high=5),
    "learning_rate": SearchSpace(type="float", low=0.01, high=0.1, log=True),
    "min_child_weight": SearchSpace(type="int", low=20, high=2000, log=True),
    "subsample": SearchSpace(type="float", low=0.5, high=1.0, step=0.1),
    "colsample_bytree": SearchSpace(type="float", low=0.5, high=1.0, step=0.1),
    "reg_lambda": SearchSpace(type="float", low=0.01, high=10.0, log=True),
}
# Locked: gamma is redundant with min_child_weight at these depths, and L2 alone regularises.
FIXED_XGBOOST_PARAMS: dict[str, Any] = {
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": 1,
}


def state_inputs(state: MarketState) -> list[RetailEvInput]:
    """The state's quotes as model inputs at its snapshot's time; none without an exchange price."""
    if state.exchange is None or state.snapshot_time is None:
        return []
    return quote_inputs(
        state.event_id,
        state.kickoff,
        state.exchange.kind,
        state.snapshot_time,
        state.retail,
        market_outcomes(state.market),
        state.exchange,
        polymarket=None,
        dynamics=None,
    )


def training_rows(event: Event, snapshots: Sequence[OddsSnapshot]) -> list[RetailEvRow]:
    """Rows of every co-timed snapshot in the decision window, against the live Betfair close.

    An event without a live Betfair close has none.
    """
    ordered = sorted(snapshots, key=lambda s: s.snapshot_time)
    close = event_close(event, ordered, sport_market(event.sport_key))
    if close is None or close.kind is not SnapshotKind.BETFAIR_LIVE:
        return []
    first = event.commence_time - DECISION_WINDOW
    last = event.commence_time - DECISION_CUTOFF
    rows: list[RetailEvRow] = []
    for snapshot in ordered:
        if not first <= snapshot.snapshot_time <= last:
            continue
        state = build_market_state(event, [snapshot], snapshot.snapshot_time, timedelta(0))
        rows.extend(RetailEvRow.with_close(inputs, close) for inputs in state_inputs(state))
    return rows


def xgboost_params(tuned: dict[str, Any], seed: int) -> dict[str, Any]:
    return {**FIXED_XGBOOST_PARAMS, **tuned, "random_state": seed}


@dataclass(frozen=True)
class _Matrix:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    baseline: np.ndarray


def _matrix(features: RetailEvFeatures, rows: Sequence[RetailEvRow]) -> _Matrix:
    dataset = RetailEvDataset(rows=list(rows))
    return _Matrix(
        X=features.matrix(rows),
        y=dataset.targets.astype(np.float32),
        weights=dataset.loss_weights,
        baseline=dataset.baselines,
    )


@dataclass(frozen=True)
class RetailEvPredictor:
    """Predicted EV of every quote in a state, from one fit."""

    strategy: XGBoostLineMovementStrategy
    features: RetailEvFeatures

    def __call__(self, state: MarketState) -> dict[tuple[Outcome, str], float]:
        inputs = state_inputs(state)
        if not inputs:
            return {}
        predicted = self.strategy.predict(self.features.matrix(inputs))
        return {(i.outcome, i.book): float(p) for i, p in zip(inputs, predicted, strict=True)}


def fit_model(
    market: str, rows: Sequence[RetailEvRow], params: dict[str, Any]
) -> RetailEvPredictor:
    if not rows:
        raise ValueError("no training rows: no co-timed snapshot of a match with a live close")
    features = RetailEvFeatures(market_outcomes(market))
    m = _matrix(features, rows)
    strategy = fit_retail_ev(m.X, m.y, m.weights, features.names, params, RetailEvArm.RESIDUAL)
    return RetailEvPredictor(strategy, features)


@dataclass(frozen=True)
class TuningFold:
    """Validates ``validate`` with a fit on the training matches that kicked off at least
    ``DECISION_WINDOW`` before the first of them, so no training close follows a validation row."""

    index: int
    train: tuple[Event, ...]
    validate: tuple[Event, ...]


def tuning_folds(prefix: Sequence[Event], params: RetailEvModelParams) -> list[TuningFold]:
    """Expanding walk-forward folds over the training matches, in kick-off order."""
    folds: list[TuningFold] = []
    starts = range(params.tuning_min_train_matches, len(prefix), params.tuning_step_matches)
    for index, start in enumerate(starts):
        validate = tuple(prefix[start : start + params.tuning_step_matches])
        cutoff = validate[0].commence_time - DECISION_WINDOW
        train = tuple(e for e in prefix[:start] if e.commence_time <= cutoff)
        folds.append(TuningFold(index, train, validate))
    return folds


@dataclass(frozen=True)
class Tuning:
    """The hyperparameters chosen on the training matches and their walk-forward score.

    ``cv_wmse`` and ``baseline_cv_wmse`` are mean fold loss-weighted MSE of the chosen model and
    of the exchange-at-t baseline, over the ``folds`` folds with training and validation rows.
    """

    params: dict[str, Any]
    cv_wmse: float
    baseline_cv_wmse: float
    folds: int
    trials: int


def choose_hyperparameters(
    market: str,
    folds: Sequence[tuple[Sequence[RetailEvRow], Sequence[RetailEvRow]]],
    params: RetailEvModelParams,
) -> Tuning:
    """Pick the trial with the lowest mean fold weighted MSE: TPE, ``tuning_trials`` trials,
    seeded with ``tuning_seed``. Folds without training or validation rows are skipped."""
    import optuna

    from odds_analytics.training.tuner import suggest_params_from_search_space

    features = RetailEvFeatures(market_outcomes(market))
    usable = [
        (_matrix(features, train), _matrix(features, val)) for train, val in folds if train and val
    ]
    if not usable:
        raise ValueError("no tuning fold has both training and validation rows")

    def objective(trial: optuna.Trial) -> float:
        xgb = xgboost_params(
            suggest_params_from_search_space(trial, SEARCH_SPACES), params.tuning_seed
        )
        scores = []
        for train, val in usable:
            strategy = fit_retail_ev(
                train.X, train.y, train.weights, features.names, xgb, RetailEvArm.RESIDUAL
            )
            scores.append(weighted_mse(val.y, strategy.predict(val.X), val.weights))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=params.tuning_seed)
    )
    study.optimize(objective, n_trials=params.tuning_trials)
    return Tuning(
        params=dict(study.best_params),
        cv_wmse=float(study.best_value),
        baseline_cv_wmse=float(
            np.mean([weighted_mse(val.y, val.baseline, val.weights) for _, val in usable])
        ),
        folds=len(usable),
        trials=params.tuning_trials,
    )


@dataclass(frozen=True)
class RefitBlock:
    """Scored matches decided by one fit, made at ``at`` from the matches kicked off before it."""

    index: int
    at: datetime
    events: tuple[Event, ...]


def refit_blocks(events: Sequence[Event], params: RetailEvModelParams) -> list[RefitBlock]:
    """The scored matches of ``events`` (kick-off order) in blocks, after the training prefix."""
    scored = events[params.training_matches :]
    blocks: list[RefitBlock] = []
    for index, start in enumerate(range(0, len(scored), params.refit_every_matches)):
        block = tuple(scored[start : start + params.refit_every_matches])
        at = min(decision_times(e.commence_time, params)[0][1] for e in block)
        blocks.append(RefitBlock(index, at, block))
    return blocks


@dataclass(frozen=True)
class Refit:
    index: int
    at: datetime
    training_matches: int
    training_rows: int
    event_ids: tuple[str, ...]


class ModelReplay:
    """Tunes and refits the model over one replay's range, ``events`` in kick-off order.

    A match's training rows are built once, from all of its snapshots.
    """

    def __init__(
        self,
        session: AsyncSession,
        spec: ModelRuleSpec,
        params: RetailEvModelParams,
        events: Sequence[Event],
    ) -> None:
        self._session = session
        self._reader = OddsReader(session)
        self._spec = spec
        self._params = params
        self._events = events
        self._rows: dict[str, list[RetailEvRow]] = {}

    async def _training_rows(self, events: Sequence[Event]) -> list[RetailEvRow]:
        for event in events:
            if event.id not in self._rows:
                snapshots = await self._reader.get_snapshots_for_event(event.id)
                self._rows[event.id] = training_rows(event, snapshots)
                for snapshot in snapshots:
                    self._session.expunge(snapshot)
        return [row for event in events for row in self._rows[event.id]]

    def _market(self) -> str:
        return sport_market(self._events[0].sport_key)

    async def tuned_hyperparameters(self) -> Tuning | None:
        """``None`` when the range has no match to score; ``ValueError`` when nothing can tune."""
        if len(self._events) <= self._params.training_matches:
            return None
        prefix = self._events[: self._params.training_matches]
        folds = [
            (await self._training_rows(fold.train), await self._training_rows(fold.validate))
            for fold in tuning_folds(prefix, self._params)
        ]
        return choose_hyperparameters(self._market(), folds, self._params)

    async def blocks(
        self, tuning: Tuning
    ) -> AsyncIterator[tuple[Refit, RuleSpec, tuple[Event, ...]]]:
        """For each block: its refit record, the rule bound to the refitted model, and its matches."""
        xgb = xgboost_params(tuning.params, self._params.tuning_seed)
        for block in refit_blocks(self._events, self._params):
            training = [e for e in self._events if e.commence_time < block.at]
            rows = await self._training_rows(training)
            try:
                predictor = fit_model(self._market(), rows, xgb)
            except ValueError as exc:
                raise ValueError(f"refit {block.index} at {block.at.isoformat()}: {exc}") from exc
            refit = Refit(
                index=block.index,
                at=block.at,
                training_matches=sum(bool(self._rows[e.id]) for e in training),
                training_rows=len(rows),
                event_ids=tuple(e.id for e in block.events),
            )
            yield refit, self._spec.bind(predictor), block.events
