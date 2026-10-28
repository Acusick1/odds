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
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from odds_core.as_of import by_snapshot
from odds_core.betfair_close import close_quotes, event_close
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Source
from odds_lambda.candidate_rules import (
    RULES,
    MarketState,
    ModelRuleSpec,
    RetailEvModelParams,
    RuleSpec,
    build_market_state,
    decision_times,
)
from odds_lambda.candidate_rules.inputs import RETAIL_SOURCES, SharpInput, rule_quotes

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

_MODEL_SPEC = RULES["retail_ev_model"]
_SHARP_INPUTS: frozenset[SharpInput] = _MODEL_SPEC.sharp_inputs

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
    """The state's quotes as model inputs at its snapshot's time; none without an exchange price.

    The model is defined for markets without lines, whose one line is None."""
    exchange = state.exchange.get(None)
    if exchange is None or state.snapshot_time is None:
        return []
    # The exchange price is read from the state's own snapshot, so it carries that snapshot's
    # source and time basis.
    return quote_inputs(
        state.event_id,
        state.kickoff,
        exchange.source,
        exchange.time_basis,
        state.snapshot_time,
        state.retail.get(None, {}),
        market_outcomes(state.market),
        exchange,
        polymarket=None,
        dynamics=None,
    )


def training_rows(event: Event, market: Market, quoted: Sequence[Quote]) -> list[RetailEvRow]:
    """Rows of every co-timed snapshot in the decision window, against the live Betfair close.

    ``quoted`` holds the event's retail quotes and its close quotes. An event without a live
    Betfair close has none.
    """
    close = event_close(quoted, event.commence_time, market)
    if close is None or close.source is not Source.BETFAIR:
        return []
    first = event.commence_time - DECISION_WINDOW
    last = event.commence_time - DECISION_CUTOFF
    rows: list[RetailEvRow] = []
    for snapshot in by_snapshot(q for q in quoted if q.source in RETAIL_SOURCES):
        time = snapshot[0].snapshot_time
        if not first <= time <= last:
            continue
        state = build_market_state(event, market, snapshot, time, timedelta(0), _SHARP_INPUTS)
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


def tuning_folds(matches: Sequence[Event], params: RetailEvModelParams) -> list[TuningFold]:
    """Expanding walk-forward folds over the tuning matches, in kick-off order."""
    folds: list[TuningFold] = []
    starts = range(params.tuning_min_train_matches, len(matches), params.tuning_step_matches)
    for index, start in enumerate(starts):
        validate = tuple(matches[start : start + params.tuning_step_matches])
        cutoff = validate[0].commence_time - DECISION_WINDOW
        train = tuple(e for e in matches[:start] if e.commence_time <= cutoff)
        folds.append(TuningFold(index, train, validate))
    return folds


def at_decision_times(row: RetailEvRow, params: RetailEvModelParams) -> bool:
    """Whether the row's snapshot could be the input of one of the rule's decisions: stamped
    within ``max_input_age_minutes`` before a decision time."""
    age = timedelta(minutes=params.max_input_age_minutes)
    return any(
        t - age <= row.decision_time <= t for _, t in decision_times(row.commence_time, params)
    )


def training_prefix(events: Sequence[Event], params: RetailEvModelParams) -> int:
    """How many of ``events`` (kick-off order) are training only: the first
    ``training_matches``, and any later match kicking off with the last of them."""
    n = min(params.training_matches, len(events))
    while 0 < n < len(events) and events[n].commence_time == events[n - 1].commence_time:
        n += 1
    return n


@dataclass(frozen=True)
class Tuning:
    """The hyperparameters chosen on the training matches and their walk-forward score.

    ``cv_wmse`` and ``baseline_cv_wmse`` are mean fold loss-weighted MSE of the chosen model and
    of the exchange-at-t baseline, over the ``folds`` folds with training and validation rows.
    ``tuning_matches`` were tuned on, and ``validation_rows`` counts each scored fold's
    validation rows.
    """

    params: dict[str, Any]
    cv_wmse: float
    baseline_cv_wmse: float
    folds: int
    trials: int
    tuning_matches: int = 0
    validation_rows: tuple[int, ...] = ()


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
        counts = [(len(train), len(val)) for train, val in folds]
        raise ValueError(
            "no tuning fold has both training and validation rows "
            f"(training and validation rows per fold: {counts})"
        )

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
        validation_rows=tuple(len(val.y) for _, val in usable),
    )


@dataclass(frozen=True)
class RefitBlock:
    """Scored matches decided by one fit, made at ``at`` from the matches kicked off before it."""

    index: int
    at: datetime
    events: tuple[Event, ...]


def refit_blocks(events: Sequence[Event], params: RetailEvModelParams) -> list[RefitBlock]:
    """The scored matches of ``events`` (kick-off order) in blocks, after the training prefix."""
    scored = events[training_prefix(events, params) :]
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

    A match's training rows are built once, from its quotes in the decision window and its
    close quotes.
    """

    def __init__(
        self,
        session: AsyncSession,
        spec: ModelRuleSpec,
        params: RetailEvModelParams,
        market: Market,
        events: Sequence[Event],
    ) -> None:
        self._session = session
        self._spec = spec
        self._params = params
        self._market = market
        self._events = events
        self._rows: dict[str, list[RetailEvRow]] = {}

    async def _training_rows(self, events: Sequence[Event]) -> list[RetailEvRow]:
        market = self._market
        new = [e for e in {e.id: e for e in events}.values() if e.id not in self._rows]
        closes = await close_quotes(self._session, new, market)
        for event in new:
            retail = await rule_quotes(
                self._session,
                event.id,
                market,
                _SHARP_INPUTS,
                since=event.commence_time - DECISION_WINDOW,
                until=event.commence_time - DECISION_CUTOFF,
            )
            self._rows[event.id] = training_rows(event, market, retail + closes[event.id])
        return [row for event in events for row in self._rows[event.id]]

    async def tuned_hyperparameters(self) -> Tuning | None:
        """Hyperparameters chosen on the training matches whose close exists before the first
        scored decision: those that kicked off before it, as a refit's training matches did.

        Validation rows are those at the rule's decision times (``at_decision_times``); training
        rows cover the whole decision window. ``None`` when the range has no match to score;
        ``ValueError`` when nothing can tune.
        """
        blocks = refit_blocks(self._events, self._params)
        if not blocks:
            return None
        prefix = self._events[: training_prefix(self._events, self._params)]
        matches = [e for e in prefix if e.commence_time < blocks[0].at]
        folds = [
            (
                await self._training_rows(fold.train),
                [
                    row
                    for row in await self._training_rows(fold.validate)
                    if at_decision_times(row, self._params)
                ],
            )
            for fold in tuning_folds(matches, self._params)
        ]
        tuning = choose_hyperparameters(self._market, folds, self._params)
        return replace(tuning, tuning_matches=len(matches))

    async def blocks(
        self, tuning: Tuning
    ) -> AsyncIterator[tuple[Refit, RuleSpec, tuple[Event, ...]]]:
        """For each block: its refit record, the rule bound to the refitted model, and its matches."""
        xgb = xgboost_params(tuning.params, self._params.tuning_seed)
        for block in refit_blocks(self._events, self._params):
            training = [e for e in self._events if e.commence_time < block.at]
            rows = await self._training_rows(training)
            try:
                predictor = fit_model(self._market, rows, xgb)
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


def library_versions() -> dict[str, str]:
    """Versions of the libraries that fit and tune the model, which the rule hash does not pin."""
    import optuna
    import xgboost

    return {"xgboost": xgboost.__version__, "optuna": optuna.__version__, "numpy": np.__version__}
