"""The retail-EV model behind a model candidate rule, refitted walk-forward during a replay.

A replay leaves the first ``training_matches`` of its range to training only, and scores the
rest in blocks of ``refit_every_matches`` in kick-off order. Before each block the model is
refitted on every range match that kicked off before the block's first decision time. Its
rows, and the inputs it predicts from, come from co-timed snapshots only: each is the market
state ``odds_lambda.candidate_rules.build_market_state`` builds at a snapshot's own time, so
a training row is exactly what the rule would have seen at that time.
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
from odds_analytics.training.config import RetailEvArm

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy

# The price-only residual arm with the frozen residual seed-42 trial of
# experiments/results/retail_ev_as_of_t_groups/summary.json (variant price_only), the model
# the ADR-0016 evaluation spec freezes, written out in full so the rule hash covers every value.
MODEL_XGBOOST_PARAMS: dict[str, dict[str, Any]] = {
    "retail_ev_price_only": {
        "n_estimators": 100,
        "max_depth": 4,
        "min_child_weight": 1982,
        "learning_rate": 0.010075960398505468,
        "gamma": 0.0,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.9487910798188237,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 1,
    },
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
    """Rows of every co-timed snapshot in the decision window, against the event's close.

    An event without a close has none.
    """
    ordered = sorted(snapshots, key=lambda s: s.snapshot_time)
    close = event_close(event, ordered, sport_market(event.sport_key))
    if close is None:
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


def fit_model(model: str, market: str, rows: Sequence[RetailEvRow]) -> RetailEvPredictor:
    if not rows:
        raise ValueError("no training rows: no co-timed snapshot of a match with a close")
    features = RetailEvFeatures(market_outcomes(market))
    dataset = RetailEvDataset(rows=list(rows))
    strategy = fit_retail_ev(
        features.matrix(rows),
        dataset.targets.astype(np.float32),
        dataset.loss_weights,
        features.names,
        MODEL_XGBOOST_PARAMS[model],
        RetailEvArm.RESIDUAL,
    )
    return RetailEvPredictor(strategy, features)


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


async def model_blocks(
    session: AsyncSession,
    spec: ModelRuleSpec,
    params: RetailEvModelParams,
    events: Sequence[Event],
) -> AsyncIterator[tuple[Refit, RuleSpec, tuple[Event, ...]]]:
    """For each block: its refit record, the rule bound to the refitted model, and its matches.

    ``events`` is the replay's range in kick-off order. A match's training rows are built once.
    """
    reader = OddsReader(session)
    cache: dict[str, list[RetailEvRow]] = {}
    for block in refit_blocks(events, params):
        training = [e for e in events if e.commence_time < block.at]
        for event in training:
            if event.id not in cache:
                snapshots = await reader.get_snapshots_for_event(event.id)
                cache[event.id] = training_rows(event, snapshots)
                for snapshot in snapshots:
                    session.expunge(snapshot)
        rows = [row for event in training for row in cache[event.id]]
        try:
            predictor = fit_model(params.model, sport_market(block.events[0].sport_key), rows)
        except ValueError as exc:
            raise ValueError(f"refit {block.index} at {block.at.isoformat()}: {exc}") from exc
        refit = Refit(
            index=block.index,
            at=block.at,
            training_matches=sum(bool(cache[e.id]) for e in training),
            training_rows=len(rows),
            event_ids=tuple(e.id for e in block.events),
        )
        yield refit, spec.bind(predictor), block.events
