"""The retail-EV model rule: back retail prices whose EV a fitted model predicts exceeds a
threshold."""

from __future__ import annotations

from typing import Literal

from odds_core.market_prices import market_outcomes
from odds_core.vocabulary import Market
from pydantic import PositiveFloat, PositiveInt, model_validator

from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    EvPredictor,
    ModelRuleSpec,
    RuleParams,
    longest_retail,
)


class RetailEvModelParams(RuleParams):
    """``model`` names the model. A replay leaves the first ``training_matches`` of its range
    to training only and refits before every ``refit_every_matches`` scored matches.

    Its hyperparameters are tuned once, on the training matches alone: ``tuning_trials``
    trials seeded with ``tuning_seed`` (also the model's random state), scored on expanding
    walk-forward folds that first train on ``tuning_min_train_matches`` matches and then
    validate ``tuning_step_matches`` matches at a time.
    """

    model: Literal["retail_ev_price_only"]
    threshold: PositiveFloat
    training_matches: PositiveInt
    refit_every_matches: PositiveInt
    tuning_trials: PositiveInt
    tuning_seed: int
    tuning_min_train_matches: PositiveInt
    tuning_step_matches: PositiveInt
    stake: PositiveFloat = 1.0

    @model_validator(mode="after")
    def _folds_fit_the_training_matches(self) -> RetailEvModelParams:
        if self.tuning_min_train_matches >= self.training_matches:
            raise ValueError(
                f"tuning_min_train_matches {self.tuning_min_train_matches} leaves no match of "
                f"the {self.training_matches} training matches to validate"
            )
        return self


def retail_ev_model(
    state: MarketState, params: RetailEvModelParams, predict: EvPredictor
) -> list[BetIntent]:
    """Back the longest UK retail price of each outcome whose predicted EV exceeds the threshold."""
    if None not in state.exchange:
        return []
    predicted = predict(state)
    bets: list[BetIntent] = []
    for outcome in market_outcomes(state.market):
        longest = longest_retail(state.retail.get(None, {}), outcome)
        if longest is None:
            continue
        price, book = longest
        ev = predicted.get((outcome, book))
        if ev is not None and ev > params.threshold:
            bets.append(BetIntent(outcome, book, price, params.stake))
    return bets


RETAIL_EV_MODEL = ModelRuleSpec(
    retail_ev_model,
    RetailEvModelParams,
    frozenset({SharpInput.EXCHANGE}),
    ("odds_analytics.candidate_model.ModelReplay",),
    frozenset({Market.ONE_X_TWO, Market.H2H}),
)
