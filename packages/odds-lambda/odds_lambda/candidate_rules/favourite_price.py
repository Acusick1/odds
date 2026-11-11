"""The favourite-price rule: back the longest UK retail price on a heavy favourite."""

from __future__ import annotations

from odds_core.market_prices import market_outcomes
from pydantic import Field, PositiveFloat

from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    RuleParams,
    RuleSpec,
    best_bet,
    longest_retail,
)


class FavouriteParams(RuleParams):
    min_probability: float = Field(gt=0.5, lt=1)
    threshold: float = Field(allow_inf_nan=False)
    stake: PositiveFloat = 1.0


def favourite_price(state: MarketState, params: FavouriteParams) -> list[BetIntent]:
    """Bet the outcome whose exchange fair probability is at least ``min_probability``, at its
    longest retail price, when its EV against the exchange exceeds ``threshold``."""
    exchange = state.exchange.get(None)
    if exchange is None:
        return []
    retail = state.retail.get(None, {})
    scored: list[tuple[BetIntent, float]] = []
    for outcome in market_outcomes(state.market):
        fair = exchange.fair[outcome]
        longest = longest_retail(retail, outcome)
        if fair < params.min_probability or longest is None:
            continue
        price, book = longest
        ev = fair * price - 1
        if ev > params.threshold:
            scored.append((BetIntent(outcome, book, price, params.stake), ev))
    return best_bet(state.market, scored)


FAVOURITE_PRICE = RuleSpec(favourite_price, FavouriteParams, frozenset({SharpInput.EXCHANGE}))
