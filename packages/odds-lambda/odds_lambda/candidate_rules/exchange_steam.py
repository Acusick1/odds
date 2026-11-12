"""The exchange-steam rule: back the longest retail price on an outcome the exchange has moved
towards, while retail has not caught up."""

from __future__ import annotations

from odds_core.market_prices import market_outcomes
from pydantic import Field, NonNegativeFloat, PositiveFloat

from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    LookbackParams,
    RuleParams,
    RuleSpec,
    best_bet,
    longest_retail,
)


class ExchangeSteamParams(RuleParams):
    min_rise: PositiveFloat
    gap: NonNegativeFloat
    lookbacks: tuple[LookbackParams, ...] = Field(min_length=1, max_length=1)


def exchange_steam(state: MarketState, params: ExchangeSteamParams) -> list[BetIntent]:
    """Bet the one outcome whose exchange fair probability rose by at least ``min_rise`` since
    the lookback and whose longest retail price still has EV above ``gap`` against the fair
    price now, the one of the highest EV."""
    before = state.lookbacks[params.lookbacks[0].minutes_before_decision].exchange
    scored: list[tuple[BetIntent, float]] = []
    for line, now in state.exchange.items():
        earlier = before.get(line)
        if earlier is None:
            continue
        for outcome in market_outcomes(state.market):
            longest = longest_retail(state.retail.get(line, {}), outcome)
            if longest is None or now.fair[outcome] - earlier.fair[outcome] < params.min_rise:
                continue
            price, book = longest
            ev = now.fair[outcome] * price - 1
            if ev > params.gap:
                scored.append((BetIntent(outcome, book, price, 1.0, line), ev))
    return best_bet(state.market, scored)


EXCHANGE_STEAM = RuleSpec(exchange_steam, ExchangeSteamParams, frozenset({SharpInput.EXCHANGE}))
