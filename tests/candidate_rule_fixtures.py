"""Rules for tests of the code that runs them, in a module whose imports the rule hash can
follow: it imports only the candidate rules."""

from __future__ import annotations

from odds_core.market_prices import Outcome
from odds_lambda.candidate_rules import BetIntent, MarketState, RuleParams

# The estimated EV of every bet the fixture rules take.
ESTIMATED_EV = 0.04


class FixtureRuleParams(RuleParams):
    pass


def bet_home(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    price = state.retail[None]["bet365"][Outcome.HOME]
    return [BetIntent(Outcome.HOME, "bet365", price, 1.0, estimated_ev=ESTIMATED_EV)]


def no_bet(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    return []


def twice(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    return bet_home(state, params) * 2


def bet_longest(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    prices = state.retail[None]["bet365"]
    outcome = max(prices, key=lambda o: prices[o])
    return [BetIntent(outcome, "bet365", prices[outcome], 1.0, estimated_ev=ESTIMATED_EV)]
