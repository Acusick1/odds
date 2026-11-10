"""Rules for tests of the code that runs them, in a module whose imports the rule hash can
follow: it imports only the candidate rules."""

from __future__ import annotations

from odds_core.market_prices import Outcome
from odds_lambda.candidate_rules import BetIntent, MarketState, RuleParams


class FixtureRuleParams(RuleParams):
    pass


def bet_home(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    return [BetIntent(Outcome.HOME, "bet365", state.retail[None]["bet365"][Outcome.HOME], 1.0)]


def no_bet(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    return []


def twice(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    return bet_home(state, params) * 2


def bet_longest(state: MarketState, params: FixtureRuleParams) -> list[BetIntent]:
    prices = state.retail[None]["bet365"]
    outcome = max(prices, key=lambda o: prices[o])
    return [BetIntent(outcome, "bet365", prices[outcome], 1.0)]
