"""The subsidised-outcome rule: back an outcome a high-margin book prices above the retail
consensus while it takes its margin on the other outcomes."""

from __future__ import annotations

from odds_core.market_prices import market_outcomes
from pydantic import PositiveFloat

from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.retail_consensus import consensus, group_probabilities
from odds_lambda.candidate_rules.rules import BetIntent, RuleParams, RuleSpec, best_bet

# How far a book's implied probability of each other outcome must sit above the consensus.
MIN_OTHER_MARGIN = 0.01


class SubsidisedOutcomeParams(RuleParams):
    threshold: PositiveFloat
    min_overround: PositiveFloat
    stake: PositiveFloat = 1.0


def subsidised_outcome(state: MarketState, params: SubsidisedOutcomeParams) -> list[BetIntent]:
    """Back the one (book, outcome) of highest consensus EV where the book prices every outcome
    with an overround of at least ``min_overround``, its EV against the consensus exceeds
    ``threshold``, and its implied probability of each other outcome is at least
    ``MIN_OTHER_MARGIN`` above the consensus."""
    retail = state.retail.get(None, {})
    outcomes = market_outcomes(state.market)
    groups = group_probabilities(retail, state.market)
    scored: list[tuple[BetIntent, float]] = []
    for book, prices in retail.items():
        if not all(o in prices for o in outcomes):
            continue
        if sum(1 / prices[o] for o in outcomes) - 1 < params.min_overround:
            continue
        q = consensus(groups, book)
        if q is None:
            continue
        for outcome in outcomes:
            ev = q[outcome] * prices[outcome] - 1
            if ev > params.threshold and all(
                1 / prices[o] - q[o] >= MIN_OTHER_MARGIN for o in outcomes if o != outcome
            ):
                scored.append((BetIntent(outcome, book, prices[outcome], params.stake), ev))
    return best_bet(state.market, scored)


SUBSIDISED_OUTCOME = RuleSpec(subsidised_outcome, SubsidisedOutcomeParams, frozenset())
