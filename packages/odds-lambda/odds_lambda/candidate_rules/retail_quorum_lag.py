"""The retail-quorum lag rule: back a retail price left unchanged while other book groups moved
that outcome up."""

from __future__ import annotations

from odds_core.market_prices import market_outcomes
from pydantic import Field, PositiveFloat

from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.retail_consensus import (
    MIN_OTHER_GROUPS,
    consensus,
    group_probabilities,
    other_groups,
)
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    EstimateBasis,
    LookbackParams,
    RuleParams,
    RuleSpec,
    best_bet,
)

# The rise in a group's probability of the outcome over the lookback that counts as a move.
MIN_GROUP_RISE = 0.01


class RetailQuorumLagParams(RuleParams):
    threshold: PositiveFloat
    lookbacks: tuple[LookbackParams, ...] = Field(min_length=1, max_length=1)
    stake: PositiveFloat = 1.0


def retail_quorum_lag(state: MarketState, params: RetailQuorumLagParams) -> list[BetIntent]:
    """Back the one (book, outcome) of highest consensus EV whose price is the same at the
    decision and the lookback, while at least ``MIN_OTHER_GROUPS`` groups other than the book's
    own, present at both points, raised their probability of the outcome by
    ``MIN_GROUP_RISE``, and whose EV against the consensus at the decision exceeds
    ``threshold``."""
    now = state.retail.get(None, {})
    then = state.lookbacks[params.lookbacks[0].minutes_before_decision].retail.get(None, {})
    groups_now = group_probabilities(now, state.market)
    groups_then = group_probabilities(then, state.market)
    scored: list[BetIntent] = []
    for book, prices in now.items():
        q = consensus(groups_now, book)
        if q is None:
            continue
        for outcome in market_outcomes(state.market):
            price = prices.get(outcome)
            if price is None or then.get(book, {}).get(outcome) != price:
                continue
            risen = [
                group
                for group, p in other_groups(groups_now, book).items()
                if group in groups_then
                and p[outcome] - groups_then[group][outcome] >= MIN_GROUP_RISE
            ]
            ev = q[outcome] * price - 1
            if len(risen) >= MIN_OTHER_GROUPS and ev > params.threshold:
                scored.append(BetIntent(outcome, book, price, params.stake, estimated_ev=ev))
    return best_bet(state.market, scored)


RETAIL_QUORUM_LAG = RuleSpec(
    retail_quorum_lag,
    RetailQuorumLagParams,
    frozenset(),
    estimate_basis=EstimateBasis.RETAIL_CONSENSUS,
)
