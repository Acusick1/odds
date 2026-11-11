"""The last-result rule: back a side whose previous league match was a heavy loss."""

from __future__ import annotations

from odds_core.market_prices import Outcome
from pydantic import PositiveFloat, PositiveInt

from odds_lambda.candidate_rules.fixture_context import FixtureInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    RuleParams,
    RuleSpec,
    best_bet,
    longest_retail,
)


class LastResultParams(RuleParams):
    min_margin: PositiveInt
    stake: PositiveFloat = 1.0


def last_result_overreaction(state: MarketState, params: LastResultParams) -> list[BetIntent]:
    """Bet the longest retail price of the home or away side whose previous league match was a
    loss by at least ``min_margin`` goals.

    The rule has no fair price, so every qualifying bet scores the same and ``best_bet``'s tie
    order, home first, picks the one when both sides qualify.
    """
    retail = state.retail.get(None, {})
    scored: list[tuple[BetIntent, float]] = []
    for side in (Outcome.HOME, Outcome.AWAY):
        previous = state.fixture.previous.get(side)
        longest = longest_retail(retail, side)
        if previous is None or longest is None or previous.goal_difference > -params.min_margin:
            continue
        price, book = longest
        scored.append((BetIntent(side, book, price, params.stake), 0.0))
    return best_bet(state.market, scored)


LAST_RESULT_OVERREACTION = RuleSpec(
    last_result_overreaction,
    LastResultParams,
    frozenset(),
    fixture_inputs=frozenset({FixtureInput.PREVIOUS_RESULT}),
)
