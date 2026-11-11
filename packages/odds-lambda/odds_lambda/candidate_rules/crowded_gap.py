"""The crowded-kick-off gap rule: the exchange gap on a match with many concurrent kick-offs."""

from __future__ import annotations

from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt

from odds_lambda.candidate_rules.fixture_context import FixtureInput
from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import BetIntent, RuleParams, RuleSpec, best_bet, gap_bets


class CrowdedGapParams(RuleParams):
    min_concurrent_kickoffs: PositiveInt
    threshold: NonNegativeFloat
    stake: PositiveFloat = 1.0


def crowded_gap(state: MarketState, params: CrowdedGapParams) -> list[BetIntent]:
    """Bet the one exchange-gap bet of the highest EV when the match, counted with itself, has
    at least ``min_concurrent_kickoffs`` kick-offs within the concurrent window."""
    concurrent = state.fixture.concurrent_kickoffs
    if concurrent is None or concurrent < params.min_concurrent_kickoffs:
        return []
    fairs = {line: p.fair for line, p in state.exchange.items()}
    bets = gap_bets(state, fairs, params.threshold, params.stake)
    return best_bet(
        state.market, ((b, fairs[b.line][b.outcome] * b.price_decimal - 1) for b in bets)
    )


CROWDED_GAP = RuleSpec(
    crowded_gap,
    CrowdedGapParams,
    frozenset({SharpInput.EXCHANGE}),
    fixture_inputs=frozenset({FixtureInput.CONCURRENT_KICKOFFS}),
)
