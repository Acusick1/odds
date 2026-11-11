"""The gap rules: back retail prices whose EV against a sharp fair price exceeds a threshold."""

from __future__ import annotations

from pydantic import PositiveFloat

from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import BetIntent, RuleParams, RuleSpec, gap_bets


class GapParams(RuleParams):
    threshold: PositiveFloat
    stake: PositiveFloat = 1.0


def exchange_gap(state: MarketState, params: GapParams) -> list[BetIntent]:
    """Bet the gap to the exchange fair price on each line with one."""
    fairs = {line: p.fair for line, p in state.exchange.items()}
    return gap_bets(state, fairs, params.threshold, params.stake)


def polymarket_gap(state: MarketState, params: GapParams) -> list[BetIntent]:
    """Bet the gap to Polymarket's fair price on each line with one."""
    fairs = {line: p.fair for line, p in state.polymarket.items()}
    return gap_bets(state, fairs, params.threshold, params.stake)


EXCHANGE_GAP = RuleSpec(exchange_gap, GapParams, frozenset({SharpInput.EXCHANGE}))
POLYMARKET_GAP = RuleSpec(polymarket_gap, GapParams, frozenset({SharpInput.POLYMARKET}))
