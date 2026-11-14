"""The single-book outlier rule: back a retail price well clear of every other UK retail book's
on that outcome, when the exchange agrees it is long."""

from __future__ import annotations

from odds_core.market_prices import market_outcomes
from pydantic import Field, NonNegativeFloat, PositiveFloat

from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import MarketState
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    EstimateBasis,
    RuleParams,
    RuleSpec,
    best_bet,
    longest_retail,
)


class SingleBookOutlierParams(RuleParams):
    min_ratio: float = Field(gt=1)
    gap: NonNegativeFloat
    stake: PositiveFloat = 1.0


def single_book_outlier(state: MarketState, params: SingleBookOutlierParams) -> list[BetIntent]:
    """Back the one selection of highest exchange EV whose longest retail price, on a line with
    an exchange price, is at least ``min_ratio`` times the longest of any other book on it, and
    whose EV against the exchange fair price exceeds ``gap``."""
    scored: list[BetIntent] = []
    for line, exchange in state.exchange.items():
        retail = state.retail.get(line, {})
        for outcome in market_outcomes(state.market):
            longest = longest_retail(retail, outcome)
            if longest is None:
                continue
            price, book = longest
            others = [p[outcome] for b, p in retail.items() if b != book and outcome in p]
            if not others or price / max(others) < params.min_ratio:
                continue
            ev = exchange.fair[outcome] * price - 1
            if ev > params.gap:
                scored.append(BetIntent(outcome, book, price, params.stake, line, estimated_ev=ev))
    return best_bet(state.market, scored)


SINGLE_BOOK_OUTLIER = RuleSpec(
    single_book_outlier,
    SingleBookOutlierParams,
    frozenset({SharpInput.EXCHANGE}),
    estimate_basis=EstimateBasis.EXCHANGE,
)
