"""The outcomes each market prices."""

from __future__ import annotations

from odds_core.vocabulary import PriceOutcome

THREE_WAY_MARKET = "1x2"
TWO_WAY_MARKET = "h2h"
TOTALS_MARKET = "totals"

Outcome = PriceOutcome


def market_outcomes(market: str) -> tuple[Outcome, ...]:
    if market == THREE_WAY_MARKET:
        return (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)
    if market == TWO_WAY_MARKET:
        return (Outcome.HOME, Outcome.AWAY)
    if market == TOTALS_MARKET:
        return (Outcome.OVER, Outcome.UNDER)
    raise ValueError(f"No outcomes defined for market {market!r}")
