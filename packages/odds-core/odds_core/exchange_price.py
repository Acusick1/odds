"""The devigged Betfair Exchange price in one snapshot's quotes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from odds_core.market_prices import Outcome, market_outcomes
from odds_core.odds_math import devig_probabilities
from odds_core.quotes import Quote
from odds_core.vocabulary import Side, Source, TimeBasis

EXCHANGE_BOOKMAKER = "betfair_exchange"
# A wider back book is a thin market whose best backs are not a price (ADR-0014).
MAX_EXCHANGE_OVERROUND = 0.05


@dataclass(frozen=True)
class ExchangePrice:
    """Devigged Betfair Exchange back prices from one snapshot."""

    time: datetime
    source: Source
    time_basis: TimeBasis
    in_play: bool
    fair: dict[Outcome, float]
    overround: float


def devig(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> dict[Outcome, float]:
    fair = devig_probabilities(*(1 / prices[o] for o in outcomes))
    return dict(zip(outcomes, fair, strict=True))


def overround(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> float:
    return sum(1 / prices[o] for o in outcomes) - 1


def exchange_price(snapshot: Sequence[Quote], market: str) -> ExchangePrice | None:
    """The exchange price in one snapshot's quotes: its Betfair Exchange back price for every
    outcome of ``market``, or None when any is missing."""
    outcomes = market_outcomes(market)
    backs = {
        Outcome(q.outcome.value): q.price
        for q in snapshot
        if q.bookmaker == EXCHANGE_BOOKMAKER and q.side is Side.BACK
    }
    if not snapshot or set(backs) != set(outcomes):
        return None
    first = snapshot[0]
    return ExchangePrice(
        time=first.snapshot_time,
        source=first.source,
        time_basis=first.time_basis,
        in_play=bool(first.in_play),
        fair=devig(backs, outcomes),
        overround=overround(backs, outcomes),
    )
