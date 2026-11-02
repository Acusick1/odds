"""The devigged price of one book's side in one snapshot's quotes."""

from __future__ import annotations

from collections.abc import Sequence

from odds_core.exchange_price import ExchangePrice, devig, overround
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.quotes import Quote
from odds_core.vocabulary import Side


def book_price(
    snapshot: Sequence[Quote],
    market: str,
    line: float | None,
    bookmaker: str,
    side: Side,
    book_side: Side | None = None,
) -> ExchangePrice | None:
    """``bookmaker``'s ``side`` price on ``line`` for every outcome of ``market``, devigged
    proportionally, or None when any outcome lacks one.

    The overround is that of the ``book_side`` prices when the snapshot has them for every
    outcome, else of the ``side`` prices; a book whose ``side`` is a mid has no overround of its
    own, so a ``book_side`` of its back prices gives it a liquidity check.
    """
    outcomes = market_outcomes(market)

    def prices_of(wanted: Side) -> dict[Outcome, float]:
        return {
            q.outcome: q.price
            for q in snapshot
            if q.bookmaker == bookmaker and q.side is wanted and q.line == line
        }

    prices = prices_of(side)
    if set(prices) != set(outcomes):
        return None
    book = prices_of(book_side) if book_side is not None else {}
    first = snapshot[0]
    return ExchangePrice(
        time=first.snapshot_time,
        source=first.source,
        time_basis=first.time_basis,
        in_play=bool(first.in_play),
        fair=devig(prices, outcomes),
        overround=overround(book if set(book) == set(outcomes) else prices, outcomes),
    )
