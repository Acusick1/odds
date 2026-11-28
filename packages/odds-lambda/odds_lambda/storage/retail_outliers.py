"""Find the corrupt long UK retail prices in one snapshot's prices."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.exchange_price import EXCHANGE_BOOKMAKER, MAX_EXCHANGE_OVERROUND, devig, overround
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Price
from odds_core.vocabulary import Market, Side

# A price at least this many times the next-longest UK retail price on its outcome...
MIN_RATIO_TO_NEXT_LONGEST = 2.0
# ...and above this EV against the exchange price is a corrupt quote (#731, #888).
MIN_EV = 0.5
# Markets whose outcomes share one line, so their exchange backs devig together.
GUARDED_MARKETS: frozenset[Market] = frozenset({Market.ONE_X_TWO, Market.H2H, Market.TOTALS})


def _line(price: Price) -> float | None:
    return None if price.line is None else float(price.line)


def long_retail_outliers(market: Market | str | None, prices: Sequence[Price]) -> list[int]:
    """The indexes in ``prices`` of the UK retail back prices that are at least
    ``MIN_RATIO_TO_NEXT_LONGEST`` times the next-longest UK retail back price on the same
    outcome and line, and whose EV against the same line's exchange fair probability is above
    ``MIN_EV``.

    A line's exchange fair probability is its Betfair Exchange backs devigged, when they price
    every outcome with an overround of at most ``MAX_EXCHANGE_OVERROUND``; a line without one,
    and a market outside ``GUARDED_MARKETS``, has no outliers.
    """
    if market is None or Market(market) not in GUARDED_MARKETS:
        return []
    outcomes = market_outcomes(market)
    exchange: dict[float | None, dict[Outcome, float]] = defaultdict(dict)
    retail: dict[tuple[float | None, Outcome], list[int]] = defaultdict(list)
    for index, price in enumerate(prices):
        if price.side != Side.BACK:
            continue
        if price.bookmaker == EXCHANGE_BOOKMAKER:
            exchange[_line(price)][Outcome(price.outcome)] = float(price.price)
        elif price.bookmaker in UK_RETAIL_BOOKMAKERS:
            retail[(_line(price), Outcome(price.outcome))].append(index)

    outliers: list[int] = []
    for (line, outcome), indexes in retail.items():
        backs = exchange.get(line, {})
        if len(indexes) < 2 or set(backs) != set(outcomes):
            continue
        if overround(backs, outcomes) > MAX_EXCHANGE_OVERROUND:
            continue
        longest, *rest = sorted(indexes, key=lambda i: float(prices[i].price), reverse=True)
        price = float(prices[longest].price)
        next_longest = float(prices[rest[0]].price)
        fair = devig(backs, outcomes)[outcome]
        if price >= MIN_RATIO_TO_NEXT_LONGEST * next_longest and price * fair - 1 > MIN_EV:
            outliers.append(longest)
    return sorted(outliers)
