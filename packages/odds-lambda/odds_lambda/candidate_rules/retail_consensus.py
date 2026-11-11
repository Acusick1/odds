"""The retail consensus: a fair price from groups of UK retail books, leaving out the group of
the book it is compared with."""

from __future__ import annotations

from collections.abc import Mapping

from odds_core.market_prices import Outcome, market_outcomes
from odds_core.odds_math import devig_probabilities
from odds_core.vocabulary import Market

BookGroup = frozenset[str]

# Books that share a parent's pricing form one group, so their prices count once.
BOOK_GROUPS: tuple[BookGroup, ...] = (
    frozenset({"paddypower", "skybet"}),
    frozenset({"ladbrokes", "coral"}),
    frozenset({"williamhill", "888sport"}),
    frozenset({"betfred"}),
    frozenset({"betvictor"}),
)
MIN_OTHER_GROUPS = 3


def group_probabilities(
    retail: Mapping[str, Mapping[Outcome, float]], market: Market
) -> dict[BookGroup, dict[Outcome, float]]:
    """Each group with a book that prices every outcome of ``market``, and per outcome the mean
    of those books' proportionally devigged probabilities."""
    outcomes = market_outcomes(market)
    groups: dict[BookGroup, dict[Outcome, float]] = {}
    for group in BOOK_GROUPS:
        devigged = [
            devig_probabilities(*(1 / retail[book][o] for o in outcomes))
            for book in sorted(group)
            if book in retail and all(o in retail[book] for o in outcomes)
        ]
        if devigged:
            groups[group] = {
                o: sum(d[i] for d in devigged) / len(devigged) for i, o in enumerate(outcomes)
            }
    return groups


def other_groups(
    groups: Mapping[BookGroup, Mapping[Outcome, float]], book: str
) -> dict[BookGroup, Mapping[Outcome, float]]:
    """``groups`` without ``book``'s own group; all of them for a book in no group."""
    return {group: p for group, p in groups.items() if book not in group}


def consensus(
    groups: Mapping[BookGroup, Mapping[Outcome, float]], book: str
) -> dict[Outcome, float] | None:
    """The consensus ``book`` is compared with: per outcome, the median of the probabilities of
    the groups other than its own, normalised to sum to 1. None with fewer than
    ``MIN_OTHER_GROUPS`` such groups."""
    others = list(other_groups(groups, book).values())
    if len(others) < MIN_OTHER_GROUPS:
        return None
    medians = {o: _median([p[o] for p in others]) for o in others[0]}
    total = sum(medians.values())
    return {o: m / total for o, m in medians.items()}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
