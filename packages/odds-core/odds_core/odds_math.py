"""Pure odds-arithmetic helpers.

These utilities are dependency-free and safe to use from any layer of the
pipeline (core, lambda, analytics, mcp). They intentionally avoid any data or
ML dependencies so that the scraper Lambda — which bundles odds-lambda but not
odds-analytics — can call them without pulling in numpy, pandas, etc.
"""

from __future__ import annotations


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds.

    Examples:
        >>> american_to_decimal(-110)
        1.909
        >>> american_to_decimal(+150)
        2.50
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds.

    Examples:
        >>> decimal_to_american(1.909)
        -110
        >>> decimal_to_american(2.50)
        +150
    """
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    return int(-100 / (decimal_odds - 1))


def calculate_implied_probability(american_odds: int) -> float:
    """Implied probability (0-1) from American odds.

    Examples:
        >>> calculate_implied_probability(-110)
        0.524
    """
    return 1 / american_to_decimal(american_odds)


def devig_probabilities(*probs: float) -> tuple[float, ...]:
    """Proportional devigging: remove overround by normalizing to sum=1.

    Accepts any number of implied probabilities (2 for two-way markets,
    3 for 1x2) and returns devigged probabilities preserving the original
    ratios. Returns uniform probabilities if the inputs sum to zero.
    """
    if not probs:
        return ()
    total = sum(probs)
    if total <= 0:
        n = len(probs)
        return tuple(1.0 / n for _ in probs)
    return tuple(p / total for p in probs)


def determine_h2h_winner(home_score: int, away_score: int) -> str:
    """Return ``"home"``, ``"away"``, or ``"draw"`` from final scores."""
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def calculate_profit_from_odds(stake: float, american_odds: int, won: bool) -> float:
    """Profit (positive) or loss (negative) for a bet at given American odds.

    Examples:
        >>> calculate_profit_from_odds(100, -110, True)
        90.91
        >>> calculate_profit_from_odds(100, -110, False)
        -100.0
    """
    if not won:
        return -stake
    decimal_odds = american_to_decimal(american_odds)
    return stake * decimal_odds - stake
