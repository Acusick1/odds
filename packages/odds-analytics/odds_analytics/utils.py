"""Market hold arithmetic and coverage rendering.

Pure odds arithmetic (american/decimal conversion, implied probability,
devigging, h2h winner, profit-from-odds) lives in ``odds_core.odds_math``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from odds_core.odds_math import decimal_to_implied_probability

if TYPE_CHECKING:
    from rich.table import Table


def calculate_market_hold(decimal_odds_list: list[float]) -> float:
    """Market hold (bookmaker's edge/vig) from the decimal odds of all outcomes."""
    implied_probs = [decimal_to_implied_probability(odds) for odds in decimal_odds_list]
    return sum(implied_probs) - 1.0


def per_book_market_holds(
    prices: Iterable[tuple[str, str, float]],
    required_outcomes: Collection[str],
) -> dict[str, float]:
    """Market hold per bookmaker from ``(bookmaker, outcome, decimal price)`` tuples.

    Books missing any of ``required_outcomes`` are dropped — a partial-market
    sum would understate the hold (possibly negative) with no signal to the
    caller. Duplicate ``(bookmaker, outcome)`` entries keep the last price.
    """
    required = set(required_outcomes)
    by_book: dict[str, dict[str, float]] = {}
    for bookmaker, outcome, price in prices:
        by_book.setdefault(bookmaker, {})[outcome] = price
    return {
        bm: calculate_market_hold(list(outcome_prices.values()))
        for bm, outcome_prices in by_book.items()
        if set(outcome_prices) >= required
    }


def create_tier_coverage_table(
    total_games: int,
    missing_tier_breakdown: dict,
) -> Table:
    """Rich table showing per-tier snapshot coverage."""
    from odds_lambda.fetch_tier import FetchTier
    from rich.table import Table

    tier_table = Table(show_header=True)
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Games with Tier", justify="right")
    tier_table.add_column("Coverage %", justify="right")

    all_tiers = [
        FetchTier.OPENING,
        FetchTier.EARLY,
        FetchTier.SHARP,
        FetchTier.PREGAME,
        FetchTier.CLOSING,
    ]

    for tier in all_tiers:
        games_missing = missing_tier_breakdown.get(tier, 0)
        games_with = total_games - games_missing
        coverage_pct = (games_with / total_games * 100) if total_games > 0 else 0

        if coverage_pct == 100:
            coverage_str = f"[green]{coverage_pct:.1f}%[/green]"
        elif coverage_pct >= 80:
            coverage_str = f"[yellow]{coverage_pct:.1f}%[/yellow]"
        else:
            coverage_str = f"[red]{coverage_pct:.1f}%[/red]"

        tier_table.add_row(tier.value.upper(), f"{games_with}/{total_games}", coverage_str)

    return tier_table
