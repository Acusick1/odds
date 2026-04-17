"""Analytics utility functions (risk metrics, market analysis, reporting helpers).

Pure odds arithmetic (american/decimal conversion, implied probability,
devigging, h2h winner, profit-from-odds) lives in ``odds_core.odds_math``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from odds_core.odds_math import calculate_implied_probability

if TYPE_CHECKING:
    from rich.table import Table


def calculate_ev(
    bet_probability: float,
    american_odds: int,
    stake: float = 100.0,
) -> float:
    """Expected value of a bet.

    Examples:
        >>> calculate_ev(0.55, -110, 100)
        4.09
    """
    from odds_core.odds_math import american_to_decimal

    decimal_odds = american_to_decimal(american_odds)
    net_win = stake * (decimal_odds - 1)

    expected_win = bet_probability * net_win
    expected_loss = (1 - bet_probability) * stake
    return expected_win - expected_loss


def calculate_kelly_stake(
    bet_probability: float,
    american_odds: int,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_percentage: float = 0.05,
) -> float:
    """Optimal stake via Kelly Criterion with an optional fractional-Kelly cap."""
    from odds_core.odds_math import american_to_decimal

    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    p = bet_probability
    q = 1 - p

    kelly_percentage = (b * p - q) / b

    if kelly_percentage <= 0:
        return 0.0

    fractional_kelly = kelly_percentage * kelly_fraction
    capped_percentage = min(fractional_kelly, max_stake_percentage)
    return max(0.0, bankroll * capped_percentage)


def calculate_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Sharpe ratio. Returns 0.0 if std dev is 0 or fewer than 2 samples."""
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_dev


def calculate_sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Sortino ratio (Sharpe but penalises downside only)."""
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)

    downside_returns = [min(0, r - risk_free_rate) for r in returns]
    downside_variance = sum(r**2 for r in downside_returns) / len(returns)
    downside_std = math.sqrt(downside_variance)

    if downside_std == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(equity_curve: Sequence[float]) -> tuple[float, float]:
    """Maximum drawdown from an equity curve.

    Returns:
        Tuple of (max_drawdown_amount, max_drawdown_percentage).
    """
    if len(equity_curve) < 2:
        return (0.0, 0.0)

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value

        drawdown = value - peak
        drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0

        if drawdown < max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown_pct

    return (max_dd, max_dd_pct)


def calculate_profit_factor(winning_profit: float, losing_loss: float) -> float:
    """Profit factor (gross profit / gross loss)."""
    if losing_loss == 0:
        return float("inf") if winning_profit > 0 else 0.0

    return winning_profit / abs(losing_loss)


def calculate_market_hold(odds_list: list[int]) -> float:
    """Market hold (bookmaker's edge/vig) from a list of odds for all outcomes."""
    implied_probs = [calculate_implied_probability(odds) for odds in odds_list]
    return sum(implied_probs) - 1.0


def detect_arbitrage(odds_list: list[tuple[str, int]]) -> tuple[bool, float, dict]:
    """Detect arbitrage opportunity across multiple bookmakers.

    Returns:
        Tuple of (has_arbitrage, profit_percentage, stake_distribution).
    """
    implied_probs = [calculate_implied_probability(odds) for _, odds in odds_list]

    total_implied_prob = sum(implied_probs)

    has_arbitrage = total_implied_prob < 1.0

    if not has_arbitrage:
        return (False, 0.0, {})

    profit_percentage = (1 - total_implied_prob) * 100

    stake_distribution = {}
    for (bookmaker, _), implied_prob in zip(odds_list, implied_probs, strict=True):
        stake_pct = (implied_prob / total_implied_prob) * 100
        stake_distribution[bookmaker] = round(stake_pct, 2)

    return (has_arbitrage, round(profit_percentage, 2), stake_distribution)


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
