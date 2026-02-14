"""Utility functions for odds calculations and backtesting metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table


def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.909, 2.50)

    Examples:
        >>> american_to_decimal(-110)
        1.909
        >>> american_to_decimal(+150)
        2.50
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.

    Args:
        decimal_odds: Decimal odds (e.g., 1.909, 2.50)

    Returns:
        American odds (e.g., -110, +150)

    Examples:
        >>> decimal_to_american(1.909)
        -110
        >>> decimal_to_american(2.50)
        +150
    """
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def calculate_implied_probability(american_odds: int) -> float:
    """
    Calculate implied probability from American odds.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability as decimal (e.g., 0.524 = 52.4%)

    Examples:
        >>> calculate_implied_probability(-110)
        0.524
        >>> calculate_implied_probability(+150)
        0.400
    """
    decimal_odds = american_to_decimal(american_odds)
    return 1 / decimal_odds


def devig_probabilities(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Proportional devigging: remove overround by normalizing to sum=1."""
    total = home_prob + away_prob
    if total <= 0:
        return 0.5, 0.5
    return home_prob / total, away_prob / total


def calculate_ev(
    bet_probability: float,
    american_odds: int,
    stake: float = 100.0,
) -> float:
    """
    Calculate expected value of a bet.

    Args:
        bet_probability: True probability of winning (0-1)
        american_odds: Odds being offered
        stake: Bet amount

    Returns:
        Expected value in same units as stake

    Examples:
        >>> calculate_ev(0.55, -110, 100)  # 55% win prob, -110 odds, $100 bet
        4.09  # Positive EV bet
    """
    decimal_odds = american_to_decimal(american_odds)
    # Net profit if win (not including stake back)
    net_win = stake * (decimal_odds - 1)

    # EV = (probability of win * net profit) - (probability of loss * stake)
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
    """
    Calculate optimal stake using Kelly Criterion.

    Args:
        bet_probability: True probability of winning (0-1)
        american_odds: Odds being offered
        bankroll: Current bankroll
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
        max_stake_percentage: Maximum percentage of bankroll to bet

    Returns:
        Stake amount

    Notes:
        - Returns 0 if bet has negative expected value
        - Caps stake at max_stake_percentage of bankroll
        - Quarter-Kelly (0.25) is recommended for practical betting
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1  # Net odds (profit per unit)
    p = bet_probability
    q = 1 - p

    # Kelly formula: f = (bp - q) / b
    kelly_percentage = (b * p - q) / b

    # Negative Kelly means negative EV - don't bet
    if kelly_percentage <= 0:
        return 0.0

    # Apply Kelly fraction (e.g., quarter-Kelly)
    fractional_kelly = kelly_percentage * kelly_fraction

    # Cap at max percentage
    capped_percentage = min(fractional_kelly, max_stake_percentage)

    stake = bankroll * capped_percentage

    return max(0.0, stake)


def calculate_profit_from_odds(stake: float, american_odds: int, won: bool) -> float:
    """
    Calculate profit/loss from a bet.

    Args:
        stake: Amount wagered
        american_odds: Odds bet was placed at
        won: Whether the bet won

    Returns:
        Profit if won (positive), loss if lost (negative)

    Examples:
        >>> calculate_profit_from_odds(100, -110, True)
        90.91  # Win $90.91 on $100 bet
        >>> calculate_profit_from_odds(100, -110, False)
        -100.0  # Lose $100 stake
        >>> calculate_profit_from_odds(100, +150, True)
        150.0  # Win $150 on $100 bet
    """
    if not won:
        return -stake

    decimal_odds = american_to_decimal(american_odds)
    total_return = stake * decimal_odds
    profit = total_return - stake
    return profit


def calculate_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio from sequence of returns.

    Args:
        returns: Sequence of returns (e.g., daily profits)
        risk_free_rate: Risk-free rate of return (default 0)

    Returns:
        Sharpe ratio (higher is better, >1.0 is good, >2.0 is excellent)

    Notes:
        Returns 0.0 if standard deviation is 0 or insufficient data
    """
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
    """
    Calculate Sortino ratio (Sharpe but only penalizes downside volatility).

    Args:
        returns: Sequence of returns (e.g., daily profits)
        risk_free_rate: Risk-free rate of return (default 0)

    Returns:
        Sortino ratio (higher is better)

    Notes:
        Like Sharpe but only considers negative deviations (downside risk)
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Only calculate downside deviation (negative returns)
    downside_returns = [min(0, r - risk_free_rate) for r in returns]
    downside_variance = sum(r**2 for r in downside_returns) / len(returns)
    downside_std = math.sqrt(downside_variance)

    if downside_std == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(equity_curve: Sequence[float]) -> tuple[float, float]:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Sequence of bankroll values over time

    Returns:
        Tuple of (max_drawdown_amount, max_drawdown_percentage)

    Examples:
        >>> calculate_max_drawdown([10000, 10500, 10200, 9800, 10100])
        (-700.0, -6.67)  # Drew down $700 (6.67%) from peak of $10,500
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
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        winning_profit: Total profit from winning bets (positive)
        losing_loss: Total loss from losing bets (positive, will be treated as absolute)

    Returns:
        Profit factor (>1.0 is profitable, >2.0 is very good)

    Examples:
        >>> calculate_profit_factor(1500, 1000)
        1.5  # Made $1.50 for every $1.00 lost
    """
    if losing_loss == 0:
        return float("inf") if winning_profit > 0 else 0.0

    return winning_profit / abs(losing_loss)


def calculate_market_hold(odds_list: list[int]) -> float:
    """
    Calculate market hold (bookmaker's edge/vig) from a list of odds.

    Args:
        odds_list: List of American odds for all outcomes in a market

    Returns:
        Market hold as decimal (e.g., 0.048 = 4.8% hold)

    Examples:
        >>> calculate_market_hold([-110, -110])
        0.048  # Standard two-way market with 4.8% hold
        >>> calculate_market_hold([+200, -150])
        -0.067  # Arbitrage opportunity (negative hold)
    """
    implied_probs = [calculate_implied_probability(odds) for odds in odds_list]
    total_implied_prob = sum(implied_probs)
    return total_implied_prob - 1.0


def detect_arbitrage(odds_list: list[tuple[str, int]]) -> tuple[bool, float, dict]:
    """
    Detect arbitrage opportunity across multiple bookmakers.

    Args:
        odds_list: List of (bookmaker_key, american_odds) tuples for each outcome

    Returns:
        Tuple of (has_arbitrage, profit_percentage, stake_distribution)

    Examples:
        >>> detect_arbitrage([("pinnacle", +150), ("fanduel", -165)])
        (True, 2.3, {"pinnacle": 40.0, "fanduel": 60.0})  # 2.3% risk-free profit
    """
    # Convert to implied probabilities
    implied_probs = [calculate_implied_probability(odds) for _, odds in odds_list]

    # Sum of implied probabilities < 1.0 means arbitrage exists
    total_implied_prob = sum(implied_probs)

    has_arbitrage = total_implied_prob < 1.0

    if not has_arbitrage:
        return (False, 0.0, {})

    # Calculate profit percentage
    profit_percentage = (1 - total_implied_prob) * 100

    # Calculate optimal stake distribution (percentage of total stake)
    stake_distribution = {}
    for (bookmaker, _), implied_prob in zip(odds_list, implied_probs, strict=True):
        stake_pct = (implied_prob / total_implied_prob) * 100
        stake_distribution[bookmaker] = round(stake_pct, 2)

    return (has_arbitrage, round(profit_percentage, 2), stake_distribution)


def create_tier_coverage_table(
    total_games: int,
    missing_tier_breakdown: dict,  # dict[FetchTier, int] but avoiding circular import
) -> Table:
    """
    Create a Rich table showing tier coverage breakdown.

    Extracted from validate.py to enable reuse across commands (validate, gaps, etc.).

    Args:
        total_games: Total number of games
        missing_tier_breakdown: Dict mapping FetchTier to count of games missing it

    Returns:
        Rich Table object ready to display

    Example:
        from rich.console import Console
        from odds_analytics.utils import create_tier_coverage_table
        from odds_lambda.fetch_tier import FetchTier

        console = Console()
        table = create_tier_coverage_table(
            total_games=50,
            missing_tier_breakdown={FetchTier.CLOSING: 5, FetchTier.OPENING: 2}
        )
        console.print(table)
    """
    from odds_lambda.fetch_tier import FetchTier
    from rich.table import Table

    tier_table = Table(show_header=True)
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Games with Tier", justify="right")
    tier_table.add_column("Coverage %", justify="right")

    # Use tier priority order (OPENING to CLOSING)
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

        # Color coding
        if coverage_pct == 100:
            coverage_str = f"[green]{coverage_pct:.1f}%[/green]"
        elif coverage_pct >= 80:
            coverage_str = f"[yellow]{coverage_pct:.1f}%[/yellow]"
        else:
            coverage_str = f"[red]{coverage_pct:.1f}%[/red]"

        tier_table.add_row(tier.value.upper(), f"{games_with}/{total_games}", coverage_str)

    return tier_table
