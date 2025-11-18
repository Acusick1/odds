"""Table generation functions for backtest reports using Great Tables."""

from __future__ import annotations

import pandas as pd
from great_tables import GT, html, loc, style

from odds_analytics.backtesting.models import BacktestResult


def create_risk_metrics_table(result: BacktestResult) -> str:
    """Create formatted risk metrics table using Great Tables.

    Args:
        result: BacktestResult object containing risk metrics

    Returns:
        HTML string with formatted table
    """
    # Create DataFrame with risk metrics
    data = {
        "Metric": [
            "Max Drawdown",
            "Max Drawdown %",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Profit Factor",
            "Longest Win Streak",
            "Longest Loss Streak",
            "Average Win",
            "Average Loss",
            "Largest Win",
            "Largest Loss",
        ],
        "Value": [
            f"${abs(result.max_drawdown):,.2f}",
            f"{result.max_drawdown_percentage:.2f}%",
            f"{result.sharpe_ratio:.2f}",
            f"{result.sortino_ratio:.2f}",
            f"{result.calmar_ratio:.2f}",
            f"{result.profit_factor:.2f}",
            str(result.longest_winning_streak),
            str(result.longest_losing_streak),
            f"${result.average_win:,.2f}",
            f"${abs(result.average_loss):,.2f}",
            f"${result.largest_win:,.2f}",
            f"${abs(result.largest_loss):,.2f}",
        ],
    }

    df = pd.DataFrame(data)

    # Create Great Tables object
    gt_table = (
        GT(df)
        .tab_header(title="Risk & Performance Metrics")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(align="left"),
            locations=loc.body(columns="Metric"),
        )
        .tab_style(
            style=style.text(align="right"),
            locations=loc.body(columns="Value"),
        )
        .tab_options(
            table_font_size="14px",
            heading_background_color="#667eea",
            heading_title_font_weight="bold",
            column_labels_background_color="#f8f9fa",
            table_width="100%",
        )
    )

    return gt_table.as_raw_html()


def create_market_breakdown_table(result: BacktestResult) -> str:
    """Create formatted market breakdown table using Great Tables.

    Args:
        result: BacktestResult object containing market breakdown data

    Returns:
        HTML string with formatted table
    """
    # Map market keys to readable names
    market_names = {"h2h": "Moneyline", "spreads": "Spreads", "totals": "Totals"}

    # Create DataFrame with market breakdown
    data = []
    for market, stats in result.market_breakdown.items():
        data.append(
            {
                "Market": market_names.get(market, market),
                "Bets": stats.bets,
                "Win Rate": f"{stats.win_rate:.1f}%",
                "Profit": stats.profit,
                "ROI": stats.roi,
                "Total Wagered": stats.total_wagered,
            }
        )

    df = pd.DataFrame(data)

    # Sort by ROI descending
    df = df.sort_values("ROI", ascending=False)

    # Create Great Tables object
    gt_table = (
        GT(df)
        .tab_header(title="Performance by Market Type")
        .fmt_currency(columns=["Profit", "Total Wagered"], currency="USD")
        .fmt_number(columns=["ROI"], decimals=2, pattern="{x}%")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(align="left"),
            locations=loc.body(columns="Market"),
        )
        .tab_style(
            style=style.text(align="center"),
            locations=loc.body(columns=["Bets", "Win Rate"]),
        )
        .tab_style(
            style=style.text(align="right"),
            locations=loc.body(columns=["Profit", "ROI", "Total Wagered"]),
        )
        .data_color(
            columns="ROI",
            palette=["#dc3545", "#28a745"],
            domain=[-100, 100],
        )
        .tab_options(
            table_font_size="14px",
            heading_background_color="#667eea",
            heading_title_font_weight="bold",
            column_labels_background_color="#f8f9fa",
            table_width="100%",
        )
    )

    return gt_table.as_raw_html()


def create_bookmaker_breakdown_table(result: BacktestResult) -> str:
    """Create formatted bookmaker breakdown table using Great Tables.

    Args:
        result: BacktestResult object containing bookmaker breakdown data

    Returns:
        HTML string with formatted table
    """
    # Create DataFrame with bookmaker breakdown
    data = []
    for bookmaker, stats in result.bookmaker_breakdown.items():
        data.append(
            {
                "Bookmaker": bookmaker.title(),
                "Bets": stats.bets,
                "Win Rate": f"{stats.win_rate:.1f}%",
                "Profit": stats.profit,
                "ROI": stats.roi,
                "Total Wagered": stats.total_wagered,
            }
        )

    df = pd.DataFrame(data)

    # Sort by ROI descending
    df = df.sort_values("ROI", ascending=False)

    # Create Great Tables object
    gt_table = (
        GT(df)
        .tab_header(title="Performance by Bookmaker")
        .fmt_currency(columns=["Profit", "Total Wagered"], currency="USD")
        .fmt_number(columns=["ROI"], decimals=2, pattern="{x}%")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(align="left"),
            locations=loc.body(columns="Bookmaker"),
        )
        .tab_style(
            style=style.text(align="center"),
            locations=loc.body(columns=["Bets", "Win Rate"]),
        )
        .tab_style(
            style=style.text(align="right"),
            locations=loc.body(columns=["Profit", "ROI", "Total Wagered"]),
        )
        .data_color(
            columns="ROI",
            palette=["#dc3545", "#28a745"],
            domain=[-100, 100],
        )
        .tab_options(
            table_font_size="14px",
            heading_background_color="#667eea",
            heading_title_font_weight="bold",
            column_labels_background_color="#f8f9fa",
            table_width="100%",
        )
    )

    return gt_table.as_raw_html()


def create_monthly_performance_table(result: BacktestResult) -> str:
    """Create formatted monthly performance table using Great Tables.

    Args:
        result: BacktestResult object containing monthly performance data

    Returns:
        HTML string with formatted table
    """
    # Create DataFrame with monthly performance
    data = []
    for month_stats in result.monthly_performance:
        data.append(
            {
                "Month": month_stats.month,
                "Bets": month_stats.bets,
                "Win Rate": f"{month_stats.win_rate:.1f}%",
                "Profit": month_stats.profit,
                "ROI": month_stats.roi,
                "Start Bankroll": month_stats.start_bankroll,
                "End Bankroll": month_stats.end_bankroll,
            }
        )

    df = pd.DataFrame(data)

    # Create Great Tables object
    gt_table = (
        GT(df)
        .tab_header(title="Monthly Performance Breakdown")
        .fmt_currency(
            columns=["Profit", "Start Bankroll", "End Bankroll"], currency="USD"
        )
        .fmt_number(columns=["ROI"], decimals=2, pattern="{x}%")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(align="left"),
            locations=loc.body(columns="Month"),
        )
        .tab_style(
            style=style.text(align="center"),
            locations=loc.body(columns=["Bets", "Win Rate"]),
        )
        .tab_style(
            style=style.text(align="right"),
            locations=loc.body(
                columns=["Profit", "ROI", "Start Bankroll", "End Bankroll"]
            ),
        )
        .data_color(
            columns="Profit",
            palette=["#dc3545", "#28a745"],
            domain=[-1000, 1000],
        )
        .tab_options(
            table_font_size="14px",
            heading_background_color="#667eea",
            heading_title_font_weight="bold",
            column_labels_background_color="#f8f9fa",
            table_width="100%",
        )
    )

    return gt_table.as_raw_html()


def create_bet_summary_table(result: BacktestResult) -> str:
    """Create formatted bet summary statistics table using Great Tables.

    Args:
        result: BacktestResult object containing bet statistics

    Returns:
        HTML string with formatted table
    """
    # Create DataFrame with bet statistics
    data = {
        "Metric": [
            "Total Bets",
            "Winning Bets",
            "Losing Bets",
            "Push Bets",
            "Win Rate",
            "Total Wagered",
            "Average Stake",
            "Average Odds",
            "Median Odds",
        ],
        "Value": [
            str(result.total_bets),
            str(result.winning_bets),
            str(result.losing_bets),
            str(result.push_bets),
            f"{result.win_rate:.2f}%",
            f"${result.total_wagered:,.2f}",
            f"${result.average_stake:,.2f}",
            f"{result.average_odds:+.0f}",
            f"{result.median_odds:+.0f}",
        ],
    }

    df = pd.DataFrame(data)

    # Create Great Tables object
    gt_table = (
        GT(df)
        .tab_header(title="Bet Statistics")
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(align="left"),
            locations=loc.body(columns="Metric"),
        )
        .tab_style(
            style=style.text(align="right"),
            locations=loc.body(columns="Value"),
        )
        .tab_options(
            table_font_size="14px",
            heading_background_color="#667eea",
            heading_title_font_weight="bold",
            column_labels_background_color="#f8f9fa",
            table_width="100%",
        )
    )

    return gt_table.as_raw_html()
