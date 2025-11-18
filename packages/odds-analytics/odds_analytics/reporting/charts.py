"""Chart generation functions for backtest reports using Plotly."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from odds_analytics.backtesting.models import BacktestResult


def create_equity_curve_chart(result: BacktestResult) -> str:
    """Create interactive equity curve visualization showing bankroll over time.

    Args:
        result: BacktestResult object containing equity curve data

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    dates = [point.date for point in result.equity_curve]
    bankroll = [point.bankroll for point in result.equity_curve]
    cumulative_profit = [point.cumulative_profit for point in result.equity_curve]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Bankroll Over Time", "Cumulative Profit"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Bankroll line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bankroll,
            mode="lines",
            name="Bankroll",
            line={"color": "#667eea", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.1)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Bankroll:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add initial bankroll reference line
    fig.add_hline(
        y=result.initial_bankroll,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Bankroll",
        annotation_position="right",
        row=1,
        col=1,
    )

    # Cumulative profit line
    profit_color = "#28a745" if result.total_profit > 0 else "#dc3545"
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_profit,
            mode="lines",
            name="Cumulative Profit",
            line={"color": profit_color, "width": 2},
            fill="tozeroy",
            fillcolor=f"rgba({int(profit_color[1:3], 16)}, {int(profit_color[3:5], 16)}, {int(profit_color[5:7], 16)}, 0.1)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Profit:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Bankroll ($)", row=1, col=1)
    fig.update_yaxes(title_text="Profit ($)", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode="x unified",
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    return fig.to_html(include_plotlyjs="cdn", div_id="equity-curve-chart")


def create_monthly_performance_chart(result: BacktestResult) -> str:
    """Create monthly performance bar chart showing profit by month.

    Args:
        result: BacktestResult object containing monthly performance data

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    months = [stats.month for stats in result.monthly_performance]
    profits = [stats.profit for stats in result.monthly_performance]
    roi = [stats.roi for stats in result.monthly_performance]
    bets = [stats.bets for stats in result.monthly_performance]

    colors = ["#28a745" if p > 0 else "#dc3545" for p in profits]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=months,
            y=profits,
            marker_color=colors,
            text=[f"${p:,.2f}" for p in profits],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Profit: $%{y:,.2f}<br>"
                "ROI: %{customdata[0]:.2f}%<br>"
                "Bets: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(roi, bets)),
        )
    )

    fig.update_layout(
        title="Monthly Performance",
        xaxis_title="Month",
        yaxis_title="Profit ($)",
        height=400,
        showlegend=False,
        hovermode="x",
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig.to_html(include_plotlyjs="cdn", div_id="monthly-performance-chart")


def create_market_breakdown_chart(result: BacktestResult) -> str:
    """Create market type breakdown showing ROI by market (h2h, spreads, totals).

    Args:
        result: BacktestResult object containing market breakdown data

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    markets = list(result.market_breakdown.keys())
    roi = [stats.roi for stats in result.market_breakdown.values()]
    profits = [stats.profit for stats in result.market_breakdown.values()]
    bets = [stats.bets for stats in result.market_breakdown.values()]
    win_rates = [stats.win_rate for stats in result.market_breakdown.values()]

    # Map market keys to readable names
    market_names = {"h2h": "Moneyline", "spreads": "Spreads", "totals": "Totals"}
    display_names = [market_names.get(m, m) for m in markets]

    colors = ["#667eea", "#764ba2", "#f093fb"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=display_names,
            y=roi,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in roi],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ROI: %{y:.2f}%<br>"
                "Profit: $%{customdata[0]:,.2f}<br>"
                "Win Rate: %{customdata[1]:.1f}%<br>"
                "Bets: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(profits, win_rates, bets)),
        )
    )

    fig.update_layout(
        title="Performance by Market Type",
        xaxis_title="Market",
        yaxis_title="ROI (%)",
        height=400,
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig.to_html(include_plotlyjs="cdn", div_id="market-breakdown-chart")


def create_bookmaker_breakdown_chart(result: BacktestResult) -> str:
    """Create bookmaker ROI comparison showing performance across different books.

    Args:
        result: BacktestResult object containing bookmaker breakdown data

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    bookmakers = list(result.bookmaker_breakdown.keys())
    roi = [stats.roi for stats in result.bookmaker_breakdown.values()]
    profits = [stats.profit for stats in result.bookmaker_breakdown.values()]
    bets = [stats.bets for stats in result.bookmaker_breakdown.values()]
    win_rates = [stats.win_rate for stats in result.bookmaker_breakdown.values()]

    # Sort by ROI descending
    sorted_data = sorted(
        zip(bookmakers, roi, profits, bets, win_rates), key=lambda x: x[1], reverse=True
    )
    bookmakers, roi, profits, bets, win_rates = zip(*sorted_data) if sorted_data else ([], [], [], [], [])

    colors = ["#28a745" if r > 0 else "#dc3545" for r in roi]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bookmakers,
            y=roi,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in roi],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ROI: %{y:.2f}%<br>"
                "Profit: $%{customdata[0]:,.2f}<br>"
                "Win Rate: %{customdata[1]:.1f}%<br>"
                "Bets: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(profits, win_rates, bets)),
        )
    )

    fig.update_layout(
        title="Performance by Bookmaker",
        xaxis_title="Bookmaker",
        yaxis_title="ROI (%)",
        height=400,
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 80, "b": 100},
        xaxis_tickangle=-45,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig.to_html(include_plotlyjs="cdn", div_id="bookmaker-breakdown-chart")


def create_drawdown_chart(result: BacktestResult) -> str:
    """Create drawdown progression analysis showing peak-to-trough declines.

    Args:
        result: BacktestResult object containing equity curve data

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    dates = [point.date for point in result.equity_curve]
    bankroll = [point.bankroll for point in result.equity_curve]

    # Calculate running maximum and drawdown
    running_max = []
    drawdown = []
    current_max = bankroll[0] if bankroll else 0

    for value in bankroll:
        current_max = max(current_max, value)
        running_max.append(current_max)
        drawdown.append(value - current_max)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Bankroll vs Peak", "Drawdown"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    # Bankroll vs running max
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bankroll,
            mode="lines",
            name="Bankroll",
            line={"color": "#667eea", "width": 2},
            hovertemplate="<b>Bankroll:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=running_max,
            mode="lines",
            name="Peak",
            line={"color": "#28a745", "width": 2, "dash": "dash"},
            hovertemplate="<b>Peak:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"color": "#dc3545", "width": 0},
            fill="tozeroy",
            fillcolor="rgba(220, 53, 69, 0.3)",
            hovertemplate="<b>Drawdown:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add max drawdown annotation
    if drawdown:
        max_dd_idx = drawdown.index(min(drawdown))
        fig.add_annotation(
            x=dates[max_dd_idx],
            y=drawdown[max_dd_idx],
            text=f"Max DD: ${abs(drawdown[max_dd_idx]):,.2f}",
            showarrow=True,
            arrowhead=2,
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        legend={"x": 0, "y": 1.15, "orientation": "h"},
        hovermode="x unified",
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    return fig.to_html(include_plotlyjs="cdn", div_id="drawdown-chart")


def create_profit_distribution_chart(result: BacktestResult) -> str:
    """Create profit distribution histogram showing win/loss patterns.

    Args:
        result: BacktestResult object containing bet records

    Returns:
        HTML string with embedded Plotly chart using CDN
    """
    profits = [bet.profit for bet in result.bets if bet.profit is not None]

    if not profits:
        # Return empty chart if no profit data
        fig = go.Figure()
        fig.add_annotation(
            text="No profit data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(height=400)
        return fig.to_html(include_plotlyjs="cdn", div_id="profit-distribution-chart")

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    fig = go.Figure()

    # Add histogram for all profits
    fig.add_trace(
        go.Histogram(
            x=profits,
            nbinsx=30,
            name="All Bets",
            marker_color="#667eea",
            opacity=0.7,
            hovertemplate="<b>Profit Range:</b> $%{x:,.2f}<br><b>Count:</b> %{y}<extra></extra>",
        )
    )

    # Add vertical lines for mean and median
    mean_profit = sum(profits) / len(profits)
    median_profit = sorted(profits)[len(profits) // 2]

    fig.add_vline(
        x=mean_profit,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: ${mean_profit:.2f}",
        annotation_position="top",
    )

    fig.add_vline(
        x=median_profit,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Median: ${median_profit:.2f}",
        annotation_position="top",
    )

    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)

    fig.update_layout(
        title="Profit Distribution",
        xaxis_title="Profit per Bet ($)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    return fig.to_html(include_plotlyjs="cdn", div_id="profit-distribution-chart")
