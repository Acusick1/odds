"""HTML report generator for backtest results."""

from __future__ import annotations

import json
from pathlib import Path

from odds_analytics.backtesting.models import BacktestResult
from odds_analytics.reporting import charts, tables, templates


class HTMLReportGenerator:
    """Generate interactive HTML reports from BacktestResult objects.

    This class orchestrates the creation of a comprehensive HTML report that includes:
    - Summary metrics cards
    - Interactive Plotly charts (equity curve, monthly performance, etc.)
    - Formatted tables using Great Tables
    - Strategy information and metadata

    The output is a single self-contained HTML file that can be opened in any browser.
    """

    def __init__(self, result: BacktestResult) -> None:
        """Initialize the HTML report generator.

        Args:
            result: BacktestResult object containing all backtest data and metrics
        """
        self.result = result

    def generate(self, output_path: str) -> None:
        """Generate the HTML report and save to file.

        Args:
            output_path: Path where the HTML report should be saved

        Raises:
            ValueError: If result is invalid or missing required data
            IOError: If unable to write to output path
        """
        # Validate result has required data
        if not self.result.bets:
            raise ValueError("BacktestResult must contain at least one bet")

        if not self.result.equity_curve:
            raise ValueError("BacktestResult must contain equity curve data")

        # Generate all components
        html_content = self._build_html()

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as file_object:
            file_object.write(html_content)

    def _build_html(self) -> str:
        """Build the complete HTML content by assembling all components.

        Returns:
            Complete HTML document as string
        """
        # Build individual sections
        metrics_section = self._build_metrics_section()
        strategy_info = self._build_strategy_info()
        charts_section = self._build_charts_section()
        tables_section = self._build_tables_section()

        # Combine all sections
        content = f"""
        {strategy_info}
        {metrics_section}

        <h2>Performance Visualizations</h2>
        {charts_section}

        <h2>Detailed Breakdowns</h2>
        {tables_section}
        """

        # Apply base template
        base_template = templates.get_base_template()
        html = base_template.format(
            strategy_name=self.result.strategy_name,
            start_date=self.result.start_date.strftime("%Y-%m-%d"),
            end_date=self.result.end_date.strftime("%Y-%m-%d"),
            content=content,
        )

        return html

    def _build_metrics_section(self) -> str:
        """Build the summary metrics cards section.

        Returns:
            HTML string with metrics cards
        """
        roi_class = "positive" if self.result.roi > 0 else "negative"
        profit_class = "positive" if self.result.total_profit > 0 else "negative"

        metrics_template = templates.get_metrics_section_template()
        return metrics_template.format(
            roi=f"{self.result.roi:.2f}",
            roi_class=roi_class,
            win_rate=f"{self.result.win_rate:.2f}",
            sharpe_ratio=f"{self.result.sharpe_ratio:.2f}",
            profit=f"{self.result.total_profit:,.2f}",
            profit_class=profit_class,
        )

    def _build_strategy_info(self) -> str:
        """Build the strategy information section.

        Returns:
            HTML string with strategy metadata
        """
        # Format strategy params as readable text
        params_str = json.dumps(self.result.strategy_params, indent=2)

        strategy_template = templates.get_strategy_info_template()
        return strategy_template.format(
            strategy_name=self.result.strategy_name,
            strategy_params=params_str,
            initial_bankroll=self.result.initial_bankroll,
            final_bankroll=self.result.final_bankroll,
            total_bets=self.result.total_bets,
            winning_bets=self.result.winning_bets,
            losing_bets=self.result.losing_bets,
            push_bets=self.result.push_bets,
            events_with_complete_data=self.result.events_with_complete_data,
            total_events=self.result.total_events,
            execution_time_seconds=self.result.execution_time_seconds,
        )

    def _build_charts_section(self) -> str:
        """Build all charts section with Plotly visualizations.

        Returns:
            HTML string with all chart containers and embedded Plotly charts
        """
        chart_htmls = []

        # Generate all 6 charts
        equity_chart = charts.create_equity_curve_chart(self.result)
        monthly_chart = charts.create_monthly_performance_chart(self.result)
        market_chart = charts.create_market_breakdown_chart(self.result)
        bookmaker_chart = charts.create_bookmaker_breakdown_chart(self.result)
        drawdown_chart = charts.create_drawdown_chart(self.result)
        profit_dist_chart = charts.create_profit_distribution_chart(self.result)

        # Wrap each chart in container
        chart_template = templates.get_chart_container_template()

        chart_htmls.append(
            chart_template.format(
                title="Equity Curve", chart_id="equity-curve-container"
            ).replace('<div id="equity-curve-container"></div>', equity_chart)
        )

        chart_htmls.append(
            chart_template.format(
                title="Drawdown Analysis", chart_id="drawdown-container"
            ).replace('<div id="drawdown-container"></div>', drawdown_chart)
        )

        chart_htmls.append(
            chart_template.format(
                title="Monthly Performance", chart_id="monthly-performance-container"
            ).replace(
                '<div id="monthly-performance-container"></div>', monthly_chart
            )
        )

        chart_htmls.append(
            chart_template.format(
                title="Market Type Breakdown", chart_id="market-breakdown-container"
            ).replace('<div id="market-breakdown-container"></div>', market_chart)
        )

        chart_htmls.append(
            chart_template.format(
                title="Bookmaker Comparison",
                chart_id="bookmaker-breakdown-container",
            ).replace(
                '<div id="bookmaker-breakdown-container"></div>', bookmaker_chart
            )
        )

        chart_htmls.append(
            chart_template.format(
                title="Profit Distribution",
                chart_id="profit-distribution-container",
            ).replace(
                '<div id="profit-distribution-container"></div>', profit_dist_chart
            )
        )

        return "\n".join(chart_htmls)

    def _build_tables_section(self) -> str:
        """Build all tables section with Great Tables formatting.

        Returns:
            HTML string with all formatted tables
        """
        table_htmls = []

        # Generate all tables
        bet_summary = tables.create_bet_summary_table(self.result)
        risk_metrics = tables.create_risk_metrics_table(self.result)
        market_breakdown = tables.create_market_breakdown_table(self.result)
        bookmaker_breakdown = tables.create_bookmaker_breakdown_table(self.result)
        monthly_performance = tables.create_monthly_performance_table(self.result)

        # Wrap each table in container
        table_template = templates.get_table_container_template()

        table_htmls.append(
            table_template.format(title="Bet Statistics", table_html=bet_summary)
        )

        table_htmls.append(
            table_template.format(
                title="Risk & Performance Metrics", table_html=risk_metrics
            )
        )

        table_htmls.append(
            table_template.format(
                title="Market Type Performance", table_html=market_breakdown
            )
        )

        table_htmls.append(
            table_template.format(
                title="Bookmaker Performance", table_html=bookmaker_breakdown
            )
        )

        table_htmls.append(
            table_template.format(
                title="Monthly Performance", table_html=monthly_performance
            )
        )

        return "\n".join(table_htmls)
