"""HTML templates for backtest reports."""

from __future__ import annotations


def get_base_template() -> str:
    """Return the base HTML template with Bootstrap styling."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        .report-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .positive {{
            color: #28a745;
        }}
        .negative {{
            color: #dc3545;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        .table-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .strategy-info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .strategy-info dt {{
            font-weight: 600;
            color: #495057;
        }}
        .strategy-info dd {{
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="report-header">
            <h1>{strategy_name}</h1>
            <p class="mb-0">Backtest Period: {start_date} to {end_date}</p>
        </div>

        {content}
    </div>
</body>
</html>
"""


def get_metrics_section_template() -> str:
    """Return template for summary metrics section."""
    return """
<div class="row mb-4">
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Return on Investment</div>
            <div class="metric-value {roi_class}">{roi}%</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{win_rate}%</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{sharpe_ratio}</div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Total Profit</div>
            <div class="metric-value {profit_class}">${profit}</div>
        </div>
    </div>
</div>
"""


def get_chart_container_template() -> str:
    """Return template for chart containers."""
    return """
<div class="chart-container">
    <h3>{title}</h3>
    <div id="{chart_id}"></div>
</div>
"""


def get_table_container_template() -> str:
    """Return template for table containers."""
    return """
<div class="table-container">
    <h3>{title}</h3>
    {table_html}
</div>
"""


def get_strategy_info_template() -> str:
    """Return template for strategy information section."""
    return """
<div class="strategy-info">
    <h2>Strategy Information</h2>
    <dl class="row mb-0">
        <dt class="col-sm-3">Strategy Name</dt>
        <dd class="col-sm-9">{strategy_name}</dd>

        <dt class="col-sm-3">Parameters</dt>
        <dd class="col-sm-9">{strategy_params}</dd>

        <dt class="col-sm-3">Initial Bankroll</dt>
        <dd class="col-sm-9">${initial_bankroll:,.2f}</dd>

        <dt class="col-sm-3">Final Bankroll</dt>
        <dd class="col-sm-9">${final_bankroll:,.2f}</dd>

        <dt class="col-sm-3">Total Bets</dt>
        <dd class="col-sm-9">{total_bets} ({winning_bets} wins, {losing_bets} losses, {push_bets} pushes)</dd>

        <dt class="col-sm-3">Data Quality</dt>
        <dd class="col-sm-9">{events_with_complete_data} of {total_events} events with complete data</dd>

        <dt class="col-sm-3">Execution Time</dt>
        <dd class="col-sm-9">{execution_time_seconds:.2f} seconds</dd>
    </dl>
</div>
"""
