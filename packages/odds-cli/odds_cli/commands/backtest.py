"""CLI commands for backtesting betting strategies."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from odds_analytics.backtesting import BacktestConfig, BacktestEngine, BacktestResult
from odds_analytics.lstm_strategy import LSTMStrategy
from odds_analytics.strategies import ArbitrageStrategy, BasicEVStrategy, FlatBettingStrategy
from odds_core.database import get_session
from odds_lambda.storage.readers import OddsReader
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

# Strategy registry
STRATEGIES = {
    "flat": FlatBettingStrategy,
    "basic_ev": BasicEVStrategy,
    "arbitrage": ArbitrageStrategy,
    "lstm": LSTMStrategy,
}


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[red]Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.[/red]")
        raise typer.Exit(1) from e


@app.command("run")
def run_backtest(
    strategy: str = typer.Option(
        ..., "--strategy", "-s", help="Strategy name (flat, basic_ev, arbitrage, lstm)"
    ),
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)"),
    bankroll: float = typer.Option(10000.0, "--bankroll", "-b", help="Initial bankroll"),
    output_json: str | None = typer.Option(
        None, "--output-json", "-j", help="Save results to JSON file"
    ),
    output_csv: str | None = typer.Option(None, "--output-csv", "-c", help="Save bets to CSV file"),
    bet_sizing: str = typer.Option(
        "fractional_kelly",
        "--bet-sizing",
        help="Bet sizing method (fractional_kelly, flat, percentage)",
    ),
    kelly_fraction: float = typer.Option(
        0.25, "--kelly-fraction", help="Kelly fraction (default: 0.25)"
    ),
):
    """
    Run a backtest for a strategy over a date range.

    Example:
        odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
    """
    # Parse dates
    start_date = parse_date(start)
    end_date = parse_date(end)

    # Get strategy
    if strategy not in STRATEGIES:
        console.print(f"[red]Error: Unknown strategy '{strategy}'[/red]")
        console.print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        raise typer.Exit(1)

    # Run backtest
    asyncio.run(
        _run_backtest_async(
            strategy_name=strategy,
            start_date=start_date,
            end_date=end_date,
            bankroll=bankroll,
            output_json=output_json,
            output_csv=output_csv,
            bet_sizing=bet_sizing,
            kelly_fraction=kelly_fraction,
        )
    )


async def _run_backtest_async(
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    bankroll: float,
    output_json: str | None,
    output_csv: str | None,
    bet_sizing: str,
    kelly_fraction: float,
):
    """Run backtest asynchronously."""
    # Create strategy instance
    strategy_class = STRATEGIES[strategy_name]
    strategy = strategy_class()

    # Create bet sizing config
    from odds_analytics.backtesting import BetSizingConfig

    sizing_config = BetSizingConfig(
        method=bet_sizing,
        kelly_fraction=kelly_fraction,
    )

    # Create config
    config = BacktestConfig(
        initial_bankroll=bankroll,
        start_date=start_date,
        end_date=end_date,
        sizing=sizing_config,
    )

    # Get database session
    async for session in get_session():
        # Create engine and run
        reader = OddsReader(session)
        engine = BacktestEngine(strategy, config, reader)

        console.print(f"\n[bold]Running backtest: {strategy.get_name()}[/bold]")
        console.print(f"Period: {start_date.date()} to {end_date.date()}")
        console.print(f"Initial bankroll: ${bankroll:,.2f}\n")

        result = await engine.run()

    # Display summary
    console.print(result.to_summary_text())

    # Export if requested
    if output_json:
        result.to_json(output_json)
        console.print(f"\n[green]✓[/green] Results saved to {output_json}")

    if output_csv:
        result.to_csv(output_csv)
        console.print(f"[green]✓[/green] Bets exported to {output_csv}")

    # Show export options
    if not output_json and not output_csv:
        console.print("\n[dim]Export options:[/dim]")
        console.print(
            "[dim]  --output-json results.json    (full results with reconstruction)[/dim]"
        )
        console.print("[dim]  --output-csv bets.csv         (bet-by-bet details)[/dim]")


@app.command("show")
def show_results(
    filepath: Annotated[str, typer.Argument(help="Path to JSON results file")],
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed breakdown"),
):
    """
    Display results from a saved backtest JSON file.

    Example:
        odds backtest show results.json
        odds backtest show results.json --verbose
    """
    if not Path(filepath).exists():
        console.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(1)

    try:
        result = BacktestResult.from_json(filepath)
    except Exception as e:
        console.print(f"[red]Error loading results: {e}[/red]")
        raise typer.Exit(1) from e

    # Display summary
    console.print(result.to_summary_text())

    if verbose:
        # Show market breakdown
        if result.market_breakdown:
            console.print("\n[bold]Market Breakdown:[/bold]")
            table = Table()
            table.add_column("Market")
            table.add_column("Bets", justify="right")
            table.add_column("Profit", justify="right")
            table.add_column("ROI", justify="right")
            table.add_column("Win Rate", justify="right")

            for market, stats in result.market_breakdown.items():
                profit_color = "green" if stats.profit > 0 else "red"
                table.add_row(
                    market,
                    str(stats.bets),
                    f"[{profit_color}]${stats.profit:,.2f}[/{profit_color}]",
                    f"{stats.roi:.2f}%",
                    f"{stats.win_rate:.1f}%",
                )
            console.print(table)

        # Show bookmaker breakdown
        if result.bookmaker_breakdown:
            console.print("\n[bold]Bookmaker Breakdown:[/bold]")
            table = Table()
            table.add_column("Bookmaker")
            table.add_column("Bets", justify="right")
            table.add_column("Profit", justify="right")
            table.add_column("ROI", justify="right")

            for bookmaker, stats in result.bookmaker_breakdown.items():
                profit_color = "green" if stats.profit > 0 else "red"
                table.add_row(
                    bookmaker,
                    str(stats.bets),
                    f"[{profit_color}]${stats.profit:,.2f}[/{profit_color}]",
                    f"{stats.roi:.2f}%",
                )
            console.print(table)

        # Show monthly performance
        if result.monthly_performance:
            console.print("\n[bold]Monthly Performance:[/bold]")
            table = Table()
            table.add_column("Month")
            table.add_column("Bets", justify="right")
            table.add_column("Profit", justify="right")
            table.add_column("ROI", justify="right")
            table.add_column("Bankroll", justify="right")

            for month_stats in result.monthly_performance:
                profit_color = "green" if month_stats.profit > 0 else "red"
                table.add_row(
                    month_stats.month,
                    str(month_stats.bets),
                    f"[{profit_color}]${month_stats.profit:,.2f}[/{profit_color}]",
                    f"{month_stats.roi:.2f}%",
                    f"${month_stats.end_bankroll:,.2f}",
                )
            console.print(table)


@app.command("compare")
def compare_results(
    files: Annotated[list[str], typer.Argument(help="Paths to JSON result files to compare")],
):
    """
    Compare multiple backtest results side-by-side.

    Example:
        odds backtest compare strategy1.json strategy2.json strategy3.json
    """
    # Load all results
    results = []
    for filepath in files:
        if not Path(filepath).exists():
            console.print(f"[red]Error: File not found: {filepath}[/red]")
            continue

        try:
            result = BacktestResult.from_json(filepath)
            results.append((Path(filepath).stem, result))
        except Exception as e:
            console.print(f"[red]Error loading {filepath}: {e}[/red]")
            continue

    if len(results) < 2:
        console.print("[red]Error: Need at least 2 valid result files to compare[/red]")
        raise typer.Exit(1)

    # Create comparison table
    console.print("\n[bold]Strategy Comparison[/bold]\n")

    table = Table()
    table.add_column("Metric")
    for name, _ in results:
        table.add_column(name, justify="right")

    # Add rows for each metric
    metrics = [
        ("Strategy", lambda r: r.strategy_name),
        ("Period", lambda r: f"{r.start_date.date()} to {r.end_date.date()}"),
        ("Initial Bankroll", lambda r: f"${r.initial_bankroll:,.2f}"),
        ("Final Bankroll", lambda r: f"${r.final_bankroll:,.2f}"),
        ("Total Profit", lambda r: f"${r.total_profit:,.2f}"),
        ("ROI", lambda r: f"{r.roi:.2f}%"),
        ("Total Bets", lambda r: str(r.total_bets)),
        ("Win Rate", lambda r: f"{r.win_rate:.1f}%"),
        ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Sortino Ratio", lambda r: f"{r.sortino_ratio:.2f}"),
        (
            "Max Drawdown",
            lambda r: f"${abs(r.max_drawdown):,.2f} ({r.max_drawdown_percentage:.1f}%)",
        ),
        ("Profit Factor", lambda r: f"{r.profit_factor:.2f}"),
        ("Avg Win", lambda r: f"${r.average_win:.2f}"),
        ("Avg Loss", lambda r: f"${r.average_loss:.2f}"),
        ("Largest Win", lambda r: f"${r.largest_win:.2f}"),
        ("Largest Loss", lambda r: f"${r.largest_loss:.2f}"),
    ]

    for metric_name, metric_fn in metrics:
        row = [metric_name]
        for _, result in results:
            row.append(metric_fn(result))
        table.add_row(*row)

    console.print(table)

    # Highlight best strategy
    best_roi = max(results, key=lambda x: x[1].roi)
    best_sharpe = max(results, key=lambda x: x[1].sharpe_ratio)

    console.print(f"\n[bold]Best ROI:[/bold] {best_roi[0]} ({best_roi[1].roi:.2f}%)")
    console.print(
        f"[bold]Best Sharpe Ratio:[/bold] {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})"
    )


@app.command("export")
def export_csv(
    json_file: Annotated[str, typer.Argument(help="Path to JSON results file")],
    csv_file: Annotated[str, typer.Argument(help="Output CSV file path")],
):
    """
    Export bets from a JSON results file to CSV format.

    Example:
        odds backtest export results.json bets.csv
    """
    if not Path(json_file).exists():
        console.print(f"[red]Error: File not found: {json_file}[/red]")
        raise typer.Exit(1)

    try:
        result = BacktestResult.from_json(json_file)
        message = result.to_csv(csv_file)
        console.print(f"[green]✓[/green] {message}")
    except Exception as e:
        console.print(f"[red]Error exporting to CSV: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("report")
def generate_html_report(
    json_file: Annotated[str, typer.Argument(help="Path to JSON results file")],
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML file path (default: results_report.html)",
    ),
):
    """
    Generate an interactive HTML report from backtest results.

    Example:
        odds backtest report results.json
        odds backtest report results.json --output custom_report.html
    """
    if not Path(json_file).exists():
        console.print(f"[red]Error: File not found: {json_file}[/red]")
        raise typer.Exit(1)

    # Load backtest results
    try:
        result = BacktestResult.from_json(json_file)
    except Exception as e:
        console.print(f"[red]Error loading results: {e}[/red]")
        raise typer.Exit(1) from e

    # Determine output path
    if output is None:
        output = str(Path(json_file).with_suffix("")) + "_report.html"

    # Generate HTML report
    try:
        from odds_analytics.reporting import HTMLReportGenerator

        generator = HTMLReportGenerator(result)
        generator.generate(output)

        console.print(f"[green]✓[/green] HTML report generated: {output}")
        console.print(f"\n[dim]Open in browser to view interactive visualizations[/dim]")
    except Exception as e:
        console.print(f"[red]Error generating HTML report: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("list-strategies")
def list_strategies():
    """
    List all available betting strategies.

    Example:
        odds backtest list-strategies
    """
    console.print("\n[bold]Available Strategies:[/bold]\n")

    for name, strategy_class in STRATEGIES.items():
        # Create instance to get default params
        instance = strategy_class()
        console.print(f"[bold]{name}[/bold]")
        console.print(f"  {strategy_class.__doc__.strip().split(chr(10))[0]}")
        console.print(f"  Parameters: {instance.get_params()}")
        console.print()


if __name__ == "__main__":
    app()
