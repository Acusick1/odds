"""Main CLI entry point using Typer."""

import typer
from rich.console import Console

from cli.commands import backfill, backtest, fetch, scheduler, status

app = typer.Typer(
    name="odds",
    help="Betting Odds Data Pipeline - NBA odds collection and analysis",
    add_completion=False,
)

# Add command groups
app.add_typer(fetch.app, name="fetch", help="Fetch odds data")
app.add_typer(status.app, name="status", help="System status and monitoring")
app.add_typer(backfill.app, name="backfill", help="Historical data backfill")
app.add_typer(backtest.app, name="backtest", help="Backtest betting strategies")
app.add_typer(scheduler.app, name="scheduler", help="Scheduler management (local testing)")

console = Console()


@app.callback()
def callback():
    """
    Betting Odds Data Pipeline

    A single-user NBA betting odds data collection and analysis system.
    """
    pass


if __name__ == "__main__":
    app()
