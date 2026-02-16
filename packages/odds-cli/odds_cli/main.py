"""Main CLI entry point using Typer."""

import typer
from odds_core.config import get_settings
from odds_core.logging_setup import configure_logging
from rich.console import Console

from odds_cli.commands import (
    backfill,
    backtest,
    copy_from_prod,
    discover,
    fetch,
    polymarket,
    quality,
    scheduler,
    status,
    train,
    validate,
)

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
app.add_typer(train.app, name="train", help="Train ML models from configuration")
app.add_typer(scheduler.app, name="scheduler", help="Scheduler management (local testing)")
app.add_typer(validate.app, name="validate", help="Validate data completeness")
app.add_typer(quality.app, name="quality", help="Data quality coverage analysis")
app.add_typer(copy_from_prod.app, name="copy", help="Copy data from production database")
app.add_typer(discover.app, name="discover", help="Discover upcoming and historical games")
app.add_typer(polymarket.app, name="polymarket", help="Polymarket data operations")

console = Console()


@app.callback()
def callback():
    """
    Betting Odds Data Pipeline

    A single-user NBA betting odds data collection and analysis system.
    """
    # Initialize logging for CLI (human-readable output)
    settings = get_settings()
    configure_logging(settings, json_output=False)


if __name__ == "__main__":
    app()
