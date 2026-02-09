"""Polymarket data operations commands."""

import asyncio

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Polymarket prediction market operations")
console = Console()


@app.command("backfill")
def backfill(
    include_spreads: bool = typer.Option(
        False,
        "--include-spreads",
        help="Include spread market price histories",
    ),
    include_totals: bool = typer.Option(
        False,
        "--include-totals",
        help="Include total (over/under) market price histories",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate execution without storing data",
    ),
):
    """
    Backfill historical Polymarket price data from closed NBA markets.

    This command captures historical price data before it expires. The CLOB
    /prices-history endpoint has a ~30-day rolling retention window.

    Behavior:
    - Fetches all closed NBA events from Gamma API
    - Checks which moneyline markets need backfill (skips if >10 snapshots exist)
    - Fetches 5-minute resolution price history from CLOB API
    - Bulk inserts into polymarket_price_snapshots table
    - Optionally backfills spread/total markets if flags enabled

    Examples:
        # Backfill moneyline markets only (default)
        odds polymarket backfill

        # Backfill all market types
        odds polymarket backfill --include-spreads --include-totals

        # Test without storing data
        odds polymarket backfill --dry-run

    Notes:
    - Safe to run repeatedly - skip logic prevents re-fetching
    - Gracefully handles markets with no available data (old games)
    - Individual market failures don't crash the job
    - Logged to polymarket_fetch_logs table
    """
    console.print("\n[bold cyan]Polymarket Historical Price Backfill[/bold cyan]\n")

    if dry_run:
        console.print("[yellow]DRY RUN - No data will be stored[/yellow]\n")

    asyncio.run(_backfill_async(include_spreads, include_totals, dry_run))


async def _backfill_async(
    include_spreads: bool,
    include_totals: bool,
    dry_run: bool,
):
    """Async implementation of backfill command."""
    # Display configuration
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Include spreads", "Yes" if include_spreads else "No")
    config_table.add_row("Include totals", "Yes" if include_totals else "No")
    config_table.add_row("Dry run", "Yes" if dry_run else "No")

    console.print(config_table)
    console.print()

    # Import job module
    from odds_lambda.jobs import backfill_polymarket

    # Execute with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching and backfilling price histories...", total=None)

        try:
            await backfill_polymarket.main(
                include_spreads=include_spreads,
                include_totals=include_totals,
                dry_run=dry_run,
            )
            progress.update(task, completed=True)

        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Backfill failed: {e}[/red]")
            raise typer.Exit(1) from e

    console.print("\n[green]Backfill completed successfully![/green]")
    console.print(
        "\n[dim]Check logs for detailed statistics (events processed, markets backfilled, etc.)[/dim]\n"
    )
