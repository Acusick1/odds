"""CLI commands for copying data from production database."""

import asyncio
from datetime import datetime

import typer
from odds_lambda.copy_from_production import copy_from_prod
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command("copy-from-prod")
def copy_from_production(
    start: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    sport: str = typer.Option("basketball_nba", "--sport", "-s", help="Sport to copy"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Perform dry run without writing data"),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--overwrite", help="Skip existing events"
    ),
    prod_url: str | None = typer.Option(None, "--prod-url", help="Production database URL"),
    local_url: str | None = typer.Option(None, "--local-url", help="Local database URL"),
):
    """
    Copy completed games from production database to local database.

    This performs a selective merge, copying only completed games with results
    while preserving existing local data.

    Examples:
        # Copy October-November 2025 data (dry run first)
        odds copy-from-prod 2025-10-23 2025-11-07 --dry-run

        # Actually copy the data
        odds copy-from-prod 2025-10-23 2025-11-07

        # Copy with overwrite
        odds copy-from-prod 2025-10-23 2025-11-07 --overwrite
    """
    # Validate dates
    try:
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError as e:
        console.print("[red]Error: Dates must be in YYYY-MM-DD format[/red]")
        raise typer.Exit(code=1) from e

    try:
        asyncio.run(
            copy_from_prod(
                start_date=start,
                end_date=end,
                prod_url=prod_url,
                local_url=local_url,
                sport=sport,
                dry_run=dry_run,
                skip_existing=skip_existing,
            )
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Copy failed: {str(e)}[/bold red]")
        raise typer.Exit(code=1) from e
