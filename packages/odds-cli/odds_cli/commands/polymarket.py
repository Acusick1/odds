"""Polymarket data operations commands."""

import asyncio
from datetime import UTC, datetime
from typing import cast

import typer
from odds_core.sports import SportKey
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Polymarket prediction market operations")
console = Console()


@app.command("backfill-odds")
def backfill_odds(
    sport: str = typer.Option("soccer_epl", "--sport", help="Sport key with three-way markets"),
    since: str = typer.Option("2025-08-01", "--since", help="Earliest kick-off (UTC date)"),
    limit: int | None = typer.Option(None, "--limit", help="Backfill at most this many events"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch and resample without writing"),
):
    """Store Polymarket price history as `polymarket` odds snapshots for finished events.

    Attaches each Polymarket match to its existing event in DATABASE_URL, recording
    the Polymarket event id, and skips events that already have history snapshots, so
    it is safe to rerun.
    """
    from odds_core.sports import SUPPORTED_SPORTS
    from odds_lambda.polymarket_odds_backfill import backfill_sport

    if sport not in SUPPORTED_SPORTS:
        console.print(f"[red]Unknown sport {sport}; use one of {', '.join(SUPPORTED_SPORTS)}[/red]")
        raise typer.Exit(1)
    report = asyncio.run(
        backfill_sport(
            cast(SportKey, sport),
            since=datetime.fromisoformat(since).replace(tzinfo=UTC),
            limit=limit,
            dry_run=dry_run,
        )
    )
    table = Table(title="Polymarket odds backfill" + (" (dry run)" if dry_run else ""))
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for name, value in (
        ("Polymarket matches", report.matches_considered),
        ("Attached to an event", report.matches_attached),
        ("Skipped (event not final, or before --since)", report.matches_out_of_scope),
        ("Skipped (already backfilled)", report.events_skipped_existing),
        ("Events written", report.events_written),
        ("Events with no usable points", report.events_without_points),
        ("Snapshots written", report.snapshots_written),
        ("Grid points dropped, missing prices", report.points_missing),
        ("Grid points dropped, prices off sum", report.points_off_sum),
        ("Events without a Polymarket match", len(report.unmatched_event_ids)),
        ("Polymarket matches without an event", len(report.unmatched_pm_tickers)),
        ("Matches that failed", len(report.errors)),
    ):
        table.add_row(name, str(value))
    console.print(table)


@app.command("discover")
def discover():
    """List active NBA events currently on Polymarket."""
    asyncio.run(_discover_async())


async def _discover_async():
    """Async implementation of discover command."""
    from odds_lambda.polymarket_fetcher import PolymarketClient

    console.print("\n[bold cyan]Active Polymarket NBA Events[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching events from Gamma API...", total=None)
        try:
            async with PolymarketClient() as client:
                events = await client.get_events("basketball_nba", active=True, closed=False)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Failed to fetch events: {e}[/red]")
            raise typer.Exit(1) from e

    if not events:
        console.print("[yellow]No active NBA events found on Polymarket[/yellow]\n")
        return

    table = Table()
    table.add_column("Ticker", style="cyan")
    table.add_column("Title")
    table.add_column("Start Date")
    table.add_column("Markets", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Liquidity", justify="right")

    for event in events:
        ticker = event.get("ticker", "")
        title = event.get("title", "")
        start_date = event.get("startDate", "")[:10] if event.get("startDate") else ""
        markets = event.get("markets", [])
        volume = event.get("volume")
        liquidity = event.get("liquidity")

        volume_str = f"${float(volume):,.0f}" if volume is not None else "-"
        liquidity_str = f"${float(liquidity):,.0f}" if liquidity is not None else "-"

        table.add_row(
            ticker,
            title,
            start_date,
            str(len(markets)),
            volume_str,
            liquidity_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(events)} active event(s)[/dim]\n")
