"""Historical data backfill commands."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from odds_lambda.backfill_executor import BackfillExecutor, BackfillProgress
from odds_core.config import get_settings
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_core.database import async_session_maker
from odds_lambda.game_selector import GameSelector

app = typer.Typer(help="Historical data backfill operations")
console = Console()


@app.command("plan")
def create_backfill_plan(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    target_games: int = typer.Option(
        166, "--games", "-g", help="Target number of games to backfill"
    ),
    output_file: str = typer.Option(
        "backfill_plan.json", "--output", "-o", help="Output file for backfill plan"
    ),
    sample_interval: int = typer.Option(
        1, "--interval", "-i", help="Days between samples when discovering games"
    ),
):
    """
    Create a backfill plan by discovering games in date range.

    This will query the API to find games and create an execution plan,
    but will NOT fetch historical odds yet. Review the plan before executing.

    Example:
        odds backfill plan --start 2023-10-01 --end 2024-04-30 --games 166
    """
    console.print("\n[bold cyan]Creating Historical Backfill Plan[/bold cyan]\n")

    asyncio.run(
        _create_plan_async(start_date, end_date, target_games, output_file, sample_interval)
    )


async def _create_plan_async(
    start_date_str: str,
    end_date_str: str,
    target_games: int,
    output_file: str,
    sample_interval: int,
):
    """Async implementation of plan creation."""
    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    # Initialize selector
    selector = GameSelector(
        start_date=start_date,
        end_date=end_date,
        target_games=target_games,
        games_per_team=max(1, target_games // 30),  # ~5-6 games per team
    )

    # Generate sample dates
    sample_dates = selector.generate_sample_dates(days_interval=sample_interval)
    console.print(f"Sampling {len(sample_dates)} dates between {start_date_str} and {end_date_str}")

    # Fetch event lists from sample dates
    events_by_date = {}

    async with TheOddsAPIClient() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering games...", total=len(sample_dates))

            for sample_date in sample_dates:
                # Query for events around this date
                date_str = sample_date.isoformat()

                try:
                    response = await client.get_historical_events(
                        sport="basketball_nba", date=date_str
                    )

                    # Response structure: {"data": {"data": [events], "timestamp": ...}, ...}
                    response_data = response.get("data", {})
                    events = (
                        response_data.get("data", []) if isinstance(response_data, dict) else []
                    )

                    if events:
                        events_by_date[date_str] = events
                        progress.console.print(f"  {sample_date.date()}: Found {len(events)} games")

                except Exception as e:
                    progress.console.print(
                        f"  [yellow]Warning: Failed to fetch {sample_date.date()}: {e}[/yellow]"
                    )

                progress.advance(task)

                # Small delay to be nice to API
                await asyncio.sleep(0.5)

    # Generate execution plan
    console.print("\n[cyan]Generating backfill plan...[/cyan]")
    plan = selector.generate_backfill_plan(events_by_date)

    # Display summary
    table = Table(title="Backfill Plan Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Games", str(plan["total_games"]))
    table.add_row("Total Snapshots", str(plan["total_snapshots"]))
    table.add_row("Estimated Quota Usage", f"{plan['estimated_quota_usage']:,}")
    app_settings = get_settings()
    table.add_row("Quota Remaining", f"{app_settings.api.quota - plan['estimated_quota_usage']:,}")
    table.add_row("Date Range", f"{start_date_str} to {end_date_str}")

    console.print("\n")
    console.print(table)

    # Save plan
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2, default=str)

    console.print(f"\n[green]✓ Plan saved to {output_file}[/green]")
    console.print("\n[yellow]Review the plan, then execute with:[/yellow]")
    console.print(f"[bold]  odds backfill execute --plan {output_file}[/bold]\n")


@app.command("execute")
def execute_backfill(
    plan_file: str = typer.Option(..., "--plan", "-p", help="Path to backfill plan JSON file"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate execution without making API calls"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip games that already have historical snapshots",
    ),
):
    """
    Execute a backfill plan to fetch historical odds.

    This will make actual API calls and consume quota. Use --dry-run first
    to validate the plan.

    Example:
        odds backfill execute --plan backfill_plan.json
        odds backfill execute --plan backfill_plan.json --dry-run
    """
    console.print("\n[bold cyan]Executing Historical Backfill[/bold cyan]\n")

    asyncio.run(_execute_backfill_async(plan_file, dry_run, skip_existing))


async def _execute_backfill_async(
    plan_file: str,
    dry_run: bool,
    skip_existing: bool,
):
    """Async implementation of backfill execution."""
    # Load plan
    try:
        with open(plan_file) as f:
            plan = json.load(f)
    except FileNotFoundError as e:
        console.print(f"[red]Error: Plan file not found: {plan_file}[/red]")
        raise typer.Exit(1) from e
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in plan file: {e}[/red]")
        raise typer.Exit(1) from e

    games = plan.get("games", [])
    if not games:
        console.print("[red]Error: No games in plan[/red]")
        raise typer.Exit(1)

    # Display execution summary
    console.print(f"[bold]Games to backfill:[/bold] {len(games)}")
    console.print(f"[bold]Total snapshots:[/bold] {plan['total_snapshots']}")
    console.print(f"[bold]Estimated quota:[/bold] {plan['estimated_quota_usage']:,}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - No API calls will be made[/yellow]")

    # Confirm execution
    if not dry_run:
        console.print("\n[yellow]This will consume API quota. Continue?[/yellow]")
        confirmed = typer.confirm("Proceed with backfill?")
        if not confirmed:
            console.print("[red]Aborted[/red]")
            raise typer.Exit(0)

    # Execute backfill with progress tracking
    async with BackfillExecutor(
        skip_existing=skip_existing,
        dry_run=dry_run,
        rate_limit_seconds=1.0,
    ) as executor:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            game_task = progress.add_task("Processing games...", total=len(games))

            current_game = None

            def on_progress(progress_update: BackfillProgress):
                """Handle progress updates from executor."""
                nonlocal current_game

                # Print game header when starting new game
                game_key = f"{progress_update.away_team}@{progress_update.home_team}"
                if current_game != game_key:
                    current_game = game_key
                    # Find snapshot count for this game
                    game_info = next(
                        (g for g in games if g["event_id"] == progress_update.event_id), None
                    )
                    snapshot_count = len(game_info["snapshots"]) if game_info else "?"
                    progress.console.print(
                        f"\n[cyan]{progress_update.away_team} @ {progress_update.home_team}[/cyan] ({snapshot_count} snapshots)"
                    )

                # Print snapshot result
                snapshot_short = progress_update.snapshot_time[:16]
                if progress_update.status == "success":
                    quota_str = (
                        f"(quota: {progress_update.quota_remaining:,})"
                        if progress_update.quota_remaining
                        else ""
                    )
                    progress.console.print(f"    [green]✓ {snapshot_short}[/green] {quota_str}")
                elif progress_update.status == "exists":
                    progress.console.print(
                        f"    [dim]⊘ Skipped (already exists): {snapshot_short}[/dim]"
                    )
                elif progress_update.status == "skipped":
                    if dry_run:
                        progress.console.print(
                            f"  [dim]Would fetch: {progress_update.snapshot_time}[/dim]"
                        )
                    else:
                        progress.console.print(
                            f"    [yellow]⚠ {progress_update.message}: {snapshot_short}[/yellow]"
                        )
                elif progress_update.status == "failed":
                    message = (
                        progress_update.message[:60]
                        if len(progress_update.message) > 60
                        else progress_update.message
                    )
                    progress.console.print(f"    [red]✗ Failed: {message}[/red]")

            # Track games processed for progress bar
            last_event_id = None

            def on_progress_with_advance(progress_update: BackfillProgress):
                nonlocal last_event_id
                on_progress(progress_update)
                # Advance progress bar when we finish a game (move to new event)
                if last_event_id and last_event_id != progress_update.event_id:
                    progress.advance(game_task)
                last_event_id = progress_update.event_id

            # Execute the backfill
            result = await executor.execute_plan(plan, progress_callback=on_progress_with_advance)

            # Advance for the last game
            progress.advance(game_task)

    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Backfill Complete[/bold cyan]")
    console.print("=" * 60)

    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Successful Games", str(result.successful_games))
    summary_table.add_row("Successful Snapshots", str(result.successful_snapshots))
    summary_table.add_row("Failed Snapshots", str(result.failed_snapshots))
    summary_table.add_row("Skipped Snapshots", str(result.skipped_snapshots))

    if not dry_run:
        summary_table.add_row("Approx. Quota Used", f"{result.total_quota_used:,}")

    console.print("\n")
    console.print(summary_table)
    console.print()


@app.command("status")
def backfill_status():
    """
    Check status of historical data in database.

    Shows how many games have historical snapshots and date coverage.
    """
    console.print("\n[bold cyan]Historical Data Status[/bold cyan]\n")
    asyncio.run(_backfill_status_async())


async def _backfill_status_async():
    """Async implementation of status check."""
    from typing import Any, cast

    from sqlalchemy import func, select
    from sqlalchemy.sql.elements import ColumnElement

    from core.models import OddsSnapshot

    async with async_session_maker() as session:
        # Count events with historical snapshots
        query = select(
            func.count(func.distinct(OddsSnapshot.event_id)),
            func.min(OddsSnapshot.snapshot_time),
            func.max(OddsSnapshot.snapshot_time),
            func.count(cast(ColumnElement[Any], OddsSnapshot.id)),
        )

        result = await session.execute(query)
        row = result.first()

        if row:
            events_count, min_date, max_date, total_snapshots = row

            table = Table(title="Historical Data Coverage")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Events with History", str(events_count))
            table.add_row("Total Snapshots", str(total_snapshots))
            if events_count > 0:
                table.add_row("Avg Snapshots/Game", f"{total_snapshots / events_count:.1f}")
            table.add_row("Earliest Snapshot", str(min_date) if min_date else "N/A")
            table.add_row("Latest Snapshot", str(max_date) if max_date else "N/A")

            console.print(table)
        else:
            console.print("[yellow]No historical data found[/yellow]")

    console.print()
