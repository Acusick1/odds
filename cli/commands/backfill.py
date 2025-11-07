"""Historical data backfill commands."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from core.backfill_executor import BackfillExecutor, BackfillProgress
from core.config import get_settings
from core.data_fetcher import TheOddsAPIClient
from core.database import async_session_maker
from core.game_selector import GameSelector
from core.gap_analyzer import GapAnalyzer

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
    max_calls: int | None = typer.Option(
        None,
        "--max-calls",
        "-n",
        help="Maximum API calls to make (stops execution when reached)",
    ),
):
    """
    Execute a backfill plan to fetch historical odds.

    This will make actual API calls and consume quota. Use --dry-run first
    to validate the plan.

    Example:
        odds backfill execute --plan backfill_plan.json
        odds backfill execute --plan backfill_plan.json --dry-run
        odds backfill execute --plan backfill_plan.json --max-calls 50
    """
    console.print("\n[bold cyan]Executing Historical Backfill[/bold cyan]\n")

    asyncio.run(_execute_backfill_async(plan_file, dry_run, skip_existing, max_calls))


async def _execute_backfill_async(
    plan_file: str,
    dry_run: bool,
    skip_existing: bool,
    max_calls: int | None,
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

    if max_calls is not None:
        console.print(f"[bold yellow]API call limit:[/bold yellow] {max_calls}")
        estimated_snapshots = min(max_calls, plan['total_snapshots'])
        console.print(f"[dim]Will process approximately {estimated_snapshots} snapshots[/dim]")

    if dry_run:
        console.print("\n[yellow]DRY RUN - No API calls will be made[/yellow]")

        # Enhanced dry-run reporting with detailed breakdown
        console.print("\n[bold cyan]Detailed Breakdown[/bold cyan]")

        # Analyze snapshots by game
        from collections import defaultdict
        snapshots_by_count = defaultdict(int)
        for game in games:
            snapshot_count = len(game.get("snapshots", []))
            snapshots_by_count[snapshot_count] += 1

        breakdown_table = Table(title="Snapshot Distribution")
        breakdown_table.add_column("Snapshots per Game", style="cyan")
        breakdown_table.add_column("Number of Games", style="yellow")
        breakdown_table.add_column("Total Snapshots", style="green")

        for count in sorted(snapshots_by_count.keys()):
            num_games = snapshots_by_count[count]
            total = count * num_games
            breakdown_table.add_row(str(count), str(num_games), str(total))

        console.print(breakdown_table)

        # Date range analysis
        if games:
            from datetime import datetime as dt
            game_dates = []
            for game in games:
                try:
                    commence_time_str = game.get("commence_time", "")
                    if commence_time_str:
                        game_date = dt.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                        game_dates.append(game_date)
                except (ValueError, AttributeError):
                    pass

            if game_dates:
                min_date = min(game_dates)
                max_date = max(game_dates)
                console.print(f"\n[bold]Date Coverage:[/bold] {min_date.date()} to {max_date.date()}")

        # Show plan metadata if available
        if "generated_from" in plan:
            console.print(f"[dim]Plan generated from: {plan['generated_from']}[/dim]")

        console.print()

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
            result = await executor.execute_plan(
                plan, progress_callback=on_progress_with_advance, max_calls=max_calls
            )

            # Advance for the last game
            progress.advance(game_task)

    # Final summary
    console.print("\n" + "=" * 60)
    if result.stopped_at_limit:
        console.print("[bold yellow]Backfill Stopped at API Limit[/bold yellow]")
    else:
        console.print("[bold cyan]Backfill Complete[/bold cyan]")
    console.print("=" * 60)

    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Successful Games", str(result.successful_games))
    summary_table.add_row("Successful Snapshots", str(result.successful_snapshots))
    summary_table.add_row("Failed Snapshots", str(result.failed_snapshots))
    summary_table.add_row("Skipped Snapshots", str(result.skipped_snapshots))

    if result.stopped_at_limit:
        summary_table.add_row(
            "Remaining Snapshots", str(result.remaining_snapshots), style="yellow"
        )

    if not dry_run:
        summary_table.add_row("Approx. Quota Used", f"{result.total_quota_used:,}")

    console.print("\n")
    console.print(summary_table)

    if result.stopped_at_limit:
        console.print(
            "\n[yellow]Note: Execution stopped due to API call limit. "
            "Run again to continue processing remaining snapshots.[/yellow]"
        )

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


@app.command("analyze")
def analyze_gaps(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    mode: str = typer.Option(
        "all",
        "--mode",
        "-m",
        help="Analysis mode: events, snapshots, tiers, or all",
    ),
    output_plan: str | None = typer.Option(
        None,
        "--output-plan",
        "-o",
        help="Generate backfill plan JSON to fill gaps",
    ),
    sport: str = typer.Option(
        "basketball_nba",
        "--sport",
        help="Sport to analyze",
    ),
):
    """
    Analyze historical data for gaps and missing snapshots.

    This command identifies:
    - Missing events: Games with no historical snapshots
    - Incomplete snapshots: Games with fewer snapshots than expected
    - Missing tiers: Games missing specific fetch tiers (opening, closing, etc.)

    Example:
        odds backfill analyze --start 2024-11-01 --end 2024-11-07 --mode all
        odds backfill analyze --start 2024-11-01 --end 2024-11-07 --mode tiers
        odds backfill analyze --start 2024-11-01 --end 2024-11-07 --output-plan gaps.json
    """
    console.print("\n[bold cyan]Analyzing Historical Data Gaps[/bold cyan]\n")

    asyncio.run(_analyze_gaps_async(start_date, end_date, mode, output_plan, sport))


async def _analyze_gaps_async(
    start_date_str: str,
    end_date_str: str,
    mode: str,
    output_plan: str | None,
    sport: str,
):
    """Async implementation of gap analysis."""
    # Parse dates
    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    # Validate mode
    valid_modes = ["events", "snapshots", "tiers", "all"]
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}. Must be one of: {', '.join(valid_modes)}[/red]")
        raise typer.Exit(1)

    # Run gap analysis
    async with async_session_maker() as session:
        analyzer = GapAnalyzer(session)

        console.print(f"[cyan]Analyzing {sport} from {start_date_str} to {end_date_str}...[/cyan]\n")

        gap_report = await analyzer.analyze_gaps(
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            sport_key=sport,
        )

        # Display summary statistics
        summary_table = Table(title="Gap Analysis Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row("Analysis Mode", mode.upper())
        summary_table.add_row("Date Range", f"{start_date_str} to {end_date_str}")
        summary_table.add_row("Total Events Checked", str(gap_report.total_events_checked))
        summary_table.add_row("Total Gaps Found", str(gap_report.total_gaps_found))

        if mode in ("events", "all"):
            summary_table.add_row("Missing Events", str(len(gap_report.missing_events)))

        if mode in ("snapshots", "all"):
            summary_table.add_row("Incomplete Events", str(len(gap_report.incomplete_events)))

        if mode in ("tiers", "all"):
            summary_table.add_row(
                "Events Missing Tiers", str(len(gap_report.events_missing_tiers))
            )

        summary_table.add_row(
            "Estimated API Calls", f"{gap_report.estimated_api_calls:,}", style="bold green"
        )

        console.print(summary_table)
        console.print()

        # Display detailed results if gaps found
        if not gap_report.has_gaps():
            console.print("[green]✓ No gaps found! Historical data is complete.[/green]\n")
            return

        # Missing events details
        if gap_report.missing_events and mode in ("events", "all"):
            console.print("\n[bold yellow]Missing Events (No Snapshots)[/bold yellow]")
            events_table = Table()
            events_table.add_column("Event ID", style="dim")
            events_table.add_column("Date", style="cyan")
            events_table.add_column("Matchup", style="white")
            events_table.add_column("Missing Snapshots", style="red")

            for event in gap_report.missing_events[:20]:  # Limit display to 20
                matchup = f"{event.away_team} @ {event.home_team}"
                events_table.add_row(
                    event.id[:8],
                    event.commence_time.strftime("%Y-%m-%d %H:%M"),
                    matchup,
                    "5",
                )

            console.print(events_table)

            if len(gap_report.missing_events) > 20:
                console.print(
                    f"[dim]... and {len(gap_report.missing_events) - 20} more events[/dim]"
                )
            console.print()

        # Incomplete events details
        if gap_report.incomplete_events and mode in ("snapshots", "all"):
            console.print("\n[bold yellow]Incomplete Events (Missing Snapshots)[/bold yellow]")
            incomplete_table = Table()
            incomplete_table.add_column("Event ID", style="dim")
            incomplete_table.add_column("Matchup", style="white")
            incomplete_table.add_column("Current", style="yellow")
            incomplete_table.add_column("Expected", style="green")
            incomplete_table.add_column("Missing", style="red")

            for gap in gap_report.incomplete_events[:20]:  # Limit display
                matchup = f"{gap.event.away_team} @ {gap.event.home_team}"
                incomplete_table.add_row(
                    gap.event.id[:8],
                    matchup,
                    str(gap.current_snapshot_count),
                    str(gap.expected_snapshot_count),
                    str(len(gap.missing_snapshot_times)),
                )

            console.print(incomplete_table)

            if len(gap_report.incomplete_events) > 20:
                console.print(
                    f"[dim]... and {len(gap_report.incomplete_events) - 20} more events[/dim]"
                )
            console.print()

        # Events missing tiers
        if gap_report.events_missing_tiers and mode in ("tiers", "all"):
            console.print("\n[bold yellow]Events Missing Specific Tiers[/bold yellow]")
            tiers_table = Table()
            tiers_table.add_column("Event ID", style="dim")
            tiers_table.add_column("Matchup", style="white")
            tiers_table.add_column("Missing Tiers", style="red")
            tiers_table.add_column("Tier Coverage", style="cyan")

            for gap in gap_report.events_missing_tiers[:20]:  # Limit display
                matchup = f"{gap.event.away_team} @ {gap.event.home_team}"
                missing_tier_names = ", ".join(t.value for t in gap.missing_tiers)

                # Format tier coverage
                coverage_str = ", ".join(
                    f"{tier}:{count}" for tier, count in gap.tier_coverage.items()
                )

                tiers_table.add_row(
                    gap.event.id[:8],
                    matchup,
                    missing_tier_names,
                    coverage_str or "none",
                )

            console.print(tiers_table)

            if len(gap_report.events_missing_tiers) > 20:
                console.print(
                    f"[dim]... and {len(gap_report.events_missing_tiers) - 20} more events[/dim]"
                )
            console.print()

        # Generate backfill plan if requested
        if output_plan:
            console.print(f"\n[cyan]Generating backfill plan to fill gaps...[/cyan]")

            plan = await analyzer.generate_backfill_plan_for_gaps(gap_report)

            # Save plan to file
            output_path = Path(output_plan)
            with open(output_path, "w") as f:
                json.dump(plan, f, indent=2, default=str)

            console.print(f"[green]✓ Backfill plan saved to {output_plan}[/green]")
            console.print(f"[yellow]Execute with:[/yellow]")
            console.print(f"[bold]  odds backfill execute --plan {output_plan}[/bold]\n")
        else:
            console.print(
                "[dim]Tip: Use --output-plan gaps.json to generate a backfill plan[/dim]\n"
            )
