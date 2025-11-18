"""Historical data backfill commands."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from odds_analytics.backfill_executor import BackfillExecutor, BackfillProgress
from odds_analytics.game_selector import GameSelector
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.time import utc_isoformat
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Historical data backfill operations")
console = Console()


@app.command("plan")
def create_backfill_plan(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    target_games: int | None = typer.Option(
        None,
        "--games",
        "-g",
        help="Target number of games to backfill (default: all discovered games)",
    ),
    output_file: str = typer.Option(
        "backfill_plan.json", "--output", "-o", help="Output file for backfill plan"
    ),
):
    """
    Create a backfill plan from games already in the database.

    This queries the local database to find completed games in the date range
    and generates an execution plan for fetching historical odds. Events must
    already exist in the database before creating a plan.

    Workflow:
        1. Discover events: odds discover games --start DATE --end DATE
        2. Plan backfill: odds backfill plan --start DATE --end DATE [--games N]
        3. Execute plan: odds backfill execute --plan backfill_plan.json

    Examples:
        # Backfill all discovered games in date range
        odds backfill plan --start 2023-10-01 --end 2024-04-30

        # Backfill specific number of games (balanced selection)
        odds backfill plan --start 2023-10-01 --end 2024-04-30 --games 166
    """
    console.print("\n[bold cyan]Creating Historical Backfill Plan[/bold cyan]\n")

    asyncio.run(_create_plan_async(start_date, end_date, target_games, output_file))


async def _create_plan_async(
    start_date_str: str,
    end_date_str: str,
    target_games: int | None,
    output_file: str,
):
    """Async implementation of plan creation."""
    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    # Initialize selector
    # If target_games is None, we'll select all games (set high limit)
    # Otherwise use specified target with team distribution
    selector = GameSelector(
        start_date=start_date,
        end_date=end_date,
        target_games=target_games or 10000,  # High default for "select all"
        games_per_team=max(1, (target_games // 30) if target_games else 10000),
    )

    # Query database for events (SCHEDULED or FINAL) with past commence times
    console.print("[cyan]Querying local database for events...[/cyan]")
    console.print(f"Date range: {start_date_str} to {end_date_str}")

    from datetime import UTC

    from odds_lambda.storage.readers import OddsReader

    events_by_date = {}

    async with async_session_maker() as session:
        reader = OddsReader(session)

        # Query all events in date range (don't filter by status yet)
        all_events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            sport_key="basketball_nba",
        )

        # Filter to events with past commence times (eligible for historical backfill)
        now = datetime.now(UTC)
        events = [e for e in all_events if e.commence_time < now]

        if not events:
            console.print(
                f"[red]No events found in database for date range {start_date_str} to {end_date_str}[/red]"
            )
            console.print("\n[yellow]Run event discovery first:[/yellow]")
            console.print(f"  odds discover games --start {start_date_str} --end {end_date_str}")
            console.print("\n[yellow]Then create your plan:[/yellow]")
            games_param = f" --games {target_games}" if target_games else ""
            console.print(
                f"  odds backfill plan --start {start_date_str} --end {end_date_str}{games_param}\n"
            )
            raise typer.Exit(1)

        console.print(f"Found {len(events)} completed games in database")

        # Group events by date (similar to API response structure)
        for event in events:
            date_str = utc_isoformat(event.commence_time)
            # Convert Event model to dict format expected by GameSelector
            event_dict = {
                "id": event.id,
                "sport_key": event.sport_key,
                "sport_title": event.sport_title,
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [],  # Not needed for plan generation
            }

            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event_dict)

        # Display date distribution
        dates_with_games = len(events_by_date)
        console.print(f"Events span {dates_with_games} unique dates")

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

    from odds_core.models import OddsSnapshot
    from sqlalchemy import func, select
    from sqlalchemy.sql.elements import ColumnElement

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


@app.command("scores")
def backfill_scores(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be updated without making changes"
    ),
):
    """
    Backfill historical scores using nba_api for events missing scores.

    This command uses the nba_api library to fetch historical game results
    and update events in the database that don't have final scores populated.

    Example:
        odds backfill scores --start 2023-10-01 --end 2024-04-30
        odds backfill scores --start 2023-10-01 --end 2024-04-30 --dry-run
    """
    console.print("\n[bold cyan]Historical Score Backfill (NBA API)[/bold cyan]\n")
    asyncio.run(_backfill_scores_async(start_date, end_date, dry_run))


async def _backfill_scores_async(start_date_str: str, end_date_str: str, dry_run: bool):
    """Async implementation of score backfill."""
    try:
        start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=UTC)
        end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=UTC)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    # Display settings
    console.print(f"[bold]Date range:[/bold] {start_date_str} to {end_date_str}")
    if dry_run:
        console.print("[yellow]DRY RUN - No database changes will be made[/yellow]\n")

    # Import here to avoid circular dependencies
    from odds_core.models import EventStatus
    from odds_lambda.nba_score_fetcher import NBAScoreFetcher
    from odds_lambda.storage.readers import OddsReader
    from odds_lambda.storage.writers import OddsWriter

    # Query events without scores in date range
    console.print("[cyan]Querying database for events without scores...[/cyan]")

    async with async_session_maker() as session:
        reader = OddsReader(session)
        writer = OddsWriter(session)

        # Get all events in date range
        all_events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            sport_key="basketball_nba",
        )

        # Filter to events without scores (home_score is None or away_score is None)
        events_without_scores = [
            e for e in all_events if e.home_score is None or e.away_score is None
        ]

        if not events_without_scores:
            console.print(
                "[green]No events found missing scores in the specified date range.[/green]"
            )
            console.print()
            return

        console.print(f"Found {len(events_without_scores)} events without scores\n")

        # Initialize NBA API fetcher
        fetcher = NBAScoreFetcher()

        # Fetch historical scores from NBA API
        console.print("[cyan]Fetching historical scores from NBA API...[/cyan]")
        try:
            nba_scores = fetcher.get_historical_scores(start_date, end_date)
            console.print(f"Fetched {len(nba_scores)} games from NBA API\n")
        except Exception as e:
            console.print(f"[red]Failed to fetch scores from NBA API: {e}[/red]")
            raise typer.Exit(1) from e

        # Match events with NBA scores and update
        updated_count = 0
        not_found_count = 0
        failed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing events...", total=len(events_without_scores))

            for event in events_without_scores:
                try:
                    # Try to match event with NBA API data
                    matched_game = fetcher.match_game_by_teams_and_date(
                        home_team=event.home_team,
                        away_team=event.away_team,
                        game_date=event.commence_time,
                        tolerance_hours=24,
                    )

                    if matched_game:
                        home_score = matched_game["home_score"]
                        away_score = matched_game["away_score"]

                        if not dry_run:
                            await writer.update_event_status(
                                event_id=event.id,
                                status=EventStatus.FINAL,
                                home_score=home_score,
                                away_score=away_score,
                            )

                        console.print(
                            f"[green]✓[/green] {event.away_team} @ {event.home_team} "
                            f"({event.commence_time.strftime('%Y-%m-%d')}): "
                            f"{away_score}-{home_score}"
                        )
                        updated_count += 1
                    else:
                        console.print(
                            f"[yellow]⚠[/yellow] {event.away_team} @ {event.home_team} "
                            f"({event.commence_time.strftime('%Y-%m-%d')}): "
                            f"No matching game found"
                        )
                        not_found_count += 1

                except Exception as e:
                    console.print(
                        f"[red]✗[/red] {event.away_team} @ {event.home_team}: "
                        f"Error: {str(e)[:60]}"
                    )
                    failed_count += 1

                progress.advance(task)

        # Commit changes if not dry run
        if not dry_run and updated_count > 0:
            await session.commit()
            console.print("\n[green]Changes committed to database[/green]")

    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Score Backfill Complete[/bold cyan]")
    console.print("=" * 60)

    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Events Processed", str(len(events_without_scores)))
    summary_table.add_row("Scores Updated", str(updated_count))
    summary_table.add_row("Not Found in NBA API", str(not_found_count))
    summary_table.add_row("Failed", str(failed_count))

    console.print("\n")
    console.print(summary_table)
    console.print()


@app.command("gaps")
def detect_and_plan_gaps(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    max_quota: int | None = typer.Option(
        None,
        "--max-quota",
        help="Maximum quota to use (default: unlimited)",
    ),
    output_file: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save plan to file (e.g., gap_plan.json)",
    ),
):
    """
    Detect gaps in tier coverage and generate backfill plan.

    This command identifies missing fetch tier data (OPENING, EARLY, SHARP,
    PREGAME, CLOSING) and generates an executable backfill plan to fill those gaps.

    Games are prioritized by highest-priority missing tier:
    - CLOSING (highest priority) - closest to game start
    - PREGAME
    - SHARP
    - EARLY
    - OPENING (lowest priority) - earliest snapshots

    When quota is limited, only complete games are included (all missing tiers
    filled per game). Partial games are never included.

    Examples:
        # Detect gaps and show summary
        odds backfill gaps --start 2024-10-01 --end 2024-10-31

        # Generate plan with quota limit
        odds backfill gaps --start 2024-10-01 --end 2024-10-31 --max-quota 3000

        # Generate and save plan
        odds backfill gaps --start 2024-10-01 --end 2024-10-31 --output gap_plan.json

    After generating a plan, execute it with:
        odds backfill execute --plan gap_plan.json
    """
    console.print("\n[bold cyan]Gap Detection and Backfill Planning[/bold cyan]\n")
    asyncio.run(_detect_and_plan_gaps_async(start_date, end_date, max_quota, output_file))


async def _detect_and_plan_gaps_async(
    start_date_str: str,
    end_date_str: str,
    max_quota: int | None,
    output_file: str | None,
):
    """Async implementation of gap detection and planning."""
    try:
        start_date = datetime.fromisoformat(start_date_str).date()
        end_date = datetime.fromisoformat(end_date_str).date()
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    # Display settings
    console.print(f"[bold]Date range:[/bold] {start_date_str} to {end_date_str}")
    if max_quota:
        console.print(f"[bold]Max quota:[/bold] {max_quota:,} units")
    else:
        console.print("[bold]Max quota:[/bold] unlimited")
    console.print()

    # Import here to avoid circular dependencies
    from odds_analytics.gap_backfill_planner import GapBackfillPlanner
    from odds_analytics.utils import create_tier_coverage_table

    # Analyze gaps
    console.print("[cyan]Analyzing tier coverage gaps...[/cyan]")

    async with async_session_maker() as session:
        planner = GapBackfillPlanner(session)

        # Run gap analysis
        try:
            analysis = await planner.analyze_gaps(start_date, end_date)
        except Exception as e:
            console.print(f"\n[red]Gap analysis failed: {e}[/red]")
            raise typer.Exit(1) from e

        # Display analysis summary
        console.print("\n[bold]Gap Analysis Summary:[/bold]\n")

        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Total Games", str(analysis.total_games))
        summary_table.add_row(
            "Games with Gaps",
            f"[red]{analysis.games_with_gaps}[/red]"
            if analysis.games_with_gaps > 0
            else "[green]0[/green]",
        )
        summary_table.add_row(
            "Total Missing Snapshots",
            f"[red]{analysis.total_missing_snapshots}[/red]"
            if analysis.total_missing_snapshots > 0
            else "[green]0[/green]",
        )

        console.print(summary_table)
        console.print()

        # If no gaps, exit early
        if analysis.games_with_gaps == 0:
            console.print("[green]No gaps found! All games have complete tier coverage.[/green]\n")
            return

        # Display games by priority tier
        console.print("[bold]Games Missing Each Tier (by priority):[/bold]\n")

        priority_table = Table(show_header=True)
        priority_table.add_column("Priority", style="cyan")
        priority_table.add_column("Tier", style="cyan")
        priority_table.add_column("Games Missing", justify="right")

        from odds_lambda.fetch_tier import FetchTier

        for idx, tier in enumerate(FetchTier.get_priority_order(), start=1):
            games_missing_tier = len(analysis.games_by_priority.get(tier, []))

            if games_missing_tier > 0:
                games_str = f"[red]{games_missing_tier}[/red]"
            else:
                games_str = f"[green]{games_missing_tier}[/green]"

            priority_table.add_row(
                f"#{idx}",
                tier.value.upper(),
                games_str,
            )

        console.print(priority_table)
        console.print()

        # Generate backfill plan
        console.print("[cyan]Generating backfill plan...[/cyan]")

        try:
            plan = await planner.generate_plan(start_date, end_date, max_quota)
        except ValueError as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print(
                "[yellow]Tip: Use --max-quota with a higher value or remove the limit.[/yellow]\n"
            )
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"\n[red]Plan generation failed: {e}[/red]")
            raise typer.Exit(1) from e

        # Display plan summary
        console.print("\n[bold]Backfill Plan Summary:[/bold]\n")

        plan_table = Table(show_header=False, box=None)
        plan_table.add_column("Metric", style="cyan")
        plan_table.add_column("Value", style="white")

        plan_table.add_row("Games in Plan", str(plan["total_games"]))
        plan_table.add_row("Total Snapshots", str(plan["total_snapshots"]))
        plan_table.add_row("Estimated Quota", f"{plan['estimated_quota_usage']:,} units")

        # Calculate coverage
        if analysis.games_with_gaps > 0:
            coverage_pct = (plan["total_games"] / analysis.games_with_gaps) * 100
            coverage_str = (
                f"[green]{coverage_pct:.1f}%[/green]"
                if coverage_pct == 100
                else f"[yellow]{coverage_pct:.1f}%[/yellow]"
            )
            plan_table.add_row("Gap Coverage", coverage_str)

        console.print(plan_table)
        console.print()

        # Show incomplete coverage warning if applicable
        if max_quota and plan["total_games"] < analysis.games_with_gaps:
            skipped_games = analysis.games_with_gaps - plan["total_games"]
            console.print(
                f"[yellow]Note: {skipped_games} games excluded due to quota limit.[/yellow]"
            )
            console.print(
                "[yellow]Increase --max-quota to include more games or remove limit for full coverage.[/yellow]\n"
            )

        # Save plan to file if requested
        if output_file:
            output_path = Path(output_file)
            with open(output_path, "w") as f:
                json.dump(plan, f, indent=2)

            console.print(f"[green]Plan saved to {output_path}[/green]\n")

            # Show execution command
            console.print("[bold]Execute plan with:[/bold]")
            console.print(f"  odds backfill execute --plan {output_file}\n")
        else:
            console.print(
                "[yellow]No output file specified. Use --output to save plan for execution.[/yellow]\n"
            )
