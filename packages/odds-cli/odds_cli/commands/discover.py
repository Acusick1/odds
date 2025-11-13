"""CLI commands for discovering historical games."""

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from odds_core.api_models import create_completed_event, parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import Event
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.storage.writers import OddsWriter
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def games(
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)"),
    sport: str = typer.Option("basketball_nba", "--sport", "-s", help="Sport to discover games for"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview operations without database writes"),
):
    """
    Discover and store historical NBA games with final scores.

    This command fetches historical events from the API for the specified date range,
    retrieves their final scores, and stores them in the database with FINAL status.
    Perfect for populating the database with completed games for backtesting.

    Examples:
        odds discover games --start 2024-10-01 --end 2024-10-31
        odds discover games --start 2024-10-01 --end 2024-10-31 --dry-run
    """
    asyncio.run(_discover_games(start, end, sport, dry_run))


async def _discover_games(start_date_str: str, end_date_str: str, sport: str, dry_run: bool):
    """Async implementation of discover games."""
    # Validate and parse dates
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as e:
        console.print(f"[bold red]✗ Invalid date format: {e}[/bold red]")
        console.print("Please use YYYY-MM-DD format (e.g., 2024-10-01)")
        raise typer.Exit(code=1) from e

    if start_date > end_date:
        console.print("[bold red]✗ Start date must be before or equal to end date[/bold red]")
        raise typer.Exit(code=1)

    # Calculate date range
    days_in_range = (end_date - start_date).days + 1

    # Display header
    console.print(f"\n[bold blue]Discovering {sport} games[/bold blue]")
    console.print(f"  Date range: {start_date_str} to {end_date_str} ({days_in_range} days)")
    if dry_run:
        console.print("  [yellow]DRY RUN MODE - No database writes[/yellow]")
    console.print()

    # Track statistics
    total_events_found = 0
    total_events_with_scores = 0
    total_api_requests = 0
    quota_remaining = None
    all_completed_events: list[Event] = []

    # Progress tracking
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Task for iterating through dates
        date_task = progress.add_task(
            f"Scanning dates...",
            total=days_in_range,
        )

        async with TheOddsAPIClient() as client:
            # Iterate through each date in range
            current_date = start_date
            while current_date <= end_date:
                # Format date for API (ISO 8601 format)
                date_str = current_date.isoformat()

                try:
                    # Fetch historical events for this date
                    progress.update(
                        date_task,
                        description=f"Fetching events for {current_date.strftime('%Y-%m-%d')}...",
                    )

                    response = await client.get_historical_events(sport=sport, date=date_str)
                    total_api_requests += 1

                    # Update quota tracking
                    if response.get("quota_remaining") is not None:
                        quota_remaining = response["quota_remaining"]

                    # Extract events from response
                    events_data = response.get("data", [])
                    if isinstance(events_data, dict):
                        events_data = events_data.get("data", [])
                    if not isinstance(events_data, list):
                        events_data = []

                    total_events_found += len(events_data)

                    # For each event, we need to fetch scores
                    if events_data:
                        progress.update(
                            date_task,
                            description=f"Fetching scores for {len(events_data)} events on {current_date.strftime('%Y-%m-%d')}...",
                        )

                        # Fetch scores for these events
                        # Use a reasonable days_from value to capture scores
                        # We'll fetch scores from 1 day after the current date to capture games that might have run late
                        days_from_current = (datetime.now(UTC) - current_date).days + 1
                        if days_from_current > 0:
                            scores_response = await client.get_scores(
                                sport=sport, days_from=min(days_from_current, 365)
                            )
                            total_api_requests += 1

                            if scores_response.quota_remaining is not None:
                                quota_remaining = scores_response.quota_remaining

                            # Build a lookup of event_id -> scores
                            scores_lookup = {}
                            for score_data in scores_response.scores_data:
                                event_id = score_data.get("id")
                                if event_id and score_data.get("completed"):
                                    scores_lookup[event_id] = score_data

                            # Match events with scores and create completed Event objects
                            for event_data in events_data:
                                event_id = event_data.get("id")

                                if event_id and event_id in scores_lookup:
                                    score_data = scores_lookup[event_id]
                                    # Merge event data with score data
                                    merged_data = {**event_data, **score_data}

                                    try:
                                        # Create completed event with scores
                                        completed_event = create_completed_event(merged_data)
                                        all_completed_events.append(completed_event)
                                        total_events_with_scores += 1
                                    except ValueError:
                                        # Score parsing failed, skip this event
                                        pass

                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to fetch data for {current_date.strftime('%Y-%m-%d')}: {str(e)}[/yellow]"
                    )

                # Move to next day
                current_date += timedelta(days=1)
                progress.update(date_task, advance=1)

    # Display discovered events summary
    console.print("\n[bold]Discovery Summary:[/bold]")
    console.print(f"  Events found: {total_events_found}")
    console.print(f"  Events with scores: {total_events_with_scores}")
    console.print(f"  API requests made: {total_api_requests}")
    if quota_remaining is not None:
        console.print(f"  API quota remaining: {quota_remaining:,}")

    # Store in database if not dry run
    if not dry_run and all_completed_events:
        console.print("\n[bold blue]Storing events in database...[/bold blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            store_task = progress.add_task(
                "Upserting events...",
                total=len(all_completed_events),
            )

            async with async_session_maker() as session:
                writer = OddsWriter(session)

                # Batch upsert in chunks of 100
                batch_size = 100
                total_inserted = 0
                total_updated = 0

                for i in range(0, len(all_completed_events), batch_size):
                    batch = all_completed_events[i : i + batch_size]

                    try:
                        result = await writer.bulk_upsert_events(batch)
                        total_inserted += result["inserted"]
                        total_updated += result["updated"]

                        progress.update(store_task, advance=len(batch))
                    except Exception as e:
                        console.print(f"[yellow]Warning: Batch upsert failed: {str(e)}[/yellow]")

                # Commit transaction
                await session.commit()

        console.print(f"\n[bold green]✓ Storage completed![/bold green]")
        console.print(f"  Events inserted: {total_inserted}")
        console.print(f"  Events updated: {total_updated}")

    elif dry_run and all_completed_events:
        # Show sample of events in dry run mode
        console.print("\n[bold]Sample Events (first 10):[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Home Team", style="green")
        table.add_column("Away Team", style="yellow")
        table.add_column("Score", style="white")

        for event in all_completed_events[:10]:
            table.add_row(
                event.commence_time.strftime("%Y-%m-%d %H:%M"),
                event.home_team,
                event.away_team,
                f"{event.home_score}-{event.away_score}",
            )

        console.print(table)

        if len(all_completed_events) > 10:
            console.print(f"\n  ... and {len(all_completed_events) - 10} more events")

    elif not all_completed_events:
        console.print("\n[yellow]No completed events with scores found in date range[/yellow]")

    console.print()
