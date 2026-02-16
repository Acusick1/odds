"""CLI commands for discovering historical games."""

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from odds_core.api_models import create_scheduled_event
from odds_core.database import async_session_maker
from odds_core.models import Event
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.storage.writers import OddsWriter
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

app = typer.Typer()
console = Console()
today = datetime.now(tz=UTC).date()


@app.command()
def games(
    start: str = typer.Option(
        (today - timedelta(days=1)).isoformat(), "--start", help="Start date (YYYY-MM-DD)"
    ),
    end: str = typer.Option(today.isoformat(), "--end", help="End date (YYYY-MM-DD)"),
    sport: str = typer.Option(
        "basketball_nba", "--sport", "-s", help="Sport to discover games for"
    ),
):
    """
    Discover and catalog historical NBA games for backfill planning.

    This command fetches historical event metadata from the API for the specified date range
    and stores them in the database with SCHEDULED status. This enables zero-cost planning
    of future backfill operations.

    Note: Final scores are NOT available from The Odds API for games older than 3 days.
    Use this command to discover game schedules, then use the backfill command to fetch
    historical odds data for selected games.

    Examples:
        odds discover games  # Discover games from the past day
        odds discover games --start 2024-10-01 --end 2024-10-31
    """
    asyncio.run(_discover_games(start, end, sport))


async def _discover_games(start_date_str: str, end_date_str: str, sport: str):
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
    console.print()

    # Track statistics
    total_events_found = 0
    total_api_requests = 0
    quota_remaining = None
    all_events: list[Event] = []

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
            "Scanning dates...",
            total=days_in_range,
        )

        async with TheOddsAPIClient() as client:
            # Iterate through each date in range
            current_date = start_date
            while current_date <= end_date:
                # Format date for API (ISO 8601 format with Z suffix - API rejects +00:00 format)
                date_str = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")

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

                    # Create Event objects with SCHEDULED status
                    for event_data in events_data:
                        try:
                            # Create scheduled event (no scores)
                            event = create_scheduled_event(event_data)
                            all_events.append(event)
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Failed to parse event {event_data.get('id')}: {str(e)}[/yellow]"
                            )

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
    console.print(f"  Events parsed: {len(all_events)}")
    console.print(f"  API requests made: {total_api_requests}")
    if quota_remaining is not None:
        console.print(f"  API quota remaining: {quota_remaining:,}")

    # Deduplicate events (API returns same event across multiple dates)
    seen: dict[str, Event] = {}
    for event in all_events:
        seen[event.id] = event
    unique_events = list(seen.values())

    if unique_events:
        console.print(f"  Unique events: {len(unique_events)}")
        console.print("\n[bold blue]Storing events in database...[/bold blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            store_task = progress.add_task(
                "Upserting events...",
                total=len(unique_events),
            )

            async with async_session_maker() as session:
                writer = OddsWriter(session)

                # Batch upsert in chunks of 100
                batch_size = 100
                total_inserted = 0
                total_updated = 0

                for i in range(0, len(unique_events), batch_size):
                    batch = unique_events[i : i + batch_size]

                    try:
                        result = await writer.bulk_upsert_events(batch)
                        total_inserted += result["inserted"]
                        total_updated += result["updated"]

                        progress.update(store_task, advance=len(batch))
                    except Exception as e:
                        console.print(f"[yellow]Warning: Batch upsert failed: {str(e)}[/yellow]")

                # Commit transaction
                await session.commit()

        console.print("\n[bold green]✓ Storage completed![/bold green]")
        console.print(f"  Events inserted: {total_inserted}")
        console.print(f"  Events updated: {total_updated}")
        console.print("\n[dim]Note: Events stored with SCHEDULED status (no scores).[/dim]")
        console.print(
            "[dim]Use 'odds backfill' to fetch historical odds for discovered games.[/dim]"
        )

    elif not unique_events:
        console.print("\n[yellow]No events found in date range[/yellow]")

    console.print()
