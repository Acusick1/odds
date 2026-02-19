"""NBA injury report operations commands."""

import asyncio
from datetime import UTC, date, datetime, timedelta

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="NBA injury report operations")
console = Console()


@app.command("fetch")
def fetch():
    """Fetch the current NBA injury report and store to database."""
    console.print("\n[bold cyan]Fetching Current Injury Report[/bold cyan]\n")
    asyncio.run(_fetch_async())


async def _fetch_async() -> None:
    """Async implementation of fetch command."""
    from odds_core.database import async_session_maker
    from odds_lambda.injury_fetcher import fetch_injury_report
    from odds_lambda.storage.injury_writer import InjuryWriter

    now = datetime.now(UTC)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Fetching injury report (JVM startup may take a few seconds)...", total=None
        )
        try:
            records = fetch_injury_report(now)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Failed to fetch injury report: {e}[/red]")
            raise typer.Exit(1) from e

    if not records:
        console.print("[yellow]No injury data available for current time[/yellow]\n")
        return

    async with async_session_maker() as session:
        writer = InjuryWriter(session)
        count = await writer.upsert_injury_reports(records)
        await session.commit()

    console.print(f"[green]Stored {count} injury report entries[/green]\n")


@app.command("backfill")
def backfill(
    season: str = typer.Option(
        ...,
        "--season",
        help="NBA season to backfill e.g. '2024-25'",
    ),
    hours_before: str = typer.Option(
        "8,2",
        "--hours-before",
        help="Comma-separated hours before game to fetch reports (default: 8,2)",
    ),
    delay_ms: int = typer.Option(
        500,
        "--delay-ms",
        help="Delay between fetches in milliseconds (rate limiting)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show plan without fetching data",
    ),
):
    """Backfill historical injury report data for a season.

    Fetches injury reports at specific times relative to each game in the
    database for the given season. Uses an event-driven strategy: rather
    than crawling every hourly slot, computes target report timestamps
    from game commence_times and deduplicates across events.

    Examples:
        odds injuries backfill --season 2024-25
        odds injuries backfill --season 2024-25 --hours-before 12,8,2 --dry-run
    """
    console.print("\n[bold cyan]Injury Report Backfill[/bold cyan]\n")

    if dry_run:
        console.print("[yellow]DRY RUN - No data will be fetched or stored[/yellow]\n")

    # Parse hours
    try:
        target_hours = [float(h.strip()) for h in hours_before.split(",")]
    except ValueError as e:
        console.print(f"[red]Invalid --hours-before value: {hours_before!r}[/red]")
        raise typer.Exit(1) from e

    # Parse season date range
    season_start, season_end = _parse_season_dates(season)
    if season_start is None or season_end is None:
        console.print(f"[red]Invalid season format: {season!r}. Expected e.g. '2024-25'[/red]")
        raise typer.Exit(1)

    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Season", season)
    config_table.add_row("Date range", f"{season_start} → {season_end}")
    config_table.add_row("Hours before game", ", ".join(str(h) for h in target_hours))
    config_table.add_row("Fetch delay", f"{delay_ms}ms")
    console.print(config_table)
    console.print()

    asyncio.run(_backfill_async(season_start, season_end, target_hours, delay_ms, dry_run))


def _parse_season_dates(season: str) -> tuple[date | None, date | None]:
    """Parse season string like '2024-25' into (start_date, end_date).

    NBA regular season typically runs October to April, playoffs through June.
    """
    try:
        parts = season.split("-")
        if len(parts) != 2:
            return None, None
        start_year = int(parts[0])
        end_year_short = int(parts[1])
        end_year = start_year // 100 * 100 + end_year_short
        # Season: October start year through June end year
        return date(start_year, 10, 1), date(end_year, 6, 30)
    except (ValueError, IndexError):
        return None, None


async def _backfill_async(
    season_start: date,
    season_end: date,
    target_hours: list[float],
    delay_ms: int,
    dry_run: bool,
) -> None:
    """Async implementation of backfill command."""
    from odds_core.database import async_session_maker
    from odds_core.models import Event
    from odds_core.time import EASTERN
    from sqlalchemy import and_, select

    # Query events in the season date range
    async with async_session_maker() as session:
        query = (
            select(Event)
            .where(
                and_(
                    Event.commence_time
                    >= datetime(
                        season_start.year, season_start.month, season_start.day, tzinfo=UTC
                    ),
                    Event.commence_time
                    <= datetime(
                        season_end.year, season_end.month, season_end.day, 23, 59, 59, tzinfo=UTC
                    ),
                    Event.sport_key == "basketball_nba",
                )
            )
            .order_by(Event.commence_time)
        )
        result = await session.execute(query)
        events = list(result.scalars().all())

    if not events:
        console.print("[yellow]No events found in the database for this season[/yellow]\n")
        return

    # Compute target report timestamps and deduplicate
    target_times: set[datetime] = set()
    for event in events:
        for hours in target_hours:
            target_utc = event.commence_time - timedelta(hours=hours)
            # Round to nearest valid report slot in ET
            target_et = target_utc.astimezone(EASTERN)
            # Round down to hour for legacy era, 15-min for new era
            if target_et.replace(tzinfo=None) < datetime(2025, 12, 22, 9, 0):
                # Legacy: hourly
                rounded_et = target_et.replace(minute=0, second=0, microsecond=0)
            else:
                # New: 15-min intervals
                rounded_minute = (target_et.minute // 15) * 15
                rounded_et = target_et.replace(minute=rounded_minute, second=0, microsecond=0)
            target_times.add(rounded_et.astimezone(UTC))

    sorted_times = sorted(target_times)
    console.print(f"Events in DB: [white]{len(events)}[/white]")
    console.print(f"Unique report timestamps to fetch: [white]{len(sorted_times)}[/white]\n")

    if dry_run:
        # Show first/last few timestamps
        for t in sorted_times[:5]:
            et_str = t.astimezone(EASTERN).strftime("%Y-%m-%d %I:%M %p ET")
            console.print(f"  {et_str}")
        if len(sorted_times) > 10:
            console.print(f"  ... ({len(sorted_times) - 10} more)")
        for t in sorted_times[-5:]:
            et_str = t.astimezone(EASTERN).strftime("%Y-%m-%d %I:%M %p ET")
            console.print(f"  {et_str}")
        console.print()
        return

    # Fetch and store
    from odds_lambda.injury_fetcher import fetch_injury_report
    from odds_lambda.storage.injury_writer import InjuryWriter

    total_stored = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Backfilling 0/{len(sorted_times)} reports...", total=len(sorted_times)
        )

        for i, target_utc in enumerate(sorted_times):
            try:
                records = fetch_injury_report(target_utc)

                if records:
                    async with async_session_maker() as session:
                        writer = InjuryWriter(session)
                        count = await writer.upsert_injury_reports(records)
                        await session.commit()
                        total_stored += count

            except Exception as e:
                errors += 1
                et_str = target_utc.astimezone(EASTERN).strftime("%Y-%m-%d %I:%M %p")
                console.print(f"  [red]Error at {et_str}: {e}[/red]")

            progress.update(
                task,
                completed=i + 1,
                description=f"Backfilling {i + 1}/{len(sorted_times)} reports...",
            )

            # Rate limiting
            if delay_ms > 0 and i < len(sorted_times) - 1:
                await asyncio.sleep(delay_ms / 1000)

    console.print(f"\n[green]Backfill complete: {total_stored} entries stored[/green]")
    if errors:
        console.print(f"[yellow]Errors: {errors}[/yellow]")
    console.print()


@app.command("status")
def status():
    """Show injury report pipeline health and statistics."""
    asyncio.run(_status_async())


async def _status_async() -> None:
    """Async implementation of status command."""
    from odds_core.database import async_session_maker
    from odds_lambda.storage.injury_reader import InjuryReader

    console.print("\n[bold cyan]Injury Report Pipeline Status[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = InjuryReader(session)
            stats = await reader.get_pipeline_stats()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Reports", f"{stats['total_reports']:,}")
        table.add_row("Unique Players", f"{stats['unique_players']:,}")
        table.add_row("Events Matched", f"[green]{stats['events_matched']:,}[/green]")
        table.add_row(
            "Game/Team Pairs Unmatched", f"[yellow]{stats['events_unmatched']:,}[/yellow]"
        )

        if stats["earliest_game_date"] and stats["latest_game_date"]:
            table.add_row(
                "Date Coverage",
                f"{stats['earliest_game_date']} → {stats['latest_game_date']}",
            )

        console.print(table)

        # Status breakdown
        if stats["status_counts"]:
            console.print()
            status_table = Table(title="Status Breakdown")
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", justify="right")
            for status_name, count in sorted(stats["status_counts"].items()):
                status_table.add_row(status_name, f"{count:,}")
            console.print(status_table)

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Failed to get status: {e}[/bold red]")
        raise typer.Exit(1) from e
