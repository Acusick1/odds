"""NBA game log operations commands."""

import asyncio
import time

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="NBA game log operations")
console = Console()

# Seasons available for backfill (nba_api data starts 2021-22)
ALL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


@app.command("fetch")
def fetch(
    season: str | None = typer.Option(
        None,
        "--season",
        help="NBA season to fetch e.g. '2024-25' (default: current season)",
    ),
    all_seasons: bool = typer.Option(
        False,
        "--all",
        help="Fetch all seasons from 2021-22 through current",
    ),
) -> None:
    """Fetch NBA team game logs from stats.nba.com and store to database.

    Uses Playwright (headless Chrome) to bypass Akamai bot detection.

    Examples:
        odds nba-stats fetch --season 2024-25
        odds nba-stats fetch --all
    """
    console.print("\n[bold cyan]NBA Game Log Fetch[/bold cyan]\n")

    seasons = ALL_SEASONS if all_seasons else [season or _current_season()]
    console.print(f"Seasons to fetch: {', '.join(seasons)}\n")

    asyncio.run(_fetch_async(seasons))


def _current_season() -> str:
    """Derive current NBA season string from today's date."""
    from datetime import date

    today = date.today()
    # NBA season spans Oct-Jun: if before October, it's the previous year's season
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"


async def _fetch_async(seasons: list[str]) -> None:
    """Async implementation of fetch command."""
    from odds_core.database import async_session_maker
    from odds_lambda.game_log_fetcher import fetch_game_logs
    from odds_lambda.storage.game_log_writer import GameLogWriter

    total_stored = 0
    errors = 0

    for i, season in enumerate(seasons):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Fetching {season} (browser session)...", total=None)
            try:
                records = fetch_game_logs(season)
                progress.update(task, completed=True)
            except Exception as e:
                progress.stop()
                console.print(f"\n[red]Failed to fetch {season}: {e}[/red]")
                errors += 1
                continue

        if not records:
            console.print(f"[yellow]No game log data for {season}[/yellow]")
            continue

        console.print(f"  Fetched {len(records)} rows for {season}")

        async with async_session_maker() as session:
            writer = GameLogWriter(session)
            count = await writer.upsert_game_logs(records)
            await session.commit()
            total_stored += count

        console.print(f"  [green]Stored {count} game log entries[/green]")

        # Rate limit between seasons
        if i < len(seasons) - 1:
            console.print("  Waiting 10s before next season...")
            time.sleep(10)

    console.print(f"\n[green]Total: {total_stored} rows stored[/green]")
    if errors:
        console.print(f"[yellow]Errors: {errors} season(s) failed[/yellow]")
    console.print()


@app.command("status")
def status() -> None:
    """Show game log pipeline health and statistics."""
    asyncio.run(_status_async())


async def _status_async() -> None:
    """Async implementation of status command."""
    from odds_core.database import async_session_maker
    from odds_lambda.storage.game_log_reader import GameLogReader

    console.print("\n[bold cyan]Game Log Pipeline Status[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = GameLogReader(session)
            stats = await reader.get_pipeline_stats()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Rows", f"{stats['total_rows']:,}")
        table.add_row("Events Matched", f"[green]{stats['events_matched']:,}[/green]")
        table.add_row("Games Unmatched", f"[yellow]{stats['events_unmatched']:,}[/yellow]")

        if stats["earliest_game_date"] and stats["latest_game_date"]:
            table.add_row(
                "Date Coverage",
                f"{stats['earliest_game_date']} \u2192 {stats['latest_game_date']}",
            )

        console.print(table)

        # Per-season breakdown
        if stats["season_counts"]:
            console.print()
            season_table = Table(title="Rows by Season")
            season_table.add_column("Season", style="cyan")
            season_table.add_column("Rows", justify="right")
            for season_name, count in sorted(stats["season_counts"].items()):
                season_table.add_row(season_name, f"{count:,}")
            console.print(season_table)

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Failed to get status: {e}[/bold red]")
        raise typer.Exit(1) from e
