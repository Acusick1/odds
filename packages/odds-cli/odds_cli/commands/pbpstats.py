"""PBPStats player season statistics commands."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from odds_lambda.pbpstats_fetcher import PlayerSeasonRecord

app = typer.Typer(help="PBPStats player season statistics")
console = Console()

ALL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]


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
    """Fetch player season stats from PBPStats API and store to database.

    Examples:
        odds pbpstats fetch --season 2024-25
        odds pbpstats fetch --all
    """
    console.print("\n[bold cyan]PBPStats Player Season Stats Fetch[/bold cyan]\n")

    seasons = ALL_SEASONS if all_seasons else [season or _current_season()]
    console.print(f"Seasons to fetch: {', '.join(seasons)}\n")

    from odds_lambda.pbpstats_fetcher import fetch_player_season_stats

    fetched: list[tuple[str, list[PlayerSeasonRecord]]] = []
    errors = 0

    for i, s in enumerate(seasons):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Fetching {s}...", total=None)
            try:
                records = fetch_player_season_stats(s)
                progress.update(task, completed=True)
            except Exception as e:
                progress.stop()
                console.print(f"\n[red]Failed to fetch {s}: {e}[/red]")
                errors += 1
                continue

        if not records:
            console.print(f"[yellow]No data for {s}[/yellow]")
            continue

        console.print(f"  Fetched {len(records)} players for {s}")
        fetched.append((s, records))

        if i < len(seasons) - 1:
            console.print("  Waiting 3s before next season...")
            time.sleep(3)

    if fetched:
        asyncio.run(_store_player_stats(fetched))

    total = sum(len(recs) for _, recs in fetched)
    console.print(f"\n[green]Total: {total} player rows stored[/green]")
    if errors:
        console.print(f"[yellow]Errors: {errors} season(s) failed[/yellow]")
    console.print()


@app.command("backfill")
def backfill() -> None:
    """Backfill all seasons (2021-22 through 2024-25).

    Equivalent to: odds pbpstats fetch --all
    """
    fetch(season=None, all_seasons=True)


@app.command("status")
def status() -> None:
    """Show player stats pipeline health and statistics."""
    asyncio.run(_status_async())


def _current_season() -> str:
    """Derive current NBA season string from today's date."""
    from datetime import date

    today = date.today()
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"


async def _store_player_stats(
    fetched: list[tuple[str, list[PlayerSeasonRecord]]],
) -> None:
    """Store fetched player stats to database."""
    from odds_core.database import async_session_maker
    from odds_lambda.storage.pbpstats_writer import PbpStatsWriter

    for season, records in fetched:
        async with async_session_maker() as session:
            writer = PbpStatsWriter(session)
            count = await writer.upsert_player_stats(records)
            await session.commit()
        console.print(f"  [green]Stored {count} player stats for {season}[/green]")


async def _status_async() -> None:
    """Async implementation of status command."""
    from odds_core.database import async_session_maker
    from odds_lambda.storage.pbpstats_reader import PbpStatsReader

    console.print("\n[bold cyan]PBPStats Player Stats Pipeline Status[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = PbpStatsReader(session)
            stats = await reader.get_pipeline_stats()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Rows", f"{stats['total_rows']:,}")
        table.add_row("Unique Players", f"{stats['unique_players']:,}")
        console.print(table)

        if stats["season_counts"]:
            console.print()
            season_table = Table(title="Players by Season")
            season_table.add_column("Season", style="cyan")
            season_table.add_column("Players", justify="right")
            for season_name, count in sorted(stats["season_counts"].items()):
                season_table.add_row(season_name, f"{count:,}")
            console.print(season_table)

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Failed to get status: {e}[/bold red]")
        raise typer.Exit(1) from e
