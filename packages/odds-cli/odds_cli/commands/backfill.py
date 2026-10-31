"""Historical data backfill commands."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from odds_analytics.backfill_executor import BackfillExecutor, CallProgress
from odds_analytics.backfill_plan import (
    Fixture,
    Manifest,
    Season,
    load_fixtures,
    plan,
)
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.time import utc_isoformat
from odds_lambda.storage.readers import OddsReader
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Historical data backfill operations")
console = Console()


@app.command("plan")
def plan_backfill(
    manifest_file: str = typer.Option(..., "--manifest", "-m", help="Acquisition manifest YAML"),
    season: list[str] | None = typer.Option(  # noqa: B008 — typer option idiom
        None, "--season", help="Plan only this manifest season (repeatable)"
    ),
    cost: bool = typer.Option(
        False, "--cost", help="Print calls and units per season and band instead of the calls"
    ),
):
    """
    Plan the Odds API historical calls an acquisition manifest needs, from stored FINAL events.

    Makes no API calls. Without --cost it prints each call's time and how many fixture bands it
    serves.

    Example:
        odds backfill plan --manifest backfills/odds-api-epl-2020-26.yaml --cost
    """
    manifest = Manifest.load(Path(manifest_file))
    seasons = _select_seasons(manifest, season)
    fixtures, balance = asyncio.run(_load_season_fixtures(manifest, seasons))
    if cost:
        console.print(_cost_table(manifest, fixtures, balance))
        return
    for name, season_fixtures in fixtures.items():
        console.print(f"[bold]{name}[/bold]")
        for call in plan(manifest, season_fixtures):
            console.print(f"  {utc_isoformat(call.time)}  {len(call.serves)}")


def _select_seasons(manifest: Manifest, season: list[str] | None) -> dict[str, Season]:
    unknown = sorted(set(season or ()) - manifest.seasons.keys())
    if unknown:
        console.print(f"[red]Not a manifest season: {', '.join(unknown)}[/red]")
        raise typer.Exit(1)
    return {name: s for name, s in manifest.seasons.items() if not season or name in season}


async def _load_season_fixtures(
    manifest: Manifest, seasons: dict[str, Season]
) -> tuple[dict[str, list[Fixture]], int | None]:
    """The seasons' fixtures, and the balance of the key the manifest's sport spends (its newest
    recorded quota), or None when no recent fetch recorded one."""
    api = get_settings().api
    async with async_session_maker() as session:
        fixtures = {
            name: await load_fixtures(session, manifest.sport, s) for name, s in seasons.items()
        }
        quota = (await OddsReader(session).get_latest_quota_by_key(api)).get(
            api.key_index_for(manifest.sport)
        )
    return fixtures, quota.remaining if quota else None


def _duration(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    for size, unit in ((1440, "d"), (60, "h")):
        if minutes % size == 0:
            return f"{minutes // size}{unit}"
    return f"{minutes}m"


def _cost_table(
    manifest: Manifest, fixtures: dict[str, list[Fixture]], balance: int | None
) -> Table:
    """Calls and units per season for all bands, then the calls each band adds to the
    others."""
    table = Table(
        title=f"Calls / units, {manifest.units_per_call} units per call. A band's row is "
        "what it adds to all the others."
    )
    table.add_column("Band")
    for name, season_fixtures in fixtures.items():
        table.add_column(f"{name} ({len(season_fixtures)} fixtures)", justify="right")
    table.add_column("Total", justify="right")

    def counts(cut: Manifest | None) -> list[int]:
        calls = [len(plan(cut, f)) if cut else 0 for f in fixtures.values()]
        return [*calls, sum(calls)]

    def cells(calls: list[int]) -> list[str]:
        return [f"{n:,} / {n * manifest.units_per_call:,}" for n in calls]

    full = counts(manifest)
    table.add_row("all", *cells(full))
    for band, rest in manifest.band_cuts():
        label = (
            f"KO-{_duration(timedelta(minutes=band.max_lead_minutes))} to "
            f"KO-{_duration(timedelta(minutes=band.min_lead_minutes))} "
            f"(max age {_duration(band.max_age)})"
        )
        table.add_row(label, *cells([a - b for a, b in zip(full, counts(rest), strict=True)]))
    key = get_settings().api.key_index_for(manifest.sport)
    if balance is None:
        table.caption = f"Key {key} balance unknown: no recent fetch recorded a quota"
    else:
        table.caption = (
            f"Key {key} remaining after the plan: "
            f"{balance - full[-1] * manifest.units_per_call:,} of {balance:,}"
        )
    return table


@app.command("execute")
def execute_backfill(
    manifest_file: str = typer.Option(..., "--manifest", "-m", help="Acquisition manifest YAML"),
    season: list[str] | None = typer.Option(  # noqa: B008 — typer option idiom
        None, "--season", help="Execute only this manifest season (repeatable)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Count the calls that would be made without making any"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip calls whose fixtures already have the snapshot the call returns",
    ),
):
    """
    Make the Odds API historical calls an acquisition manifest plans and store what they return.

    Spends paid quota on the manifest's sport key. Use --dry-run first to count the calls.

    Example:
        odds backfill execute --manifest backfills/odds-api-epl-2020-26.yaml --season 2024-25 --dry-run
    """
    manifest = Manifest.load(Path(manifest_file))
    seasons = _select_seasons(manifest, season)
    asyncio.run(_execute_backfill_async(manifest, seasons, dry_run, skip_existing))


async def _execute_backfill_async(
    manifest: Manifest, seasons: dict[str, Season], dry_run: bool, skip_existing: bool
) -> None:
    fixtures, balance = await _load_season_fixtures(manifest, seasons)
    calls = [
        call for season_fixtures in fixtures.values() for call in plan(manifest, season_fixtures)
    ]
    if not calls:
        console.print("[red]No calls planned: no FINAL fixtures in the selected seasons[/red]")
        raise typer.Exit(1)

    units = len(calls) * manifest.units_per_call
    console.print(f"[bold]Calls:[/bold] {len(calls):,}")
    console.print(f"[bold]Estimated units:[/bold] {units:,}")
    console.print(
        f"[bold]Key balance:[/bold] "
        f"{f'{balance:,}' if balance is not None else 'unknown (no recent fetch recorded a quota)'}"
    )
    if dry_run:
        console.print("\n[yellow]DRY RUN - No API calls will be made[/yellow]")
    else:
        console.print("\n[yellow]This will consume API quota. Continue?[/yellow]")
        if not typer.confirm("Proceed with backfill?"):
            console.print("[red]Aborted[/red]")
            raise typer.Exit(0)

    async with BackfillExecutor(manifest, skip_existing=skip_existing, dry_run=dry_run) as executor:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing calls...", total=len(calls))

            def on_progress(update: CallProgress) -> None:
                progress.console.print(_progress_line(update))
                progress.advance(task)

            result = await executor.execute(calls, progress_callback=on_progress)

    table = Table(title="Backfill complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Calls made", f"{result.calls_made:,}")
    table.add_row("Calls skipped", f"{result.calls_skipped:,}")
    if dry_run:
        table.add_row("Calls that would be made", f"{result.would_fetch:,}")
    table.add_row("Units used", f"{result.units_used:,}")
    table.add_row("Fixtures stored", f"{result.fixtures_stored:,}")
    table.add_row("Snapshots stored", f"{result.snapshots_stored:,}")
    table.add_row("Dropped games", f"{len(result.dropped_games):,}")
    if result.quota_remaining is not None:
        table.add_row("Key quota remaining", f"{result.quota_remaining:,}")
    console.print(table)


def _progress_line(update: CallProgress) -> str:
    when = utc_isoformat(update.time)
    if update.status == "fetched":
        return (
            f"[green]fetched[/green] {when}  {update.fixtures_stored} fixtures, "
            f"{update.units_used} units, quota {update.quota_remaining}"
        )
    if update.status == "skipped":
        return f"[dim]skipped[/dim] {when}"
    if update.status == "would_fetch":
        return f"[dim]would fetch[/dim] {when}"
    return f"[red]failed[/red] {when}  {update.message}"


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
    Backfill event scores from game log table.

    Reads scores from nba_team_game_logs (populated via `odds nba-stats fetch`)
    and updates events that are missing scores. Run `odds nba-stats fetch` first.

    Example:
        odds backfill scores --start 2023-10-01 --end 2024-04-30
        odds backfill scores --start 2023-10-01 --end 2024-04-30 --dry-run
    """
    console.print("\n[bold cyan]Score Backfill from Game Logs[/bold cyan]\n")
    asyncio.run(_backfill_scores_async(start_date, end_date, dry_run))


async def _backfill_scores_async(start_date_str: str, end_date_str: str, dry_run: bool):
    """Async implementation of score backfill using game log table."""
    try:
        start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=UTC)
        end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=UTC)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[bold]Date range:[/bold] {start_date_str} to {end_date_str}")
    if dry_run:
        console.print("[yellow]DRY RUN - No database changes will be made[/yellow]\n")

    from odds_core.models import EventStatus
    from odds_lambda.storage.game_log_reader import GameLogReader
    from odds_lambda.storage.readers import OddsReader
    from odds_lambda.storage.writers import OddsWriter

    console.print("[cyan]Querying events without scores...[/cyan]")

    async with async_session_maker() as session:
        reader = OddsReader(session)
        gl_reader = GameLogReader(session)
        writer = OddsWriter(session)

        all_events = await reader.get_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
            sport_key="basketball_nba",
        )

        events_without_scores = [
            e for e in all_events if e.home_score is None or e.away_score is None
        ]

        if not events_without_scores:
            console.print(
                "[green]No events found missing scores in the specified date range.[/green]\n"
            )
            return

        console.print(f"Found {len(events_without_scores)} events without scores\n")

        updated_count = 0
        not_found_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing events...", total=len(events_without_scores))

            for event in events_without_scores:
                game_logs = [
                    gl
                    for gl in await gl_reader.get_game_logs_for_event(event.id)
                    if gl.pts is not None
                ]

                if len(game_logs) >= 2:
                    # Determine home/away by matchup string ("vs." = home)
                    home_pts = None
                    away_pts = None
                    for gl in game_logs:
                        if "vs." in gl.matchup:
                            home_pts = gl.pts
                        elif "@" in gl.matchup:
                            away_pts = gl.pts

                    if home_pts is not None and away_pts is not None:
                        if not dry_run:
                            await writer.update_event_status(
                                event_id=event.id,
                                status=EventStatus.FINAL,
                                home_score=home_pts,
                                away_score=away_pts,
                            )

                        console.print(
                            f"[green]\u2713[/green] {event.away_team} @ {event.home_team} "
                            f"({event.commence_time.strftime('%Y-%m-%d')}): "
                            f"{away_pts}-{home_pts}"
                        )
                        updated_count += 1
                    else:
                        console.print(
                            f"[yellow]\u26a0[/yellow] {event.away_team} @ {event.home_team} "
                            f"({event.commence_time.strftime('%Y-%m-%d')}): "
                            f"Cannot determine home/away from matchup"
                        )
                        not_found_count += 1
                else:
                    console.print(
                        f"[yellow]\u26a0[/yellow] {event.away_team} @ {event.home_team} "
                        f"({event.commence_time.strftime('%Y-%m-%d')}): "
                        f"No game log data"
                    )
                    not_found_count += 1

                progress.advance(task)

        if not dry_run and updated_count > 0:
            await session.commit()
            console.print("\n[green]Changes committed to database[/green]")

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Score Backfill Complete[/bold cyan]")
    console.print("=" * 60)

    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Events Processed", str(len(events_without_scores)))
    summary_table.add_row("Scores Updated", str(updated_count))
    summary_table.add_row("No Game Log Data", str(not_found_count))

    console.print("\n")
    console.print(summary_table)

    if not_found_count > 0:
        console.print(
            "\n[dim]Tip: Run 'odds nba-stats fetch --all' to populate game log data[/dim]"
        )
    console.print()
