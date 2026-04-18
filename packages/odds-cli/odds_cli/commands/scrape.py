"""CLI commands for scraping OddsPortal upcoming odds."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("upcoming")
def scrape_upcoming(
    league: str = typer.Option(
        "england-premier-league", "--league", "-l", help="OddsHarvester league name"
    ),
    market: Annotated[
        list[str] | None, typer.Option("--market", "-m", help="Markets to scrape (repeatable)")
    ] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Scrape and convert but don't store"),
    from_file: str | None = typer.Option(
        None, "--from-file", help="Load matches from JSON file instead of scraping"
    ),
    auto_totals: bool = typer.Option(
        False,
        "--auto-totals",
        help=(
            "MLB only: look up each game's featured Over/Under line on ESPN, "
            "then run a targeted per-line scrape so each match only opens its "
            "own O/U submarket."
        ),
    ),
) -> None:
    """Scrape upcoming match odds from OddsPortal and ingest into the pipeline."""
    asyncio.run(_scrape_upcoming(league, market, dry_run, from_file, auto_totals))


async def _scrape_upcoming(
    league: str,
    markets: list[str] | None,
    dry_run: bool,
    from_file: str | None,
    auto_totals: bool,
) -> None:
    import dataclasses

    from odds_lambda.jobs.fetch_oddsportal import (
        LEAGUE_SPEC_BY_NAME,
        ingest_league,
        ingest_mlb_with_espn_totals,
    )
    from odds_lambda.oddsportal_adapter import convert_upcoming_matches

    # Look up known league spec
    if league not in LEAGUE_SPEC_BY_NAME:
        known_names = sorted(LEAGUE_SPEC_BY_NAME.keys())
        console.print(f"[red]Unknown league '{league}'. Known leagues: {known_names}[/red]")
        raise typer.Exit(code=1)

    spec = LEAGUE_SPEC_BY_NAME[league]
    if markets:
        spec = dataclasses.replace(spec, markets=markets)

    if auto_totals:
        if spec.sport_key != "baseball_mlb":
            console.print(
                f"[yellow]--auto-totals is MLB-only; ignored for sport '{spec.sport_key}'[/yellow]"
            )
        elif from_file:
            console.print(
                "[yellow]--auto-totals is incompatible with --from-file; "
                "ignoring --auto-totals[/yellow]"
            )
        else:
            console.print("Running MLB targeted scrape via ESPN main totals...")
            stats = await ingest_mlb_with_espn_totals(dry_run=dry_run)
            _print_stats(stats, dry_run=dry_run)
            return

    raw_matches: list[dict[str, Any]] | None = None
    if from_file:
        console.print(f"Loading matches from [cyan]{from_file}[/cyan]")
        with open(from_file) as f:
            loaded: list[dict[str, Any]] = json.load(f)
        raw_matches = loaded
        console.print(f"  Loaded {len(loaded)} matches")

    if dry_run:
        if raw_matches is None:
            from odds_lambda.jobs.fetch_oddsportal import run_harvester_upcoming

            console.print(f"Scraping [bold]{spec.league}[/bold] ({spec.sport})...")
            raw_matches = await run_harvester_upcoming(spec)
            console.print(f"  Scraped {len(raw_matches)} matches")

        for mkt in spec.markets:
            converted = convert_upcoming_matches(raw_matches, mkt)
            console.print(f"\n[bold]{mkt}[/bold]: {len(converted)} matches converted")

            if converted:
                table = Table(show_header=True)
                table.add_column("Match")
                table.add_column("Date")
                table.add_column("Bookmakers", justify="right")

                for m in converted[:10]:
                    table.add_row(
                        f"{m.home_team} vs {m.away_team}",
                        m.match_date.strftime("%Y-%m-%d %H:%M"),
                        str(m.bookmaker_count),
                    )

                console.print(table)
        return

    console.print(f"Scraping and ingesting [bold]{spec.league}[/bold] ({spec.sport})...")
    stats = await ingest_league(spec, raw_matches=raw_matches, dry_run=False)
    _print_stats(stats, dry_run=False)


def _print_stats(stats: Any, *, dry_run: bool) -> None:
    console.print("\n[bold green]Done[/bold green]")
    console.print(f"  Matches scraped:    {stats.matches_scraped}")
    console.print(f"  Matches converted:  {stats.matches_converted}")
    if not dry_run:
        console.print(f"  Events matched:     {stats.events_matched}")
        console.print(f"  Events created:     {stats.events_created}")
        console.print(f"  Snapshots stored:   {stats.snapshots_stored}")
    if stats.errors:
        console.print(f"  [yellow]Errors: {len(stats.errors)}[/yellow]")
        for err in stats.errors[:5]:
            console.print(f"    {err}")
