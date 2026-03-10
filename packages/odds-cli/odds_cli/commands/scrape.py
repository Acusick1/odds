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
    sport: str = typer.Option("football", "--sport", "-s", help="OddsHarvester sport name"),
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
) -> None:
    """Scrape upcoming match odds from OddsPortal and ingest into the pipeline."""
    asyncio.run(_scrape_upcoming(sport, league, market or ["1x2"], dry_run, from_file))


async def _scrape_upcoming(
    sport: str,
    league: str,
    markets: list[str],
    dry_run: bool,
    from_file: str | None,
) -> None:
    from odds_lambda.jobs.fetch_oddsportal import (
        LeagueSpec,
        ingest_league,
    )
    from odds_lambda.oddsportal_adapter import convert_upcoming_matches

    # Resolve sport_key/title from league
    sport_key, sport_title = _resolve_sport_meta(sport, league)

    spec = LeagueSpec(
        sport=sport,
        league=league,
        sport_key=sport_key,
        sport_title=sport_title,
        markets=markets,
    )

    raw_matches: list[dict[str, Any]] | None = None
    if from_file:
        console.print(f"Loading matches from [cyan]{from_file}[/cyan]")
        with open(from_file) as f:
            raw_matches = json.load(f)
        console.print(f"  Loaded {len(raw_matches)} matches")

    if dry_run:
        if raw_matches is None:
            from odds_lambda.jobs.fetch_oddsportal import run_harvester_upcoming

            console.print(f"Scraping [bold]{league}[/bold] ({sport})...")
            raw_matches = await run_harvester_upcoming(spec)
            console.print(f"  Scraped {len(raw_matches)} matches")

        for mkt in markets:
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

    console.print(f"Scraping and ingesting [bold]{league}[/bold] ({sport})...")
    stats = await ingest_league(spec, raw_matches=raw_matches, dry_run=False)

    console.print("\n[bold green]Done[/bold green]")
    console.print(f"  Matches scraped:  {stats.matches_scraped}")
    console.print(f"  Events matched:   {stats.events_matched}")
    console.print(f"  Events created:   {stats.events_created}")
    console.print(f"  Snapshots stored: {stats.snapshots_stored}")

    if stats.errors:
        console.print(f"  [yellow]Errors: {len(stats.errors)}[/yellow]")
        for err in stats.errors[:5]:
            console.print(f"    {err}")


def _resolve_sport_meta(sport: str, league: str) -> tuple[str, str]:
    """Map sport/league to pipeline sport_key and sport_title."""
    mapping: dict[str, tuple[str, str]] = {
        "england-premier-league": ("soccer_epl", "EPL"),
        "nba": ("basketball_nba", "NBA"),
    }
    if league in mapping:
        return mapping[league]
    # Generic fallback
    return f"{sport}_{league.replace('-', '_')}", league.title()
