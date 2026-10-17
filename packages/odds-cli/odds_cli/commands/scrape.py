"""CLI commands for scraping OddsPortal upcoming odds."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("upcoming")
def scrape_upcoming(
    league: str = typer.Option(
        "england-premier-league", "--league", "-l", help="League name (as in LEAGUE_SPECS)"
    ),
    market: Annotated[
        list[str] | None,
        typer.Option(
            "--market", "-m", help="Markets to scrape (repeatable): 1x2, over_under_2_5, home_away"
        ),
    ] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Scrape and convert but don't store"),
) -> None:
    """Scrape upcoming match odds from OddsPortal and ingest into the pipeline."""
    asyncio.run(_scrape_upcoming(league, market, dry_run))


async def _scrape_upcoming(
    league: str,
    markets: list[str] | None,
    dry_run: bool,
) -> None:
    import dataclasses

    from odds_lambda.jobs.fetch_oddsportal import (
        LEAGUE_SPEC_BY_NAME,
        ingest_league,
        scrape_league,
    )
    from odds_lambda.oddsportal_client import FEED_MARKETS

    if league not in LEAGUE_SPEC_BY_NAME:
        known_names = sorted(LEAGUE_SPEC_BY_NAME.keys())
        console.print(f"[red]Unknown league '{league}'. Known leagues: {known_names}[/red]")
        raise typer.Exit(code=1)

    spec = LEAGUE_SPEC_BY_NAME[league]
    if markets:
        unknown = sorted(set(markets) - FEED_MARKETS.keys())
        if unknown:
            console.print(
                f"[red]Unknown market(s) {unknown}. Known markets: {sorted(FEED_MARKETS)}[/red]"
            )
            raise typer.Exit(code=1)
        spec = dataclasses.replace(spec, markets=markets)

    if dry_run:
        console.print(f"Scraping [bold]{spec.league}[/bold] ({spec.listing_path})...")
        scraped = await scrape_league(spec)
        console.print(f"  Listed {scraped.events_listed} events")

        for mkt in spec.markets:
            converted = [m for market_key, m in scraped.matches if market_key == mkt]
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

        for err in scraped.errors[:5]:
            console.print(f"[yellow]{err}[/yellow]")
        return

    console.print(f"Scraping and ingesting [bold]{spec.league}[/bold] ({spec.listing_path})...")
    stats = await ingest_league(spec)

    console.print("\n[bold green]Done[/bold green]")
    console.print(f"  Matches scraped:  {stats.matches_scraped}")
    console.print(f"  Events matched:   {stats.events_matched}")
    console.print(f"  Events created:   {stats.events_created}")
    console.print(f"  Snapshots stored: {stats.snapshots_stored}")

    if stats.errors:
        console.print(f"  [yellow]Errors: {len(stats.errors)}[/yellow]")
        for err in stats.errors[:5]:
            console.print(f"    {err}")
