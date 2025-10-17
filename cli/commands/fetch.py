"""CLI commands for fetching odds data."""

import asyncio

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import settings
from core.data_fetcher import TheOddsAPIClient
from core.database import async_session_maker
from storage.writers import OddsWriter

app = typer.Typer()
console = Console()
logger = structlog.get_logger()


@app.command("current")
def fetch_current(
    sport: str = typer.Option("basketball_nba", "--sport", "-s", help="Sport to fetch odds for"),
):
    """Fetch current odds for upcoming games."""
    asyncio.run(_fetch_current(sport))


async def _fetch_current(sport: str):
    """Async implementation of fetch current."""
    console.print(f"[bold blue]Fetching odds for {sport}...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description="Connecting to API...", total=None)

        try:
            async with TheOddsAPIClient() as client:
                progress.update(task, description="Fetching odds data...")

                # Fetch odds
                response = await client.get_odds(
                    sport=sport,
                    regions=settings.regions,
                    markets=settings.markets,
                    bookmakers=settings.bookmakers,
                )

                events_data = response.get("data", [])
                if not isinstance(events_data, list):
                    events_data = [events_data] if events_data else []

                progress.update(
                    task,
                    description=f"Processing {len(events_data)} events...",
                )

                # Store to database (each event in its own transaction for isolation)
                processed = 0
                for event_data in events_data:
                    try:
                        async with async_session_maker() as session:
                            writer = OddsWriter(session)

                            # Upsert event
                            await writer.upsert_event(event_data)

                            # Flush to ensure event exists before logging quality issues
                            await session.flush()

                            # Store odds snapshot
                            await writer.store_odds_snapshot(
                                event_id=event_data["id"],
                                raw_data=event_data,
                                snapshot_time=response["timestamp"],
                                validate=settings.enable_validation,
                            )

                            await session.commit()
                            processed += 1

                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Failed to process event "
                            f"{event_data.get('id')}: {str(e)}[/yellow]"
                        )
                        continue

                # Log fetch in separate transaction
                async with async_session_maker() as session:
                    writer = OddsWriter(session)
                    await writer.log_fetch(
                        sport_key=sport,
                        events_count=len(events_data),
                        bookmakers_count=len(settings.bookmakers),
                        success=True,
                        api_quota_remaining=response.get("quota_remaining"),
                        response_time_ms=response.get("response_time_ms"),
                    )
                    await session.commit()

                progress.update(task, description="Complete!", completed=True)

            # Summary
            console.print("\n[bold green]✓ Fetch completed successfully![/bold green]")
            console.print(f"  Events processed: {processed}")
            console.print(f"  Bookmakers: {len(settings.bookmakers)}")
            console.print(f"  Markets: {', '.join(settings.markets)}")

            quota_remaining = response.get("quota_remaining")
            if quota_remaining:
                console.print(f"  API quota remaining: {quota_remaining:,}")

            response_time = response.get("response_time_ms")
            if response_time:
                console.print(f"  Response time: {response_time}ms")

        except Exception as e:
            progress.update(task, description="Failed!", completed=True)
            console.print(f"\n[bold red]✗ Fetch failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1) from e


@app.command("scores")
def fetch_scores(
    sport: str = typer.Option("basketball_nba", "--sport", "-s", help="Sport to fetch scores for"),
    days: int = typer.Option(3, "--days", "-d", help="Number of days back to fetch scores"),
):
    """Fetch scores for completed games."""
    asyncio.run(_fetch_scores(sport, days))


async def _fetch_scores(sport: str, days: int):
    """Async implementation of fetch scores."""
    console.print(f"[bold blue]Fetching scores for {sport} (last {days} days)...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description="Connecting to API...", total=None)

        try:
            async with TheOddsAPIClient() as client:
                progress.update(task, description="Fetching scores data...")

                # Fetch scores
                response = await client.get_scores(sport=sport, days_from=days)

                scores_data = response.get("data", [])
                if not isinstance(scores_data, list):
                    scores_data = [scores_data] if scores_data else []

                progress.update(
                    task,
                    description=f"Processing {len(scores_data)} events...",
                )

                # Update database
                async with async_session_maker() as session:
                    writer = OddsWriter(session)
                    updated = 0

                    for score_data in scores_data:
                        try:
                            event_id = score_data.get("id")
                            completed = score_data.get("completed", False)

                            if completed and event_id:
                                scores = score_data.get("scores", [])

                                # Extract scores
                                home_score = None
                                away_score = None

                                for score in scores:
                                    if score.get("name") == score_data.get("home_team"):
                                        home_score = score.get("score")
                                    if score.get("name") == score_data.get("away_team"):
                                        away_score = score.get("score")

                                if home_score is not None and away_score is not None:
                                    from core.models import EventStatus

                                    await writer.update_event_status(
                                        event_id=event_id,
                                        status=EventStatus.FINAL,
                                        home_score=int(home_score),
                                        away_score=int(away_score),
                                    )
                                    updated += 1

                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Failed to process score for "
                                f"{score_data.get('id')}: {str(e)}[/yellow]"
                            )

                    await session.commit()

                progress.update(task, description="Complete!", completed=True)

            # Summary
            console.print("\n[bold green]✓ Scores fetch completed![/bold green]")
            console.print(f"  Events updated: {updated}")
            console.print(f"  Total events: {len(scores_data)}")

        except Exception as e:
            progress.update(task, description="Failed!", completed=True)
            console.print(f"\n[bold red]✗ Fetch failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1) from e
