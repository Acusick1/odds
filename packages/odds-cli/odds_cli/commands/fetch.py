"""CLI commands for fetching odds data."""

import asyncio

import typer
from odds_core.api_models import parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.ingestion import OddsIngestionCallbacks, OddsIngestionService
from odds_lambda.storage.writers import OddsWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer()
console = Console()


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
            app_settings = get_settings()

            async with TheOddsAPIClient() as client:
                ingestion_service = OddsIngestionService(client=client, settings=app_settings)

                def _on_fetch_complete(response):
                    progress.update(
                        task,
                        description=f"Processing {len(response.events)} events...",
                    )

                def _on_event_failed(event_id: str | None, exc: Exception) -> None:
                    identifier = event_id or "unknown"
                    console.print(
                        f"[yellow]Warning: Failed to process event {identifier}: {exc}[/yellow]"
                    )

                callbacks = OddsIngestionCallbacks(
                    on_fetch_complete=_on_fetch_complete,
                    on_event_failed=_on_event_failed,
                )

                result = await ingestion_service.ingest_sport(sport, callbacks=callbacks)

                progress.update(task, description="Complete!", completed=True)

                # Summary
                console.print("\n[bold green]✓ Fetch completed successfully![/bold green]")
                console.print(
                    f"  Events processed: {result.processed_events} of {result.total_events}"
                )
                console.print(f"  Bookmakers: {len(app_settings.data_collection.bookmakers)}")
                console.print(f"  Markets: {', '.join(app_settings.data_collection.markets)}")

                if result.quota_remaining is not None:
                    console.print(f"  API quota remaining: {result.quota_remaining:,}")

                console.print(f"  Response time: {result.response_time_ms}ms")

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

                progress.update(
                    task,
                    description=f"Processing {len(response.scores_data)} events...",
                )

                # Update database
                async with async_session_maker() as session:
                    writer = OddsWriter(session)
                    updated = 0

                    for score_data in response.scores_data:
                        try:
                            event_id = score_data.get("id")
                            completed = score_data.get("completed", False)

                            if completed and event_id:
                                # Extract scores using helper
                                home_score, away_score = parse_scores_from_api_dict(score_data)

                                if home_score is not None and away_score is not None:
                                    from odds_core.models import EventStatus

                                    await writer.update_event_status(
                                        event_id=event_id,
                                        status=EventStatus.FINAL,
                                        home_score=home_score,
                                        away_score=away_score,
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
            console.print(f"  Total events: {len(response.scores_data)}")

        except Exception as e:
            progress.update(task, description="Failed!", completed=True)
            console.print(f"\n[bold red]✗ Fetch failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1) from e
