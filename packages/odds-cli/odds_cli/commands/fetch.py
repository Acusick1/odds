"""CLI commands for fetching odds data."""

import asyncio
from datetime import timedelta

import structlog
import typer
from odds_core.api_models import parse_scores_from_api_dict
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.ingestion import OddsIngestionCallbacks, OddsIngestionService
from odds_lambda.jobs import fetch_mlb_results, fetch_oddsportal_results
from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY
from odds_lambda.odds_api_events import find_event_by_odds_api_id
from odds_lambda.storage.writers import OddsWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = structlog.get_logger()
app = typer.Typer()
console = Console()


@app.command("current")
def fetch_current(
    sport: str = typer.Option(
        ..., "--sport", "-s", help="Sport to fetch odds for (spends its key's units)"
    ),
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

                def _on_event_failed(odds_api_id: str, exc: Exception) -> None:
                    console.print(
                        f"[yellow]Warning: Failed to process event {odds_api_id}: {exc}[/yellow]"
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
                console.print(
                    f"  Bookmakers: {len(app_settings.data_collection.bookmakers_for(sport))}"
                )
                console.print(f"  Markets: {', '.join(app_settings.data_collection.markets)}")

                if result.quota_remaining is not None:
                    console.print(f"  API quota remaining: {result.quota_remaining:,}")

                console.print(f"  Response time: {result.response_time_ms}ms")

        except Exception as e:
            progress.update(task, description="Failed!", completed=True)
            console.print(f"\n[bold red]✗ Fetch failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1) from e


@app.command("results")
def fetch_results(
    sport: str = typer.Option("soccer_epl", "--sport", "-s", help="Sport to settle"),
    lookback_days: int | None = typer.Option(
        None, "--lookback-days", help="Only events this recent; omit for every pending event"
    ),
) -> None:
    """Settle past events: MLB from the MLB Stats API by gamePk, other sports from OddsPortal."""
    lookback = timedelta(days=lookback_days) if lookback_days is not None else None
    if sport == MLB_SPORT_KEY:
        stats = asyncio.run(fetch_mlb_results.process_results(lookback=lookback))
    else:
        stats = asyncio.run(
            fetch_oddsportal_results.process_results(sport=sport, lookback=lookback)
        )
    console.print(
        f"fetched={stats.matches_fetched} updated={stats.events_updated} "
        f"cancelled={stats.events_cancelled} postponed={stats.events_postponed} "
        f"moved={stats.events_moved} unresolved={stats.events_unresolved} "
        f"errors={len(stats.errors)}"
    )
    if stats.errors:
        raise typer.Exit(code=1)


@app.command("scores")
def fetch_scores(
    sport: str = typer.Option(
        ..., "--sport", "-s", help="Sport to fetch scores for (spends its key's units)"
    ),
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
                            odds_api_id = score_data.get("id")
                            completed = score_data.get("completed", False)

                            if completed and odds_api_id:
                                # Extract scores using helper
                                home_score, away_score = parse_scores_from_api_dict(score_data)

                                if home_score is not None and away_score is not None:
                                    from odds_core.models import EventStatus

                                    event = await find_event_by_odds_api_id(session, odds_api_id)
                                    if event is None:
                                        logger.warning(
                                            "score_event_not_found", odds_api_id=odds_api_id
                                        )
                                        continue
                                    await writer.update_event_status(
                                        event_id=event.id,
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
