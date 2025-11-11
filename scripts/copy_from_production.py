"""
Copy completed games from production database to local database.

This script performs a selective merge, copying only completed games with results
from the production database while preserving existing local data.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import structlog
from odds_core.models import Event, EventStatus
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = structlog.get_logger()
console = Console()


class ProductionDataCopier:
    """Handles copying data from production to local database."""

    def __init__(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
        dry_run: bool = False,
        skip_existing: bool = True,
    ):
        """
        Initialize copier with database sessions.

        Args:
            prod_session_maker: Production database session maker
            local_session_maker: Local database session maker
            dry_run: If True, don't actually write to database
            skip_existing: If True, skip events that already exist locally
        """
        self.prod_session_maker = prod_session_maker
        self.local_session_maker = local_session_maker
        self.dry_run = dry_run
        self.skip_existing = skip_existing

        # Statistics
        self.stats = {
            "total_events": 0,
            "events_copied": 0,
            "events_skipped": 0,
            "snapshots_copied": 0,
            "errors": 0,
        }

    async def copy_completed_games(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str = "basketball_nba",
    ) -> dict:
        """
        Copy completed games from production to local.

        Args:
            start_date: Start of date range
            end_date: End of date range
            sport_key: Sport to copy (default: basketball_nba)

        Returns:
            Dictionary with copy statistics
        """
        console.print("\n[bold]Copy Parameters:[/bold]")
        console.print(f"  Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"  Sport: {sport_key}")
        console.print(f"  Dry run: {self.dry_run}")
        console.print(f"  Skip existing: {self.skip_existing}")
        console.print()

        # Query production for completed games
        async with self.prod_session_maker() as prod_session:
            prod_reader = OddsReader(prod_session)

            console.print("[bold blue]Querying production database...[/bold blue]")
            events = await prod_reader.get_events_by_date_range(
                start_date=start_date,
                end_date=end_date,
                sport_key=sport_key,
                status=EventStatus.FINAL,
            )

            # Filter to only events with scores
            events_with_scores = [
                e for e in events if e.home_score is not None and e.away_score is not None
            ]

            self.stats["total_events"] = len(events_with_scores)

            console.print(
                f"[green]Found {len(events_with_scores)} completed games in production[/green]"
            )

            if not events_with_scores:
                console.print("[yellow]No completed games found in date range[/yellow]")
                return self.stats

            # Copy each event with progress tracking
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    description="Copying games...",
                    total=len(events_with_scores),
                )

                for event in events_with_scores:
                    try:
                        await self._copy_event(event, prod_reader, progress, task)
                    except Exception as e:
                        self.stats["errors"] += 1
                        console.print(
                            f"[red]Error copying event {event.id} ({event.away_team} @ {event.home_team}): {e}[/red]"
                        )
                        logger.error("copy_event_failed", event_id=event.id, error=str(e))

                    progress.advance(task)

        # Print summary
        self._print_summary()

        return self.stats

    async def _copy_event(
        self,
        event: Event,
        prod_reader: OddsReader,
        progress: Progress,
        task_id,
    ) -> None:
        """
        Copy a single event and its snapshots from prod to local.

        Args:
            event: Event to copy
            prod_reader: Reader for production database
            progress: Rich progress bar
            task_id: Progress task ID
        """
        game_desc = f"{event.away_team} @ {event.home_team}"
        progress.update(task_id, description=f"Copying: {game_desc[:50]}...")

        async with self.local_session_maker() as local_session:
            local_reader = OddsReader(local_session)
            local_writer = OddsWriter(local_session)

            # Check if event already exists locally
            existing_event = await local_reader.get_event_by_id(event.id)

            if existing_event and self.skip_existing:
                self.stats["events_skipped"] += 1
                logger.debug("skipping_existing_event", event_id=event.id)
                return

            if self.dry_run:
                # In dry run, just count what would be copied
                snapshots = await prod_reader.get_snapshots_for_event(event.id)
                self.stats["events_copied"] += 1
                self.stats["snapshots_copied"] += len(snapshots)
                logger.info(
                    "dry_run_would_copy",
                    event_id=event.id,
                    snapshot_count=len(snapshots),
                )
                return

            # Copy the event - use model_dump() to create a new detached instance
            # This avoids session conflicts and automatically handles all fields
            event_dict = event.model_dump()
            new_event = Event(**event_dict)
            await local_writer.upsert_event(new_event)
            self.stats["events_copied"] += 1

            # Get all snapshots for this event from production
            async with self.prod_session_maker() as prod_session_snapshot:
                prod_reader_snapshot = OddsReader(prod_session_snapshot)
                snapshots = await prod_reader_snapshot.get_snapshots_for_event(event.id)

                # Copy each snapshot
                for snapshot in snapshots:
                    # Check if snapshot already exists
                    snapshot_exists = await local_reader.snapshot_exists(
                        event.id,
                        snapshot.snapshot_time,
                        tolerance_minutes=1,  # Strict tolerance
                    )

                    if not snapshot_exists:
                        # Store snapshot with raw data
                        # The writer will handle creating normalized Odds records
                        # and calculating tier/hours_until_commence automatically
                        await local_writer.store_odds_snapshot(
                            event_id=event.id,
                            raw_data=snapshot.raw_data,
                            snapshot_time=snapshot.snapshot_time,
                        )
                        self.stats["snapshots_copied"] += 1

                await local_session.commit()

                logger.info(
                    "event_copied",
                    event_id=event.id,
                    snapshot_count=len(snapshots),
                )

    def _print_summary(self):
        """Print summary statistics."""
        console.print("\n[bold]Copy Summary:[/bold]")
        console.print(f"  Total events found: {self.stats['total_events']}")
        console.print(f"  Events copied: [green]{self.stats['events_copied']}[/green]")
        console.print(f"  Events skipped: [yellow]{self.stats['events_skipped']}[/yellow]")
        console.print(f"  Snapshots copied: [green]{self.stats['snapshots_copied']}[/green]")

        if self.stats["errors"] > 0:
            console.print(f"  Errors: [red]{self.stats['errors']}[/red]")

        if self.dry_run:
            console.print("\n[yellow]DRY RUN - No data was actually copied[/yellow]")
        else:
            console.print("\n[bold green]✓ Copy completed successfully![/bold green]")


async def copy_from_prod(
    start_date: str,
    end_date: str,
    prod_url: str | None = None,
    local_url: str | None = None,
    sport: str = "basketball_nba",
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """
    Copy completed games from production to local database.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        prod_url: Production database URL (defaults to DATABASE_URL or PROD_DATABASE_URL)
        local_url: Local database URL (defaults to LOCAL_DATABASE_URL)
        sport: Sport key (default: basketball_nba)
        dry_run: If True, don't actually write to database
        skip_existing: If True, skip events that already exist locally
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=UTC
    )

    # Get database URLs
    if not prod_url:
        prod_url = os.getenv("PROD_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not prod_url:
            raise ValueError("PROD_DATABASE_URL or DATABASE_URL environment variable required")

    if not local_url:
        local_url = os.getenv("LOCAL_DATABASE_URL")
        if not local_url:
            raise ValueError("LOCAL_DATABASE_URL environment variable required")

    console.print("[bold]Database Configuration:[/bold]")
    console.print(f"  Production: {prod_url.split('@')[-1] if '@' in prod_url else 'configured'}")
    console.print(f"  Local: {local_url.split('@')[-1] if '@' in local_url else 'configured'}")

    # Create session makers for both databases
    prod_engine = create_async_engine(prod_url, echo=False)
    prod_session_maker = async_sessionmaker(prod_engine, expire_on_commit=False)

    local_engine = create_async_engine(local_url, echo=False)
    local_session_maker = async_sessionmaker(local_engine, expire_on_commit=False)

    try:
        # Create copier and execute
        copier = ProductionDataCopier(
            prod_session_maker=prod_session_maker,
            local_session_maker=local_session_maker,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )

        await copier.copy_completed_games(
            start_date=start_dt,
            end_date=end_dt,
            sport_key=sport,
        )

    finally:
        await prod_engine.dispose()
        await local_engine.dispose()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        console.print("Usage: python copy_from_production.py START_DATE END_DATE [--dry-run]")
        console.print("Example: python copy_from_production.py 2025-10-23 2025-11-07")
        sys.exit(1)

    import asyncio

    start = sys.argv[1]
    end = sys.argv[2]
    dry_run = "--dry-run" in sys.argv

    asyncio.run(copy_from_prod(start, end, dry_run=dry_run))
