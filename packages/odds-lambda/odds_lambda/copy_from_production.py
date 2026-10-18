"""
Copy data from production database to local database.

Replicates events, snapshots, and predictions for a given sport and date range.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import structlog
from odds_core.models import Event, OddsSnapshot
from odds_core.prediction_models import Prediction
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()
console = Console()


class ProductionDataCopier:
    """Copies events, snapshots, and predictions from production to local."""

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
            "predictions_copied": 0,
            "events_failed": 0,
        }
        self.failed_event_ids: list[str] = []

    async def copy_events(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str,
    ) -> dict:
        """
        Copy all events, snapshots, and predictions from production to local.

        Args:
            start_date: Start of date range
            end_date: End of date range
            sport_key: Sport to copy

        Returns:
            Dictionary with copy statistics
        """
        console.print("\n[bold]Copy Parameters:[/bold]")
        console.print(f"  Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"  Sport: {sport_key}")
        console.print(f"  Dry run: {self.dry_run}")
        console.print(f"  Skip existing: {self.skip_existing}")
        console.print()

        async with self.prod_session_maker() as prod_session:
            prod_reader = OddsReader(prod_session)

            console.print("[bold blue]Querying production database...[/bold blue]")
            events = await prod_reader.get_events_by_date_range(
                start_date=start_date,
                end_date=end_date,
                sport_key=sport_key,
            )

            self.stats["total_events"] = len(events)

            console.print(f"[green]Found {len(events)} events in production[/green]")

            if not events:
                console.print("[yellow]No events found matching criteria[/yellow]")
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
                    description="Copying events...",
                    total=len(events),
                )

                for event in events:
                    try:
                        await self._copy_event(event, prod_reader, progress, task)
                    except Exception as e:
                        self.stats["events_failed"] += 1
                        self.failed_event_ids.append(event.id)
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
        task_id: TaskID,
    ) -> None:
        game_desc = f"{event.away_team} @ {event.home_team}"
        progress.update(task_id, description=f"Copying: {game_desc[:50]}...")

        async with self.local_session_maker() as local_session:
            if self.skip_existing and await local_session.get(Event, event.id) is not None:
                self.stats["events_skipped"] += 1
                logger.debug("skipping_existing_event", event_id=event.id)
                return

            prod_snapshots = await prod_reader.get_snapshots_for_event(event.id)

            if self.dry_run:
                self.stats["events_copied"] += 1
                self.stats["snapshots_copied"] += len(prod_snapshots)
                logger.info(
                    "dry_run_would_copy",
                    event_id=event.id,
                    snapshot_count=len(prod_snapshots),
                )
                return

            await local_session.merge(Event(**event.model_dump()))
            await local_session.flush()

            snapshot_ids, snapshots_copied = await self._copy_snapshots(
                event.id, prod_snapshots, local_session
            )
            predictions_copied = await self._copy_predictions(
                event.id, snapshot_ids, prod_reader.session, local_session
            )
            await local_session.commit()

        self.stats["events_copied"] += 1
        self.stats["snapshots_copied"] += snapshots_copied
        self.stats["predictions_copied"] += predictions_copied
        logger.info("event_copied", event_id=event.id, snapshot_count=len(prod_snapshots))

    async def _copy_snapshots(
        self,
        event_id: str,
        prod_snapshots: list[OddsSnapshot],
        local_session: AsyncSession,
    ) -> tuple[dict[int, int], int]:
        """Copy snapshots with every column preserved.

        Returns:
            Map from production snapshot id to local snapshot id, and the number
            of snapshots newly inserted.
        """
        local_writer = OddsWriter(local_session)
        local_snapshots = (
            (
                await local_session.execute(
                    select(OddsSnapshot).where(OddsSnapshot.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        local_by_key = {(s.snapshot_time, s.api_request_id): s.id for s in local_snapshots}

        snapshot_ids: dict[int, int] = {}
        inserted = 0
        for prod_snapshot in prod_snapshots:
            key = (prod_snapshot.snapshot_time, prod_snapshot.api_request_id)
            local_id = local_by_key.get(key)
            if local_id is None:
                snapshot, _ = await local_writer.store_odds_snapshot(
                    event_id=event_id,
                    raw_data=prod_snapshot.raw_data,
                    snapshot_time=prod_snapshot.snapshot_time,
                    validate=False,
                )
                snapshot.api_request_id = prod_snapshot.api_request_id
                snapshot.fetch_tier = prod_snapshot.fetch_tier
                snapshot.hours_until_commence = prod_snapshot.hours_until_commence
                snapshot.created_at = prod_snapshot.created_at
                await local_session.flush()
                local_id = snapshot.id
                inserted += 1
            assert prod_snapshot.id is not None and local_id is not None
            snapshot_ids[prod_snapshot.id] = local_id
        return snapshot_ids, inserted

    async def _copy_predictions(
        self,
        event_id: str,
        snapshot_ids: dict[int, int],
        prod_session: AsyncSession,
        local_session: AsyncSession,
    ) -> int:
        """Copy predictions, re-pointing each at the local copy of its snapshot.

        Raises:
            KeyError: A prediction references a snapshot that was not copied.
        """
        result = await prod_session.execute(
            select(Prediction).where(Prediction.event_id == event_id)
        )
        rows = [
            {
                "event_id": pred.event_id,
                "snapshot_id": snapshot_ids[pred.snapshot_id],
                "model_name": pred.model_name,
                "model_version": pred.model_version,
                "predicted_clv": pred.predicted_clv,
                "created_at": pred.created_at,
            }
            for pred in result.scalars().all()
        ]
        if not rows:
            return 0

        inserted = await local_session.execute(
            pg_insert(Prediction)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_prediction_event_snap_model")
            .returning(Prediction.id)
        )
        return len(inserted.all())

    def _print_summary(self) -> None:
        """Print summary statistics."""
        console.print("\n[bold]Copy Summary:[/bold]")
        console.print(f"  Total events found: {self.stats['total_events']}")
        console.print(f"  Events copied: [green]{self.stats['events_copied']}[/green]")
        console.print(f"  Events skipped: [yellow]{self.stats['events_skipped']}[/yellow]")
        console.print(f"  Snapshots copied: [green]{self.stats['snapshots_copied']}[/green]")

        console.print(f"  Predictions copied: [green]{self.stats['predictions_copied']}[/green]")

        if self.failed_event_ids:
            console.print(f"  Events failed: [red]{self.stats['events_failed']}[/red]")
            console.print(f"  Failed event ids: {', '.join(self.failed_event_ids)}")

        if self.failed_event_ids:
            console.print("\n[bold red]✗ Copy finished with failed events[/bold red]")
        elif self.dry_run:
            console.print("\n[yellow]DRY RUN - No data was actually copied[/yellow]")
        else:
            console.print("\n[bold green]✓ Copy completed successfully![/bold green]")


async def copy_from_prod(
    start_date: str,
    end_date: str,
    prod_url: str | None = None,
    local_url: str | None = None,
    sport: str = "soccer_epl",
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Copy all events, snapshots, and predictions from production to local database.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        prod_url: Production database URL (defaults to PROD_DATABASE_URL)
        local_url: Local database URL (defaults to LOCAL_DATABASE_URL)
        sport: Sport key (default: soccer_epl)
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

        return await copier.copy_events(
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
