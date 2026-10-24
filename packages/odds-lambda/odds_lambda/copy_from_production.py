"""
Copy data from production database to local database.

Replicates events, snapshots, and predictions for a given sport and date range.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from odds_core.models import Event, Odds, OddsSnapshot
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
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.readers import OddsReader

logger = structlog.get_logger()
console = Console()


@dataclass
class CopyStats:
    total_events: int = 0
    events_copied: int = 0
    events_skipped: int = 0
    snapshots_copied: int = 0
    odds_copied: int = 0
    predictions_copied: int = 0
    local_twin_matches: int = 0
    failed_event_ids: list[str] = field(default_factory=list)

    @property
    def events_failed(self) -> int:
        return len(self.failed_event_ids)


class ProductionDataCopier:
    """Replaces local events' snapshots, odds and predictions with production's."""

    def __init__(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
        dry_run: bool = False,
        skip_existing: bool = True,
    ) -> None:
        self.prod_session_maker = prod_session_maker
        self.local_session_maker = local_session_maker
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.stats = CopyStats()

    async def copy_events(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str,
    ) -> CopyStats:
        console.print("\n[bold]Copy Parameters:[/bold]")
        console.print(f"  Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"  Sport: {sport_key}")
        console.print(f"  Dry run: {self.dry_run}")
        console.print(f"  Skip existing: {self.skip_existing}")
        console.print()

        console.print("[bold blue]Querying production database...[/bold blue]")
        async with self.prod_session_maker() as prod_session:
            events = await OddsReader(prod_session).get_events_by_date_range(
                start_date=start_date,
                end_date=end_date,
                sport_key=sport_key,
            )

        self.stats.total_events = len(events)
        console.print(f"[green]Found {len(events)} events in production[/green]")

        if not events:
            console.print("[yellow]No events found matching criteria[/yellow]")
            return self.stats

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description="Copying events...", total=len(events))

            for event in events:
                try:
                    await self._copy_event(event, progress, task)
                except Exception as e:
                    self.stats.failed_event_ids.append(event.id)
                    console.print(
                        f"[red]Error copying event {event.id} ({event.away_team} @ {event.home_team}): {e}[/red]"
                    )
                    logger.error("copy_event_failed", event_id=event.id, error=str(e))

                progress.advance(task)

        if not self.dry_run:
            async with self.local_session_maker() as local_session:
                self.stats.local_twin_matches = len(
                    await EventMerger(local_session).find_twin_groups(sport_key)
                )
        self._print_summary()

        return self.stats

    async def _copy_event(
        self,
        event: Event,
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        game_desc = f"{event.away_team} @ {event.home_team}"
        progress.update(task_id, description=f"Copying: {game_desc[:50]}...")

        async with (
            self.local_session_maker() as local_session,
            self.prod_session_maker() as prod_session,
        ):
            if self.skip_existing and await local_session.get(Event, event.id) is not None:
                self.stats.events_skipped += 1
                logger.debug("skipping_existing_event", event_id=event.id)
                return

            prod_snapshots = list(
                (
                    await prod_session.execute(
                        select(OddsSnapshot)
                        .where(OddsSnapshot.event_id == event.id)
                        .order_by(OddsSnapshot.id)
                    )
                )
                .scalars()
                .all()
            )

            if self.dry_run:
                self.stats.events_copied += 1
                self.stats.snapshots_copied += len(prod_snapshots)
                logger.info(
                    "dry_run_would_copy", event_id=event.id, snapshot_count=len(prod_snapshots)
                )
                return

            prod_odds = list(
                (await prod_session.execute(select(Odds).where(Odds.event_id == event.id)))
                .scalars()
                .all()
            )
            prod_predictions = list(
                (
                    await prod_session.execute(
                        select(Prediction).where(Prediction.event_id == event.id)
                    )
                )
                .scalars()
                .all()
            )

            await local_session.execute(delete(Prediction).where(Prediction.event_id == event.id))
            await local_session.execute(delete(Odds).where(Odds.event_id == event.id))
            await local_session.execute(
                delete(OddsSnapshot).where(OddsSnapshot.event_id == event.id)
            )
            await local_session.merge(Event(**event.model_dump()))
            await local_session.flush()

            local_snapshots = [
                OddsSnapshot(**snapshot.model_dump(exclude={"id"})) for snapshot in prod_snapshots
            ]
            local_session.add_all(local_snapshots)
            await local_session.flush()
            snapshot_ids = {
                prod.id: local.id
                for prod, local in zip(prod_snapshots, local_snapshots, strict=True)
                if prod.id is not None and local.id is not None
            }

            if prod_odds:
                await local_session.execute(
                    insert(Odds), [row.model_dump(exclude={"id"}) for row in prod_odds]
                )

            prediction_rows = [
                {
                    **pred.model_dump(exclude={"id"}),
                    "snapshot_id": self._local_snapshot_id(pred, snapshot_ids),
                }
                for pred in prod_predictions
            ]
            if prediction_rows:
                await local_session.execute(insert(Prediction), prediction_rows)

            await local_session.commit()

        self.stats.events_copied += 1
        self.stats.snapshots_copied += len(local_snapshots)
        self.stats.odds_copied += len(prod_odds)
        self.stats.predictions_copied += len(prediction_rows)
        logger.info("event_copied", event_id=event.id, snapshot_count=len(local_snapshots))

    @staticmethod
    def _local_snapshot_id(prediction: Prediction, snapshot_ids: dict[int, int]) -> int:
        if prediction.snapshot_id not in snapshot_ids:
            raise ValueError(
                f"prediction {prediction.id} of event {prediction.event_id} references "
                f"production snapshot {prediction.snapshot_id}, which belongs to another event; "
                "the source database has not run the migration that forbids this (#531)"
            )
        return snapshot_ids[prediction.snapshot_id]

    def _print_summary(self) -> None:
        """Print summary statistics."""
        console.print("\n[bold]Copy Summary:[/bold]")
        console.print(f"  Total events found: {self.stats.total_events}")
        console.print(f"  Events copied: [green]{self.stats.events_copied}[/green]")
        console.print(f"  Events skipped: [yellow]{self.stats.events_skipped}[/yellow]")
        console.print(f"  Snapshots copied: [green]{self.stats.snapshots_copied}[/green]")

        console.print(f"  Odds rows copied: [green]{self.stats.odds_copied}[/green]")
        console.print(f"  Predictions copied: [green]{self.stats.predictions_copied}[/green]")
        if self.stats.local_twin_matches:
            console.print(
                f"  Local matches stored as several events: "
                f"[yellow]{self.stats.local_twin_matches}[/yellow] "
                "(run `odds events merge-twins --apply`)"
            )

        if self.stats.failed_event_ids:
            console.print(f"  Events failed: [red]{self.stats.events_failed}[/red]")
            console.print(f"  Failed event ids: {', '.join(self.stats.failed_event_ids)}")

        if self.stats.failed_event_ids:
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
) -> CopyStats:
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
        prod_url = os.getenv("PROD_DATABASE_URL")
        if not prod_url:
            raise ValueError("PROD_DATABASE_URL environment variable or --prod-url required")

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
