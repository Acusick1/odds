"""Backfill execution logic for historical data collection."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from core.data_fetcher import TheOddsAPIClient
from core.database import async_session_maker
from core.time import parse_api_datetime
from storage.readers import OddsReader
from storage.writers import OddsWriter

logger = structlog.get_logger()


@dataclass
class BackfillResult:
    """Results from a backfill operation."""

    successful_games: int
    successful_snapshots: int
    failed_snapshots: int
    skipped_snapshots: int
    total_quota_used: int


@dataclass
class BackfillProgress:
    """Progress update during backfill."""

    event_id: str
    home_team: str
    away_team: str
    snapshot_time: str
    status: str  # 'success', 'failed', 'skipped', 'exists'
    message: str
    quota_remaining: int | None = None


class BackfillExecutor:
    """Executes historical data backfill operations."""

    def __init__(
        self,
        client: TheOddsAPIClient | None = None,
        session_factory: Callable | None = None,
        skip_existing: bool = True,
        dry_run: bool = False,
        rate_limit_seconds: float = 1.0,
    ):
        """
        Initialize backfill executor.

        Args:
            client: API client (will create one if not provided)
            session_factory: Factory function to create database sessions
                           (defaults to async_session_maker)
            skip_existing: Whether to skip snapshots that already exist
            dry_run: If True, simulate execution without API calls
            rate_limit_seconds: Seconds to wait between API requests
        """
        self._client = client
        self._owns_client = client is None
        self._session_factory = session_factory or async_session_maker
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.rate_limit_seconds = rate_limit_seconds

    async def __aenter__(self):
        """Context manager entry."""
        if self._owns_client:
            self._client = TheOddsAPIClient()
            await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._owns_client and self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def execute_plan(
        self,
        plan: dict,
        progress_callback: Callable[[BackfillProgress], None] | None = None,
    ) -> BackfillResult:
        """
        Execute a backfill plan.

        Args:
            plan: Backfill plan dictionary with 'games' list
            progress_callback: Optional callback for progress updates
                               Called with BackfillProgress objects

        Returns:
            BackfillResult with execution statistics

        Raises:
            ValueError: If plan is invalid
        """
        # Validate plan
        games = plan.get("games", [])
        if not games:
            raise ValueError("Plan contains no games")

        # Initialize statistics
        successful_games = 0
        successful_snapshots = 0
        failed_snapshots = 0
        skipped_snapshots = 0
        total_quota_used = 0

        # Process each game
        for game in games:
            event_id = game["event_id"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            snapshots = game["snapshots"]

            game_success = True

            # Process each snapshot for this game
            for snapshot_time in snapshots:
                try:
                    result = await self._process_snapshot(
                        event_id=event_id,
                        home_team=home_team,
                        away_team=away_team,
                        snapshot_time=snapshot_time,
                    )

                    # Update statistics
                    if result.status == "success":
                        successful_snapshots += 1
                        total_quota_used += 30  # Approximate quota cost
                    elif result.status == "exists":
                        skipped_snapshots += 1
                    elif result.status == "skipped":
                        # In dry run, skipped means "would fetch" - count as success
                        if self.dry_run:
                            successful_snapshots += 1
                        else:
                            skipped_snapshots += 1
                    elif result.status == "failed":
                        failed_snapshots += 1
                        game_success = False

                    # Call progress callback
                    if progress_callback:
                        progress_callback(result)

                    # Rate limiting
                    if not self.dry_run and result.status == "success":
                        await asyncio.sleep(self.rate_limit_seconds)

                except Exception as e:
                    failed_snapshots += 1
                    game_success = False

                    logger.error(
                        "snapshot_processing_failed",
                        event_id=event_id,
                        snapshot_time=snapshot_time,
                        error=str(e),
                    )

                    if progress_callback:
                        progress_callback(
                            BackfillProgress(
                                event_id=event_id,
                                home_team=home_team,
                                away_team=away_team,
                                snapshot_time=snapshot_time,
                                status="failed",
                                message=str(e),
                            )
                        )

            if game_success:
                successful_games += 1

        return BackfillResult(
            successful_games=successful_games,
            successful_snapshots=successful_snapshots,
            failed_snapshots=failed_snapshots,
            skipped_snapshots=skipped_snapshots,
            total_quota_used=total_quota_used,
        )

    async def _process_snapshot(
        self,
        event_id: str,
        home_team: str,
        away_team: str,
        snapshot_time: str,
    ) -> BackfillProgress:
        """
        Process a single snapshot.

        Args:
            event_id: Event identifier
            home_team: Home team name
            away_team: Away team name
            snapshot_time: ISO format snapshot time

        Returns:
            BackfillProgress with result
        """
        # Dry run mode
        if self.dry_run:
            return BackfillProgress(
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                snapshot_time=snapshot_time,
                status="skipped",
                message="Dry run - would fetch",
            )

        # Check if snapshot already exists
        if self.skip_existing:
            snapshot_dt = parse_api_datetime(snapshot_time)

            async with self._session_factory() as check_session:
                reader = OddsReader(check_session)
                exists = await reader.snapshot_exists(event_id, snapshot_dt)

                if exists:
                    return BackfillProgress(
                        event_id=event_id,
                        home_team=home_team,
                        away_team=away_team,
                        snapshot_time=snapshot_time,
                        status="exists",
                        message="Already exists",
                    )

        # Fetch historical odds - API client returns parsed Event instances
        client = self._client
        if client is None:
            raise RuntimeError("BackfillExecutor client not initialized")

        response = await client.get_historical_odds(
            sport="basketball_nba",
            date=snapshot_time,
        )

        # Find our specific event in the response
        event = None
        event_data = None
        for evt, raw_evt in zip(response.events, response.raw_events_data, strict=True):
            if evt.id == event_id:
                event = evt
                event_data = raw_evt
                break

        if not event or event_data is None:
            return BackfillProgress(
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                snapshot_time=snapshot_time,
                status="skipped",
                message="Event not found in API response",
            )

        # Store to database - keep timezone-aware for tier calculation
        snapshot_dt = parse_api_datetime(snapshot_time)

        async with self._session_factory() as session:
            writer = OddsWriter(session)

            # Upsert event - already parsed by API client
            await writer.upsert_event(event)
            await session.flush()
            await writer.store_odds_snapshot(
                event_id=event.id,
                raw_data=event_data,
                snapshot_time=snapshot_dt,
            )
            await session.commit()

        quota_remaining = response.quota_remaining or 0

        logger.info(
            "snapshot_stored",
            event_id=event_id,
            snapshot_time=snapshot_time,
            quota_remaining=quota_remaining,
        )

        return BackfillProgress(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            snapshot_time=snapshot_time,
            status="success",
            message="Stored successfully",
            quota_remaining=quota_remaining,
        )
