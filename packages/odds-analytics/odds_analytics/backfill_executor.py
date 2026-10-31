"""Backfill execution logic for historical data collection."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog
from odds_core.database import async_session_maker
from odds_core.time import parse_api_datetime
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.live_events import DroppedGame, LiveEventFinder, alert_dropped_games
from odds_lambda.odds_api_events import (
    odds_api_finder,
    resolve_odds_api_event,
)
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()


@dataclass
class BackfillResult:
    """Results from a backfill operation."""

    successful_games: int
    successful_snapshots: int
    failed_snapshots: int
    skipped_snapshots: int
    total_quota_used: int
    dropped_games: list[DroppedGame] = field(default_factory=list)


@dataclass
class BackfillProgress:
    """Progress update during backfill."""

    event_id: int
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
        sport: str,
        client: TheOddsAPIClient | None = None,
        session_factory: Callable | None = None,
        skip_existing: bool = True,
        dry_run: bool = False,
        rate_limit_seconds: float = 1.0,
    ):
        """
        Initialize backfill executor.

        Args:
            sport: Sport whose games are backfilled; its key pays for the calls
            client: API client (will create one if not provided)
            session_factory: Factory function to create database sessions
                           (defaults to async_session_maker)
            skip_existing: Whether to skip snapshots that already exist
            dry_run: If True, simulate execution without API calls
            rate_limit_seconds: Seconds to wait between API requests
        """
        self.sport = sport
        self._client = client
        self._owns_client = client is None
        self._session_factory = session_factory or async_session_maker
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.rate_limit_seconds = rate_limit_seconds
        # Not MLB, so the finder reads no MLB schedule and one serves the whole run
        self._finder: LiveEventFinder | None = None
        self._dropped_games: list[DroppedGame] = []

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
            odds_api_id = game.get("odds_api_id")
            home_team = game["home_team"]
            away_team = game["away_team"]
            snapshots = game["snapshots"]

            game_success = True

            # Process each snapshot for this game
            for snapshot_time in snapshots:
                try:
                    result = await self._process_snapshot(
                        event_id=event_id,
                        odds_api_id=odds_api_id,
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

        await alert_dropped_games("odds-backfill", self._dropped_games)

        return BackfillResult(
            dropped_games=list(self._dropped_games),
            successful_games=successful_games,
            successful_snapshots=successful_snapshots,
            failed_snapshots=failed_snapshots,
            skipped_snapshots=skipped_snapshots,
            total_quota_used=total_quota_used,
        )

    async def _process_snapshot(
        self,
        event_id: int,
        odds_api_id: str | None,
        home_team: str,
        away_team: str,
        snapshot_time: str,
    ) -> BackfillProgress:
        """
        Process a single snapshot.

        Args:
            event_id: Event identifier
            odds_api_id: The id The Odds API gives the event
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

        if odds_api_id is None:
            return BackfillProgress(
                event_id=event_id,
                home_team=home_team,
                away_team=away_team,
                snapshot_time=snapshot_time,
                status="skipped",
                message="Event has no Odds API id",
            )

        # Fetch historical odds - API client returns parsed Event instances
        client = self._client
        if client is None:
            raise RuntimeError("BackfillExecutor client not initialized")

        response = await client.get_historical_odds(
            sport=self.sport,
            date=snapshot_time,
        )

        # Find our specific event in the response
        event = None
        event_data = None
        for evt, raw_evt in zip(response.events, response.raw_events_data, strict=True):
            if evt.odds_api_id == odds_api_id:
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

        # Store to database
        snapshot_dt = parse_api_datetime(snapshot_time)

        async with self._session_factory() as session:
            writer = OddsWriter(session)

            if self._finder is None:
                self._finder = await odds_api_finder(event.sport_key, [event])
            found = await resolve_odds_api_event(self._finder, session, event)
            if isinstance(found, DroppedGame):
                self._dropped_games.append(found)
                return BackfillProgress(
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    snapshot_time=snapshot_time,
                    status="skipped",
                    message=f"Game dropped: {found.describe()}",
                )
            payload_id = await store_payload(session, [event_data])
            await writer.store_envelope(
                event_id=found.event.id,
                envelope=event_data,
                snapshot_time=snapshot_dt,
                shape=SnapshotShape(Source.ODDS_API, TimeBasis.UPPER_BOUND, payload_id),
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
