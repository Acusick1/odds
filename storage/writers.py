"""Database write operations for odds data."""

from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot
from storage.validators import OddsValidator

logger = structlog.get_logger()


class OddsWriter:
    """Handles all write operations to the database."""

    def __init__(self, session: AsyncSession):
        """
        Initialize writer with database session.

        Args:
            session: Async database session
        """
        self.session = session
        self.validator = OddsValidator()

    async def upsert_event(self, event_data: dict) -> Event:
        """
        Insert or update an event.

        Args:
            event_data: Event data from API

        Returns:
            Created or updated Event instance

        Example:
            event = await writer.upsert_event({
                "id": "abc123",
                "sport_key": "basketball_nba",
                "commence_time": "2024-10-20T00:00:00Z",
                "home_team": "Lakers",
                "away_team": "Celtics"
            })
        """
        event_id = event_data["id"]

        # Check if event exists
        result = await self.session.execute(select(Event).where(Event.id == event_id))
        existing_event = result.scalar_one_or_none()

        if existing_event:
            # Update existing event
            existing_event.sport_key = event_data.get("sport_key", existing_event.sport_key)
            existing_event.sport_title = event_data.get("sport_title", existing_event.sport_title)
            existing_event.home_team = event_data.get("home_team", existing_event.home_team)
            existing_event.away_team = event_data.get("away_team", existing_event.away_team)

            # Parse commence_time
            if "commence_time" in event_data:
                commence_time_str = event_data["commence_time"].replace("Z", "+00:00")
                existing_event.commence_time = datetime.fromisoformat(commence_time_str).replace(
                    tzinfo=None
                )

            existing_event.updated_at = datetime.utcnow()

            logger.info("event_updated", event_id=event_id)
            return existing_event
        else:
            # Create new event
            commence_time_str = event_data["commence_time"].replace("Z", "+00:00")
            commence_time = datetime.fromisoformat(commence_time_str).replace(tzinfo=None)

            event = Event(
                id=event_id,
                sport_key=event_data["sport_key"],
                sport_title=event_data.get("sport_title", event_data["sport_key"]),
                commence_time=commence_time,
                home_team=event_data["home_team"],
                away_team=event_data["away_team"],
                status=EventStatus.SCHEDULED,
            )

            self.session.add(event)
            logger.info("event_created", event_id=event_id)
            return event

    async def store_odds_snapshot(
        self,
        event_id: str,
        raw_data: dict,
        snapshot_time: datetime | None = None,
        validate: bool = True,
    ) -> tuple[OddsSnapshot, list[Odds]]:
        """
        Store odds snapshot with hybrid storage approach.

        Args:
            event_id: Event identifier
            raw_data: Complete API response for the event
            snapshot_time: Time of snapshot (defaults to now)
            validate: Whether to run validation checks

        Returns:
            Tuple of (OddsSnapshot, list of normalized Odds records)

        Storage:
            - Raw JSONB snapshot for debugging
            - Normalized odds records for querying
        """
        snapshot_time = snapshot_time or datetime.utcnow()

        # Validate if enabled
        if validate:
            is_valid, warnings = self.validator.validate_odds_snapshot(raw_data, event_id)
            if warnings:
                await self.log_data_quality_issue(
                    event_id=event_id,
                    severity="warning" if is_valid else "error",
                    issue_type="validation_warnings",
                    description=f"Validation found {len(warnings)} issues",
                    raw_data={"warnings": warnings[:10]},  # Store first 10
                )

        # Store raw snapshot
        bookmakers = raw_data.get("bookmakers", [])
        if isinstance(raw_data, list):
            # Find specific event in list
            event_data = next((e for e in raw_data if e.get("id") == event_id), {})
            bookmakers = event_data.get("bookmakers", [])

        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=len(bookmakers),
        )
        self.session.add(snapshot)

        # Parse and store normalized odds
        odds_records = await self._parse_and_store_odds(
            event_id=event_id,
            bookmakers=bookmakers,
            snapshot_time=snapshot_time,
        )

        logger.info(
            "odds_snapshot_stored",
            event_id=event_id,
            bookmakers=len(bookmakers),
            odds_records=len(odds_records),
        )

        return snapshot, odds_records

    async def _parse_and_store_odds(
        self,
        event_id: str,
        bookmakers: list[dict],
        snapshot_time: datetime,
    ) -> list[Odds]:
        """
        Parse bookmaker data and create normalized odds records.

        Args:
            event_id: Event identifier
            bookmakers: List of bookmaker data from API
            snapshot_time: Time of snapshot

        Returns:
            List of created Odds records
        """
        odds_records = []

        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title", bookmaker_key)

            # Parse last_update timestamp
            last_update_str = bookmaker.get("last_update")
            try:
                last_update = datetime.fromisoformat(
                    last_update_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                last_update = snapshot_time

            # Process each market
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                # Process each outcome
                for outcome in market.get("outcomes", []):
                    odds = Odds(
                        event_id=event_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome.get("name"),
                        price=int(outcome.get("price", 0)),
                        point=outcome.get("point"),  # Will be None for h2h
                        odds_timestamp=snapshot_time,
                        last_update=last_update,
                        is_valid=True,
                    )
                    self.session.add(odds)
                    odds_records.append(odds)

        return odds_records

    async def update_event_status(
        self,
        event_id: str,
        status: EventStatus,
        home_score: int | None = None,
        away_score: int | None = None,
    ) -> Event | None:
        """
        Update event status and scores.

        Args:
            event_id: Event identifier
            status: New event status
            home_score: Final home team score (if completed)
            away_score: Final away team score (if completed)

        Returns:
            Updated Event or None if not found
        """
        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()

        if not event:
            logger.warning("event_not_found", event_id=event_id)
            return None

        event.status = status
        event.updated_at = datetime.utcnow()

        if status == EventStatus.FINAL and home_score is not None and away_score is not None:
            event.home_score = home_score
            event.away_score = away_score
            event.completed_at = datetime.utcnow()

        logger.info(
            "event_status_updated",
            event_id=event_id,
            status=status.value,
            home_score=home_score,
            away_score=away_score,
        )

        return event

    async def log_fetch(
        self,
        sport_key: str,
        events_count: int,
        bookmakers_count: int,
        success: bool,
        error_message: str | None = None,
        api_quota_remaining: int | None = None,
        response_time_ms: int | None = None,
    ) -> FetchLog:
        """
        Log an API fetch operation.

        Args:
            sport_key: Sport that was fetched
            events_count: Number of events received
            bookmakers_count: Number of bookmakers in response
            success: Whether fetch succeeded
            error_message: Error message if failed
            api_quota_remaining: Remaining API quota
            response_time_ms: Response time in milliseconds

        Returns:
            Created FetchLog record
        """
        fetch_log = FetchLog(
            sport_key=sport_key,
            events_count=events_count,
            bookmakers_count=bookmakers_count,
            success=success,
            error_message=error_message,
            api_quota_remaining=api_quota_remaining,
            response_time_ms=response_time_ms,
        )

        self.session.add(fetch_log)

        logger.info(
            "fetch_logged",
            sport_key=sport_key,
            events_count=events_count,
            success=success,
            quota_remaining=api_quota_remaining,
        )

        return fetch_log

    async def log_data_quality_issue(
        self,
        severity: str,
        issue_type: str,
        description: str,
        event_id: str | None = None,
        raw_data: dict | None = None,
    ) -> DataQualityLog:
        """
        Log a data quality issue.

        Args:
            severity: Severity level: warning, error, critical
            issue_type: Issue type: missing_data, suspicious_odds, etc.
            description: Human-readable description
            event_id: Related event ID (optional)
            raw_data: Context data (optional)

        Returns:
            Created DataQualityLog record
        """
        quality_log = DataQualityLog(
            event_id=event_id,
            severity=severity,
            issue_type=issue_type,
            description=description,
            raw_data=raw_data,
        )

        self.session.add(quality_log)

        logger.warning(
            "data_quality_issue",
            event_id=event_id,
            severity=severity,
            issue_type=issue_type,
            description=description,
        )

        return quality_log

    async def bulk_insert_odds(self, odds_records: list[dict]) -> int:
        """
        Bulk insert odds records for efficient backfill.

        Args:
            odds_records: List of odds dictionaries

        Returns:
            Number of records inserted

        Note:
            Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING for efficiency
        """
        if not odds_records:
            return 0

        stmt = insert(Odds).values(odds_records)
        stmt = stmt.on_conflict_do_nothing()

        result = await self.session.execute(stmt)
        count = result.rowcount

        logger.info("bulk_odds_inserted", count=count)
        return count
