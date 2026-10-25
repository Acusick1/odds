"""Database write operations for odds data."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot
from odds_core.odds_math import outcome_price_decimal
from odds_core.time import ensure_utc, parse_api_datetime
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.event_kickoff import move_kickoff
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from odds_lambda.storage.validators import OddsValidator
from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier_from_timestamps

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

    async def upsert_event(self, event: Event, now: datetime | None = None) -> Event:
        """Insert an Odds API event, or update the stored one's sport and teams.

        The stored ``commence_time`` moves only through ``move_kickoff`` on the Odds API's
        authority, against the clock ``now`` (defaults to the current time).
        """
        result = await self.session.execute(select(Event).where(Event.id == event.id))
        existing_event = result.scalar_one_or_none()

        if existing_event:
            existing_event.sport_key = event.sport_key
            existing_event.sport_title = event.sport_title
            existing_event.home_team = event.home_team
            existing_event.away_team = event.away_team
            existing_event.updated_at = datetime.now(UTC)
            await move_kickoff(
                self.session,
                existing_event,
                Source.ODDS_API,
                event.commence_time,
                now=now or datetime.now(UTC),
            )

            logger.info("event_updated", event_id=event.id)
            return existing_event
        else:
            self.session.add(event)
            logger.info("event_created", event_id=event.id)
            return event

    async def store_odds_snapshot(
        self,
        event_id: str,
        raw_data: dict,
        snapshot_time: datetime | None = None,
        validate: bool = True,
        *,
        shape: SnapshotShape | None = None,
        api_request_id: str | None = None,
    ) -> tuple[OddsSnapshot, list[Odds]]:
        """Store an envelope as a snapshot with its ``odds`` rows.

        With ``shape``, the snapshot also takes its new-shape columns and price rows, and the
        envelope's other markets become their own snapshots
        (``odds_lambda.storage.shaped_snapshots.store_shaped_snapshot``). A snapshot whose key
        (event, source, market, time) is already stored is not written again, and neither are
        its ``odds`` rows; the stored snapshot is returned.

        Args:
            event_id: Event identifier
            raw_data: The envelope: what the source returned for the event, as stored
            snapshot_time: Time of snapshot (defaults to now)
            validate: Whether to run validation checks
            shape: The source, time basis and payload of the snapshot
            api_request_id: The fetch's request id

        Returns:
            Tuple of (OddsSnapshot, list of normalized Odds records)
        """
        snapshot_time = ensure_utc(snapshot_time or datetime.now(UTC))

        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()

        # Always calculate tier and hours_until_commence per-event from timestamps
        tier_value = None
        hours_until = None
        if event:
            event_commence = ensure_utc(event.commence_time)
            hours_until = calculate_hours_until_commence(snapshot_time, event_commence)
            calculated_tier = calculate_tier_from_timestamps(snapshot_time, event_commence)
            tier_value = calculated_tier.value

        # Validate if enabled
        if validate:
            is_valid, warnings = self.validator.validate_odds_snapshot(raw_data, event_id)
            if warnings:
                quality_log = DataQualityLog(
                    event_id=event_id,
                    severity="warning" if is_valid else "error",
                    issue_type="validation_warnings",
                    description=f"Validation found {len(warnings)} issues",
                    raw_data={"warnings": warnings[:10]},  # Store first 10
                )
                await self.log_data_quality_issue(quality_log)

        bookmakers = raw_data.get("bookmakers", [])
        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=len(bookmakers),
            api_request_id=api_request_id,
            fetch_tier=tier_value,
            hours_until_commence=hours_until,
        )
        if shape is None:
            self.session.add(snapshot)
            await self.session.flush()
        else:
            if event is None:
                raise ValueError(f"event {event_id} is not stored")
            written = await store_shaped_snapshot(self.session, snapshot, event, shape)
            if not written.inserted:
                logger.info(
                    "odds_snapshot_already_stored",
                    event_id=event_id,
                    snapshot_id=written.snapshot.id,
                    outcome=written.first.outcome if written.first else None,
                )
                return written.snapshot, []
            snapshot = written.snapshot
        assert snapshot.id is not None

        # Parse and store normalized odds
        odds_records = await self._parse_and_store_odds(
            event_id=event_id,
            snapshot_id=snapshot.id,
            bookmakers=bookmakers,
            snapshot_time=snapshot_time,
        )

        logger.info(
            "odds_snapshot_stored",
            event_id=event_id,
            bookmakers=len(bookmakers),
            odds_records=len(odds_records),
            fetch_tier=tier_value,
            hours_until_commence=hours_until,
        )

        return snapshot, odds_records

    async def _parse_and_store_odds(
        self,
        event_id: str,
        snapshot_id: int,
        bookmakers: list[dict],
        snapshot_time: datetime,
    ) -> list[Odds]:
        """
        Parse bookmaker data and create normalized odds records.

        Args:
            event_id: Event identifier
            snapshot_id: Snapshot the bookmaker data came from
            bookmakers: List of bookmaker data from API
            snapshot_time: Time of snapshot

        Returns:
            List of created Odds records
        """
        odds_records = []

        for bookmaker in bookmakers:
            bookmaker_key = str(bookmaker.get("key") or "").strip()
            if not bookmaker_key:
                continue  # Skip malformed bookmaker entries

            bookmaker_title = str(bookmaker.get("title") or bookmaker_key)

            # Parse last_update timestamp
            last_update = snapshot_time
            last_update_str = bookmaker.get("last_update")
            if isinstance(last_update_str, str):
                try:
                    last_update = parse_api_datetime(last_update_str)
                except Exception:
                    last_update = snapshot_time

            # Process each market
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                # Process each outcome
                for outcome in market.get("outcomes", []):
                    odds = Odds(
                        event_id=event_id,
                        snapshot_id=snapshot_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome.get("name"),
                        price_decimal=outcome_price_decimal(outcome),
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

        Raises:
            ValueError: If ``status`` is FINAL and either score is missing, or the stored
                status cannot become ``status`` (``EventStatus.can_become``)
        """
        if status == EventStatus.FINAL and (home_score is None or away_score is None):
            raise ValueError(f"event {event_id} cannot be FINAL without both scores")

        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()

        if not event:
            logger.warning("event_not_found", event_id=event_id)
            return None

        if not event.status.can_become(status):
            raise ValueError(
                f"event {event_id} cannot move from {event.status.value} to {status.value}"
            )

        event.status = status
        event.updated_at = datetime.now(UTC)

        if status == EventStatus.FINAL:
            event.home_score = home_score
            event.away_score = away_score
            event.completed_at = datetime.now(UTC)

        logger.info(
            "event_status_updated",
            event_id=event_id,
            status=status.value,
            home_score=home_score,
            away_score=away_score,
        )

        return event

    async def log_fetch(self, fetch_log: FetchLog) -> FetchLog:
        """
        Log an API fetch operation.

        Args:
            fetch_log: FetchLog instance to persist

        Returns:
            Created FetchLog record

        Example:
            fetch_log = FetchLog(
                sport_key="basketball_nba",
                events_count=10,
                bookmakers_count=8,
                success=True,
                api_quota_remaining=15000
            )
            result = await writer.log_fetch(fetch_log)
        """
        self.session.add(fetch_log)

        logger.info(
            "fetch_logged",
            sport_key=fetch_log.sport_key,
            events_count=fetch_log.events_count,
            success=fetch_log.success,
            quota_remaining=fetch_log.api_quota_remaining,
        )

        return fetch_log

    async def log_data_quality_issue(self, quality_log: DataQualityLog) -> DataQualityLog:
        """
        Log a data quality issue.

        Args:
            quality_log: DataQualityLog instance to persist

        Returns:
            Created DataQualityLog record

        Example:
            quality_log = DataQualityLog(
                event_id="abc123",
                severity="warning",
                issue_type="missing_data",
                description="Bookmaker X missing from response"
            )
            result = await writer.log_data_quality_issue(quality_log)
        """
        self.session.add(quality_log)

        logger.warning(
            "data_quality_issue",
            event_id=quality_log.event_id,
            severity=quality_log.severity,
            issue_type=quality_log.issue_type,
            description=quality_log.description,
        )

        return quality_log

    async def bulk_upsert_events(
        self, events: list[Event], now: datetime | None = None
    ) -> dict[str, int]:
        """
        Bulk upsert event records for efficient backfill operations.

        Args:
            events: List of Event instances to insert or update
            now: Clock for the LIVE exception below (defaults to the current time)

        Returns:
            Dictionary with keys 'inserted' and 'updated' containing approximate counts

        Note:
            Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE pattern.
            Preserves created_at timestamps on updates, updates updated_at field.
            Status, scores, completed_at and commence_time are never overwritten, because the
            incoming events are discovery records (scheduled, no scores). A stored event's
            kick-off moves through ``move_kickoff`` on the Odds API's authority against the
            clock ``now``, which makes a POSTPONED event that moves, or a LIVE one moved to a
            start after ``now``, SCHEDULED.

        Example:
            events = [
                Event(id="abc", sport_key="basketball_nba", ...),
                Event(id="def", sport_key="basketball_nba", ...),
            ]
            result = await writer.bulk_upsert_events(events)
            # result = {'inserted': 1, 'updated': 1}
        """
        if not events:
            logger.info("bulk_events_upserted", inserted=0, updated=0)
            return {"inserted": 0, "updated": 0}

        # Deduplicate by event ID (API returns same event across multiple dates)
        seen: dict[str, Event] = {}
        for event in events:
            seen[event.id] = event
        events = list(seen.values())

        # Determine which events already exist
        event_ids = [event.id for event in events]
        result = await self.session.execute(select(Event.id).where(Event.id.in_(event_ids)))
        existing_ids = {row[0] for row in result.fetchall()}

        # Calculate counts
        updated_count = len(existing_ids)
        inserted_count = len(events) - updated_count

        # Convert Event instances to dicts for bulk upsert
        # Use model_dump() to include all fields with defaults; the database numbers new events.
        event_dicts = [event.model_dump(exclude={"surrogate_id"}) for event in events]

        # Build upsert statement
        stmt = insert(Event).values(event_dicts)

        events_table = Event.__table__
        result_columns = {"status", "home_score", "away_score", "completed_at"}
        # A discovery record carries no gamePk or mark, so stored ones are kept.
        kept_columns = {
            "created_at",
            "surrogate_id",
            "commence_time",
            "mlb_game_pk",
            "no_game_pk_reason",
            *result_columns,
        }
        set_ = {
            col.name: stmt.excluded[col.name]
            for col in events_table.columns
            if not col.primary_key and col.name not in kept_columns
        }
        # Override updated_at to current time (SQLAlchemy doesn't auto-apply Python defaults)
        set_["updated_at"] = datetime.now(UTC)

        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_=set_,
        )

        await self.session.execute(stmt)

        now = now or datetime.now(UTC)
        if existing_ids:
            stored = await self.session.execute(
                select(Event)
                .where(Event.id.in_(existing_ids))  # type: ignore[attr-defined]
                .execution_options(populate_existing=True)
            )
            stored_by_id = {e.id: e for e in stored.scalars()}
            for event in events:
                held = stored_by_id.get(event.id)
                if held is not None:
                    await move_kickoff(
                        self.session, held, Source.ODDS_API, event.commence_time, now=now
                    )

        logger.info(
            "bulk_events_upserted",
            inserted=inserted_count,
            updated=updated_count,
            total=len(events),
        )

        return {"inserted": inserted_count, "updated": updated_count}
