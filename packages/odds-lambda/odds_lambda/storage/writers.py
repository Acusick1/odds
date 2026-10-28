"""Database write operations for odds data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot
from odds_core.odds_math import outcome_price_decimal
from odds_core.time import ensure_utc, parse_api_datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from odds_lambda.storage.snapshot_store import SnapshotWriteOutcome
from odds_lambda.storage.validators import OddsValidator
from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier_from_timestamps

logger = structlog.get_logger()


@dataclass(frozen=True)
class StoredEnvelope:
    """``snapshot`` is the stored one; ``inserted`` is false when it already held the key.

    ``snapshots_written`` counts the snapshots inserted: one per market of a shaped envelope.
    """

    snapshot: OddsSnapshot
    odds: list[Odds]
    inserted: bool
    snapshots_written: int = 0


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

    async def store_envelope(
        self,
        event_id: str,
        raw_data: dict,
        snapshot_time: datetime | None = None,
        validate: bool = True,
        *,
        shape: SnapshotShape | None = None,
        api_request_id: str | None = None,
    ) -> StoredEnvelope:
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
            The stored snapshot, its normalized Odds records, and whether it was inserted
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
            new_snapshots = [snapshot]
        else:
            if event is None:
                raise ValueError(f"event {event_id} is not stored")
            written = await store_shaped_snapshot(self.session, snapshot, event, shape)
            new_snapshots = [
                w.snapshot
                for w in ([written.first] if written.first else []) + list(written.others)
                if w.outcome is SnapshotWriteOutcome.INSERTED
            ]
            if written.first is None:
                new_snapshots = [written.snapshot]
            if not new_snapshots:
                logger.info(
                    "odds_snapshot_already_stored",
                    event_id=event_id,
                    snapshot_id=written.snapshot.id,
                )
                return StoredEnvelope(written.snapshot, [], inserted=False)
            snapshot = written.snapshot

        # Each odds row sits on the snapshot whose raw_data holds it
        odds_records: list[Odds] = []
        for stored in new_snapshots:
            assert stored.id is not None
            odds_records += await self._parse_and_store_odds(
                event_id=event_id,
                snapshot_id=stored.id,
                bookmakers=stored.raw_data.get("bookmakers", []),
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

        return StoredEnvelope(
            snapshot, odds_records, inserted=True, snapshots_written=len(new_snapshots)
        )

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
        """``store_envelope`` returning the snapshot and its ``odds`` rows."""
        stored = await self.store_envelope(
            event_id,
            raw_data,
            snapshot_time,
            validate,
            shape=shape,
            api_request_id=api_request_id,
        )
        return stored.snapshot, stored.odds

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
