"""Database write operations for odds data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, OddsSnapshot
from odds_core.time import ensure_utc
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from odds_lambda.storage.snapshot_store import SnapshotWriteOutcome
from odds_lambda.storage.validators import OddsValidator

logger = structlog.get_logger()


@dataclass(frozen=True)
class StoredEnvelope:
    """``snapshot`` is the stored snapshot of the envelope's first market; ``inserted`` is false
    when every market's key was already stored. ``snapshots_written`` counts the snapshots
    inserted, one per new market."""

    snapshot: OddsSnapshot
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
        envelope: dict[str, Any],
        snapshot_time: datetime | None = None,
        validate: bool = True,
        *,
        shape: SnapshotShape,
    ) -> StoredEnvelope:
        """Store an envelope as one snapshot per market with its price rows
        (``odds_lambda.storage.shaped_snapshots.store_shaped_snapshot``). A market whose key
        (event, source, market, time) is already stored is not written again.

        Args:
            event_id: Event identifier
            envelope: What the source returned for the event, in the envelope shape
            snapshot_time: Time of snapshot (defaults to now)
            validate: Whether to run validation checks
            shape: The source, time basis and payload of the snapshot

        Raises:
            ValueError: when the event is not stored.
        """
        snapshot_time = ensure_utc(snapshot_time or datetime.now(UTC))

        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()
        if event is None:
            raise ValueError(f"event {event_id} is not stored")

        if validate:
            is_valid, warnings = self.validator.validate_odds_snapshot(envelope, event_id)
            if warnings:
                quality_log = DataQualityLog(
                    event_id=event_id,
                    severity="warning" if is_valid else "error",
                    issue_type="validation_warnings",
                    description=f"Validation found {len(warnings)} issues",
                    raw_data={"warnings": warnings[:10]},  # Store first 10
                )
                await self.log_data_quality_issue(quality_log)

        written = await store_shaped_snapshot(self.session, event, envelope, snapshot_time, shape)
        if written.first is None:
            # An envelope quoting no market is stored as one keyless snapshot.
            inserted = 1
        else:
            writes = (written.first, *written.others)
            inserted = sum(1 for w in writes if w.outcome is SnapshotWriteOutcome.INSERTED)
        if not inserted:
            logger.info(
                "odds_snapshot_already_stored",
                event_id=event_id,
                snapshot_id=written.snapshot.id,
            )
            return StoredEnvelope(written.snapshot, inserted=False)

        logger.info(
            "odds_snapshot_stored",
            event_id=event_id,
            bookmakers=len(envelope.get("bookmakers", [])),
            snapshots=inserted,
        )
        return StoredEnvelope(written.snapshot, inserted=True, snapshots_written=inserted)

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
