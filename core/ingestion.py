"""Shared service for ingesting odds data into the database."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import structlog

from core.config import Settings, get_settings
from core.data_fetcher import OddsResponse, TheOddsAPIClient
from core.database import async_session_maker
from core.fetch_tier import FetchTier
from core.models import FetchLog
from storage.writers import OddsWriter

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EventIngestionFailure:
    """Information about a single event that failed to ingest."""

    event_id: str | None
    error: str


@dataclass(slots=True)
class SportIngestionResult:
    """Outcome of ingesting odds for a single sport."""

    sport_key: str
    total_events: int
    processed_events: int
    quota_remaining: int | None
    response_time_ms: int
    failures: list[EventIngestionFailure] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True when no failures occurred."""
        return not self.failures and self.processed_events == self.total_events

    @property
    def error_count(self) -> int:
        """Number of failed events."""
        return len(self.failures)


@dataclass(slots=True)
class IngestionResult:
    """Aggregate result for a batch of sports."""

    sport_results: list[SportIngestionResult]

    @property
    def total_sports(self) -> int:
        """Number of sports processed."""
        return len(self.sport_results)

    @property
    def total_events(self) -> int:
        """Total events encountered across all sports."""
        return sum(result.total_events for result in self.sport_results)

    @property
    def total_processed(self) -> int:
        """Total successfully processed events."""
        return sum(result.processed_events for result in self.sport_results)

    @property
    def total_failures(self) -> int:
        """Total number of failed events."""
        return sum(result.error_count for result in self.sport_results)

    def by_sport(self, sport_key: str) -> SportIngestionResult | None:
        """Find result for a specific sport if present."""
        for result in self.sport_results:
            if result.sport_key == sport_key:
                return result
        return None


@dataclass(slots=True)
class OddsIngestionCallbacks:
    """Optional callbacks for instrumentation during ingestion."""

    on_fetch_complete: Callable[[OddsResponse], None] | None = None
    on_event_processed: Callable[[str], None] | None = None
    on_event_failed: Callable[[str | None, Exception], None] | None = None


class OddsIngestionService:
    """Service responsible for fetching odds and persisting them."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        session_factory=async_session_maker,
        client_factory: Callable[[], TheOddsAPIClient] = TheOddsAPIClient,
    ) -> None:
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._client_factory = client_factory

    async def ingest_sport(
        self,
        sport_key: str,
        *,
        fetch_tier: FetchTier | None = None,
        validate: bool | None = None,
        callbacks: OddsIngestionCallbacks | None = None,
    ) -> SportIngestionResult:
        """Fetch and persist odds for a single sport."""
        validation_enabled = self._settings.enable_validation if validate is None else validate
        callback_bundle = callbacks or OddsIngestionCallbacks()

        async with self._client_factory() as client:
            response = await client.get_odds(
                sport=sport_key,
                regions=self._settings.regions,
                markets=self._settings.markets,
                bookmakers=self._settings.bookmakers,
            )

        if callback_bundle.on_fetch_complete:
            callback_bundle.on_fetch_complete(response)

        failures: list[EventIngestionFailure] = []
        processed = 0

        for event, event_data in zip(response.events, response.raw_events_data, strict=True):
            try:
                async with self._session_factory() as session:
                    writer = OddsWriter(session)
                    await writer.upsert_event(event)
                    await session.flush()
                    await writer.store_odds_snapshot(
                        event_id=event.id,
                        raw_data=event_data,
                        snapshot_time=response.timestamp,
                        validate=validation_enabled,
                        fetch_tier=fetch_tier,
                    )
                    await session.commit()

                processed += 1
                if callback_bundle.on_event_processed:
                    callback_bundle.on_event_processed(event.id)

            except Exception as exc:  # pragma: no cover - defensive logging
                event_id = getattr(event, "id", None)
                logger.warning(
                    "ingestion_event_failed",
                    sport=sport_key,
                    event_id=event_id,
                    error=str(exc),
                )
                failures.append(EventIngestionFailure(event_id=event_id, error=str(exc)))
                if callback_bundle.on_event_failed:
                    callback_bundle.on_event_failed(event_id, exc)

        async with self._session_factory() as session:
            writer = OddsWriter(session)
            fetch_log = FetchLog(
                sport_key=sport_key,
                events_count=len(response.events),
                bookmakers_count=len(self._settings.bookmakers),
                success=len(failures) == 0,
                api_quota_remaining=response.quota_remaining,
                response_time_ms=response.response_time_ms,
            )
            await writer.log_fetch(fetch_log)
            await session.commit()

        logger.info(
            "sport_ingested",
            sport=sport_key,
            processed_events=processed,
            total_events=len(response.events),
            failures=len(failures),
            quota_remaining=response.quota_remaining,
        )

        return SportIngestionResult(
            sport_key=sport_key,
            total_events=len(response.events),
            processed_events=processed,
            quota_remaining=response.quota_remaining,
            response_time_ms=response.response_time_ms,
            failures=failures,
        )

    async def ingest_sports(
        self,
        sports: Iterable[str],
        *,
        fetch_tier: FetchTier | None = None,
        validate: bool | None = None,
        callbacks_factory: Callable[[str], OddsIngestionCallbacks] | None = None,
    ) -> IngestionResult:
        """Ingest odds for multiple sports, returning aggregated results."""
        results: list[SportIngestionResult] = []

        for sport in sports:
            callbacks = callbacks_factory(sport) if callbacks_factory else None
            result = await self.ingest_sport(
                sport,
                fetch_tier=fetch_tier,
                validate=validate,
                callbacks=callbacks,
            )
            results.append(result)

        return IngestionResult(sport_results=results)
