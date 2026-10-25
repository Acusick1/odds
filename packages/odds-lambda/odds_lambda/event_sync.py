"""Event synchronization service using the free /events endpoint."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import structlog
from odds_core.database import async_session_maker

from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.odds_api_events import odds_api_finder, resolve_odds_api_events

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EventSyncResult:
    """Outcome of syncing events for a single sport."""

    sport_key: str
    inserted: int
    updated: int

    @property
    def total(self) -> int:
        """Total events upserted."""
        return self.inserted + self.updated


class EventSyncService:
    """
    Syncs upcoming events from the free /events endpoint into the database.

    This service is responsible for event discovery. It runs before scheduling
    intelligence so the scheduler always has an up-to-date game list to work with.
    """

    def __init__(
        self,
        client: TheOddsAPIClient,
        *,
        session_factory=async_session_maker,
    ) -> None:
        self._client = client
        self._session_factory = session_factory

    async def sync_sport(self, sport_key: str) -> EventSyncResult:
        """Fetch and upsert upcoming events for a single sport."""
        response = await self._client.get_events(sport_key)

        if not response.events:
            logger.info("event_sync_no_events", sport=sport_key)
            return EventSyncResult(sport_key=sport_key, inserted=0, updated=0)

        events = list({event.id: event for event in response.events}.values())
        finder = await odds_api_finder(sport_key, events)
        async with self._session_factory() as session:
            resolved = await resolve_odds_api_events(session, finder, events)
            await session.commit()

        logger.info(
            "event_sync_completed",
            sport=sport_key,
            inserted=resolved.created,
            updated=resolved.matched,
            dropped=len(resolved.dropped),
            failed=len(resolved.errors),
            total=resolved.created + resolved.matched,
        )
        if resolved.errors:
            raise RuntimeError(
                f"{len(resolved.errors)} {sport_key} events failed to sync: "
                + "; ".join(e.describe() for e in resolved.errors)
            )

        return EventSyncResult(
            sport_key=sport_key,
            inserted=resolved.created,
            updated=resolved.matched,
        )

    async def sync_sports(self, sports: Iterable[str]) -> list[EventSyncResult]:
        """Sync upcoming events for multiple sports."""
        return [await self.sync_sport(sport) for sport in sports]
