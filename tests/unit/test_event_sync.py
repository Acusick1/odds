"""Unit tests for EventSyncService."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.api_models import EventsResponse, OddsResponse
from odds_core.models import Event, EventStatus
from odds_lambda.event_sync import EventSyncResult, EventSyncService
from sqlalchemy import select

from tests.test_helpers import StubOddsClient


def _make_odds_response() -> OddsResponse:
    return OddsResponse(
        events=[],
        raw_events_data=[],
        response_time_ms=0,
        quota_remaining=None,
        timestamp=datetime.now(UTC),
    )


def _make_event(event_id: str, hours_from_now: float = 24.0) -> Event:
    return Event(
        id=event_id,
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime.now(UTC) + timedelta(hours=hours_from_now),
        home_team="Lakers",
        away_team="Celtics",
        status=EventStatus.SCHEDULED,
    )


def _make_events_response(events: list[Event]) -> EventsResponse:
    return EventsResponse(
        events=events,
        response_time_ms=50,
        quota_remaining=None,
        timestamp=datetime.now(UTC),
    )


class TestEventSyncService:
    """Tests for EventSyncService."""

    @pytest.mark.asyncio
    async def test_sync_sport_inserts_new_events(self, test_session, mock_session_factory):
        """New events from API are inserted into the database."""
        events = [_make_event("sync_event_1"), _make_event("sync_event_2", hours_from_now=48)]
        client = StubOddsClient(_make_odds_response(), _make_events_response(events))
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        result = await service.sync_sport("basketball_nba")

        assert isinstance(result, EventSyncResult)
        assert result.sport_key == "basketball_nba"
        assert result.inserted == 2
        assert result.updated == 0
        assert result.total == 2

        await test_session.rollback()
        db_events = (
            (
                await test_session.execute(
                    select(Event).where(Event.id.in_(["sync_event_1", "sync_event_2"]))
                )
            )
            .scalars()
            .all()
        )
        assert len(db_events) == 2

    @pytest.mark.asyncio
    async def test_sync_sport_updates_existing_events(self, test_session, mock_session_factory):
        """Events already in the database are updated, not double-counted as inserts."""
        from odds_lambda.storage.writers import OddsWriter

        # Pre-insert one event
        writer = OddsWriter(test_session)
        existing = _make_event("sync_existing_event")
        await writer.upsert_event(existing)
        await test_session.commit()

        # Sync returns the same event + one new one
        new_event = _make_event("sync_new_event")
        client = StubOddsClient(
            _make_odds_response(),
            _make_events_response([existing, new_event]),
        )
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        result = await service.sync_sport("basketball_nba")

        assert result.inserted == 1
        assert result.updated == 1
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_sync_sport_empty_response(self, mock_session_factory):
        """Empty API response returns zero counts without hitting the database."""
        client = StubOddsClient(_make_odds_response(), _make_events_response([]))
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        result = await service.sync_sport("basketball_nba")

        assert result.inserted == 0
        assert result.updated == 0
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_sync_sports_multiple_sports(self, mock_session_factory):
        """sync_sports returns one result per sport."""
        client = StubOddsClient(_make_odds_response(), _make_events_response([]))
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        results = await service.sync_sports(["basketball_nba", "basketball_ncaab"])

        assert len(results) == 2
        assert {r.sport_key for r in results} == {"basketball_nba", "basketball_ncaab"}
