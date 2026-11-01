"""Tests for matching an NBA team's Eastern game date to its stored event."""

from datetime import UTC, date, datetime

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.storage.nba_event_match import match_nba_event


def _event(event_id: int, commence: datetime, *, sport_key: str = "basketball_nba") -> Event:
    return Event(
        id=event_id,
        sport_key=sport_key,
        sport_title="NBA",
        commence_time=commence,
        home_team="Orlando Magic",
        away_team="Boston Celtics",
        status=EventStatus.SCHEDULED,
    )


class TestMatchNbaEvent:
    @pytest.mark.asyncio
    async def test_late_tip_after_utc_midnight_matches_its_eastern_date(
        self, pglite_async_session
    ) -> None:
        pglite_async_session.add(_event(1, datetime(2026, 1, 17, 0, 10, tzinfo=UTC)))
        await pglite_async_session.commit()

        event_id = await match_nba_event(
            pglite_async_session, "Boston Celtics", date(2026, 1, 16), source="test"
        )

        assert event_id == 1

    @pytest.mark.asyncio
    async def test_event_of_another_sport_does_not_match(self, pglite_async_session) -> None:
        pglite_async_session.add(
            _event(1, datetime(2026, 1, 17, 0, 10, tzinfo=UTC), sport_key="baseball_mlb")
        )
        await pglite_async_session.commit()

        event_id = await match_nba_event(
            pglite_async_session, "Boston Celtics", date(2026, 1, 16), source="test"
        )

        assert event_id is None

    @pytest.mark.asyncio
    async def test_event_on_the_next_eastern_date_does_not_match(
        self, pglite_async_session
    ) -> None:
        # 2026-01-17 12:00 UTC is 07:00 ET on the 17th, an hour past the window's end.
        pglite_async_session.add(_event(1, datetime(2026, 1, 17, 12, 0, tzinfo=UTC)))
        await pglite_async_session.commit()

        event_id = await match_nba_event(
            pglite_async_session, "Boston Celtics", date(2026, 1, 16), source="test"
        )

        assert event_id is None

    @pytest.mark.asyncio
    async def test_two_events_on_the_date_match_neither(self, pglite_async_session) -> None:
        pglite_async_session.add(_event(1, datetime(2026, 1, 16, 20, 0, tzinfo=UTC)))
        pglite_async_session.add(_event(2, datetime(2026, 1, 16, 23, 0, tzinfo=UTC)))
        await pglite_async_session.commit()

        event_id = await match_nba_event(
            pglite_async_session, "Boston Celtics", date(2026, 1, 16), source="test"
        )

        assert event_id is None
