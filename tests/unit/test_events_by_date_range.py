"""Tests for OddsReader.get_events_by_date_range ordering."""

from datetime import UTC, datetime

from odds_core.models import Event, EventStatus
from odds_lambda.storage.readers import OddsReader
from sqlalchemy.ext.asyncio import AsyncSession

KICKOFF = datetime(2024, 5, 19, 15, 0, tzinfo=UTC)


def _event(event_id: int, commence_time: datetime) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team=f"{event_id} home",
        away_team=f"{event_id} away",
        status=EventStatus.FINAL,
    )


class TestGetEventsByDateRangeOrder:
    async def test_kickoff_ties_are_ordered_by_event_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        early = datetime(2024, 5, 18, 15, 0, tzinfo=UTC)
        for event_id, commence in [(3, KICKOFF), (1, KICKOFF), (4, early), (2, KICKOFF)]:
            pglite_async_session.add(_event(event_id, commence))
        await pglite_async_session.commit()

        events = await OddsReader(pglite_async_session).get_events_by_date_range(
            start_date=datetime(2024, 5, 1, tzinfo=UTC),
            end_date=datetime(2024, 5, 31, tzinfo=UTC),
        )

        assert [e.id for e in events] == [4, 1, 2, 3]
