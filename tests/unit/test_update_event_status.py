"""OddsWriter.update_event_status refuses regressions and writes terminal statuses."""

from datetime import UTC, datetime

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy.ext.asyncio import AsyncSession


async def _add_event(session: AsyncSession, status: EventStatus) -> None:
    session.add(
        Event(
            id=1,
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 8, 22, 14, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=status,
            home_score=2 if status == EventStatus.FINAL else None,
            away_score=1 if status == EventStatus.FINAL else None,
        )
    )
    await session.commit()


@pytest.mark.asyncio
class TestUpdateEventStatus:
    @pytest.mark.parametrize("target", [EventStatus.LIVE, EventStatus.CANCELLED])
    async def test_final_event_is_not_moved_back(
        self, test_session: AsyncSession, target: EventStatus
    ) -> None:
        await _add_event(test_session, EventStatus.FINAL)

        with pytest.raises(ValueError, match="cannot move from final"):
            await OddsWriter(test_session).update_event_status(1, target)

        event = await test_session.get(Event, 1)
        assert event is not None
        assert (event.status, event.home_score) == (EventStatus.FINAL, 2)

    async def test_postponed_event_is_written_without_scores(
        self, test_session: AsyncSession
    ) -> None:
        await _add_event(test_session, EventStatus.SCHEDULED)

        event = await OddsWriter(test_session).update_event_status(1, EventStatus.POSTPONED)

        assert event is not None
        assert (event.status, event.home_score, event.completed_at) == (
            EventStatus.POSTPONED,
            None,
            None,
        )

    async def test_final_can_be_rewritten_with_scores(self, test_session: AsyncSession) -> None:
        await _add_event(test_session, EventStatus.FINAL)

        event = await OddsWriter(test_session).update_event_status(1, EventStatus.FINAL, 3, 1)

        assert event is not None
        assert (event.home_score, event.away_score) == (3, 1)
