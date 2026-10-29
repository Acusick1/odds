"""fetch-espn-fixtures marks events of ESPN-postponed Premier League matches POSTPONED."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from odds_core.epl_data_models import EspnFixture
from odds_core.models import Event, EventStatus
from odds_lambda.jobs import fetch_espn_fixtures
from sqlalchemy.ext.asyncio import AsyncSession

KICKOFF = datetime(2026, 5, 17, 14, 0, tzinfo=UTC)


def _event(event_id: int, status: EventStatus, commence_time: datetime = KICKOFF) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team="Bournemouth",
        away_team="Manchester City",
        status=status,
        home_score=1 if status == EventStatus.FINAL else None,
        away_score=0 if status == EventStatus.FINAL else None,
    )


def _fixture(
    team: str,
    opponent: str,
    home_away: str,
    status: str = "Postponed",
    competition: str = "Premier League",
) -> EspnFixture:
    return EspnFixture(
        date=KICKOFF,
        team=team,
        opponent=opponent,
        competition=competition,
        match_round="",
        home_away=home_away,
        score_team="0",
        score_opponent="0",
        status=status,
        state="post",
        season="2025-26",
    )


def _postponed_pair(**kwargs: str) -> list[EspnFixture]:
    return [
        _fixture("Bournemouth", "Manchester City", "home", **kwargs),
        _fixture("Manchester City", "Bournemouth", "away", **kwargs),
    ]


@pytest.fixture
def session_maker(test_session: AsyncSession):
    maker = MagicMock()
    maker.return_value.__aenter__.return_value = test_session
    maker.return_value.__aexit__.return_value = None
    with patch.object(fetch_espn_fixtures, "async_session_maker", maker):
        yield maker


@pytest.mark.asyncio
class TestMarkPostponedEvents:
    async def test_postponed_match_marks_its_event(
        self, test_session: AsyncSession, session_maker: MagicMock
    ) -> None:
        test_session.add(_event(1, EventStatus.SCHEDULED))
        test_session.add_all(_postponed_pair())
        await test_session.commit()

        assert await fetch_espn_fixtures._mark_postponed_events() == 1

        test_session.expire_all()
        event = await test_session.get(Event, 1)
        assert event is not None
        assert event.status == EventStatus.POSTPONED

    async def test_final_and_postponed_events_are_left_alone(
        self, test_session: AsyncSession, session_maker: MagicMock
    ) -> None:
        test_session.add(_event(1, EventStatus.FINAL))
        test_session.add(_event(2, EventStatus.POSTPONED, KICKOFF + timedelta(minutes=30)))
        test_session.add_all(_postponed_pair())
        await test_session.commit()

        assert await fetch_espn_fixtures._mark_postponed_events() == 0
        assert await fetch_espn_fixtures._mark_postponed_events() == 0

        test_session.expire_all()
        statuses = {
            event_id: (await test_session.get(Event, event_id)).status  # type: ignore[union-attr]
            for event_id in (1, 2)
        }
        assert statuses == {1: EventStatus.FINAL, 2: EventStatus.POSTPONED}

    async def test_pairing_listed_again_by_espn_within_the_window_is_not_marked(
        self, test_session: AsyncSession, session_maker: MagicMock
    ) -> None:
        test_session.add(_event(1, EventStatus.SCHEDULED))
        test_session.add_all(_postponed_pair())
        relisted = _fixture("Bournemouth", "Manchester City", "home", status="Scheduled")
        relisted.date = KICKOFF + timedelta(hours=1)
        test_session.add(relisted)
        await test_session.commit()

        assert await fetch_espn_fixtures._mark_postponed_events() == 0

    async def test_event_at_a_different_kickoff_is_not_marked(
        self, test_session: AsyncSession, session_maker: MagicMock
    ) -> None:
        test_session.add(_event(1, EventStatus.SCHEDULED, KICKOFF + timedelta(days=2)))
        test_session.add_all(_postponed_pair())
        await test_session.commit()

        assert await fetch_espn_fixtures._mark_postponed_events() == 0

    async def test_other_competitions_and_statuses_are_ignored(
        self, test_session: AsyncSession, session_maker: MagicMock
    ) -> None:
        test_session.add(_event(1, EventStatus.SCHEDULED))
        test_session.add_all(_postponed_pair(competition="FA Cup"))
        test_session.add_all(
            [_fixture("Bournemouth", "Manchester City", "home", status="Scheduled")]
        )
        await test_session.commit()

        assert await fetch_espn_fixtures._mark_postponed_events() == 0
