"""Integration tests for the kick-off move gated by each sport's schedule authority."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_core.models import Event, EventExternalId, EventStatus, OddsSnapshot
from odds_core.vocabulary import Source
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY, move_kickoff

from tests.test_helpers import TWIN_KICKOFF

MOVED = TWIN_KICKOFF + timedelta(days=2)
BEFORE = TWIN_KICKOFF - timedelta(days=1)
IN_PLAY = TWIN_KICKOFF + timedelta(minutes=30)


@pytest.fixture
async def mlb_event(pglite_async_session) -> Event:
    event = Event(
        id=1,
        sport_key="baseball_mlb",
        sport_title="MLB",
        commence_time=TWIN_KICKOFF,
        home_team="New York Yankees",
        away_team="Boston Red Sox",
        status=EventStatus.POSTPONED,
    )
    pglite_async_session.add(event)
    await pglite_async_session.flush()
    return event


async def _hold_oddsportal_id(session, event: Event) -> None:
    session.add(EventExternalId(source=Source.ODDSPORTAL, external_id="op-1", event_id=event.id))
    await session.flush()


class TestMoveKickoff:
    @pytest.mark.asyncio
    async def test_the_schedule_authority_moves(self, pglite_async_session, mlb_event) -> None:
        moved = await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, MOVED, now=BEFORE
        )

        assert moved
        assert mlb_event.commence_time == MOVED
        assert mlb_event.status is EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_a_price_source_never_moves_an_mlb_event(
        self, pglite_async_session, mlb_event
    ) -> None:
        moved = await move_kickoff(
            pglite_async_session, mlb_event, Source.BETFAIR, MOVED, now=BEFORE
        )

        assert not moved
        assert mlb_event.commence_time == TWIN_KICKOFF
        assert mlb_event.status is EventStatus.POSTPONED

    @pytest.mark.asyncio
    async def test_a_source_outside_the_epl_order_never_moves(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        for source in set(Source) - set(KICKOFF_AUTHORITY["soccer_epl"]):
            assert not await move_kickoff(
                pglite_async_session, stored_epl_event, source, MOVED, now=BEFORE
            )
        assert stored_epl_event.commence_time == TWIN_KICKOFF

    @pytest.mark.asyncio
    async def test_before_kickoff_odds_api_yields_to_oddsportal_holding_an_id(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await _hold_oddsportal_id(pglite_async_session, stored_epl_event)

        assert not await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDS_API, MOVED, now=BEFORE
        )
        assert await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDSPORTAL, MOVED, now=BEFORE
        )
        assert stored_epl_event.commence_time == MOVED

    @pytest.mark.asyncio
    async def test_no_start_or_the_same_start_is_no_move(
        self, pglite_async_session, mlb_event
    ) -> None:
        assert not await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, None, now=BEFORE
        )
        assert not await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, TWIN_KICKOFF, now=BEFORE
        )


class TestMoveKickoffAfterTheStoredKickoff:
    @pytest.mark.asyncio
    async def test_an_in_play_start_is_refused(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        assert not await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDS_API, IN_PLAY, now=IN_PLAY
        )
        assert not await move_kickoff(
            pglite_async_session,
            stored_epl_event,
            Source.ODDSPORTAL,
            TWIN_KICKOFF - timedelta(hours=1),
            now=IN_PLAY,
        )
        assert stored_epl_event.commence_time == TWIN_KICKOFF

    @pytest.mark.asyncio
    async def test_a_postponement_to_a_future_date_follows_a_lower_source(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await _hold_oddsportal_id(pglite_async_session, stored_epl_event)

        assert await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDS_API, MOVED, now=IN_PLAY
        )
        assert stored_epl_event.commence_time == MOVED

    @pytest.mark.asyncio
    async def test_a_source_outside_the_order_still_never_moves(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        assert not await move_kickoff(
            pglite_async_session, stored_epl_event, Source.BETFAIR, MOVED, now=IN_PLAY
        )

    @pytest.mark.asyncio
    async def test_a_played_start_source_may_move_to_a_start_in_the_past(
        self, pglite_async_session, mlb_event
    ) -> None:
        assert await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, IN_PLAY, now=MOVED
        )
        assert mlb_event.commence_time == IN_PLAY

    @pytest.mark.asyncio
    async def test_a_live_event_moved_to_a_future_start_is_scheduled_again(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        stored_epl_event.status = EventStatus.LIVE

        assert await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDSPORTAL, MOVED, now=IN_PLAY
        )
        assert stored_epl_event.status == EventStatus.SCHEDULED


def _snapshot(event: Event, source: Source) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event.id,
        snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
        source=source,
    )


class TestFootballDataUkMovesOnlyEventsOnlyItObserved:
    PAST_NOW = TWIN_KICKOFF + timedelta(days=400)

    @pytest.mark.asyncio
    async def test_an_event_with_no_snapshots_moves(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        assert await move_kickoff(
            pglite_async_session,
            stored_epl_event,
            Source.FOOTBALL_DATA_UK,
            MOVED,
            now=self.PAST_NOW,
        )

    @pytest.mark.asyncio
    async def test_an_event_with_only_its_own_snapshots_moves(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        pglite_async_session.add(_snapshot(stored_epl_event, Source.FOOTBALL_DATA_UK))
        await pglite_async_session.flush()

        assert await move_kickoff(
            pglite_async_session,
            stored_epl_event,
            Source.FOOTBALL_DATA_UK,
            MOVED,
            now=self.PAST_NOW,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", [Source.ODDSPORTAL, Source.BETFAIR, Source.ODDS_API])
    async def test_an_event_another_source_observed_does_not_move(
        self, pglite_async_session, stored_epl_event, source: Source
    ) -> None:
        pglite_async_session.add_all(
            [
                _snapshot(stored_epl_event, Source.FOOTBALL_DATA_UK),
                _snapshot(stored_epl_event, source),
            ]
        )
        await pglite_async_session.flush()

        assert not await move_kickoff(
            pglite_async_session,
            stored_epl_event,
            Source.FOOTBALL_DATA_UK,
            MOVED,
            now=self.PAST_NOW,
        )
        assert stored_epl_event.commence_time == TWIN_KICKOFF

    @pytest.mark.asyncio
    async def test_the_top_source_is_not_held_to_it(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        pglite_async_session.add(_snapshot(stored_epl_event, Source.FOOTBALL_DATA_UK))
        await pglite_async_session.flush()

        assert await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDSPORTAL, MOVED, now=BEFORE
        )
