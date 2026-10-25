"""Integration tests for the kick-off move gated by each sport's schedule authority."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_core.models import Event, EventExternalId, EventStatus, OddsSnapshot
from odds_core.vocabulary import Source
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY, move_kickoff

from tests.test_helpers import TWIN_KICKOFF

MOVED = TWIN_KICKOFF + timedelta(days=2)
BEFORE = TWIN_KICKOFF - timedelta(days=1)
IN_PLAY = TWIN_KICKOFF + timedelta(minutes=30)


@pytest.fixture
async def mlb_event(pglite_async_session) -> Event:
    event = Event(
        id="mlb_1",
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
    session.add(
        EventExternalId(source=Source.ODDSPORTAL, external_id="op-1", event_id=event.surrogate_id)
    )
    await session.flush()


class TestMoveKickoff:
    @pytest.mark.asyncio
    async def test_the_schedule_authority_moves_and_retimes(
        self, pglite_async_session, mlb_event
    ) -> None:
        snapshot = OddsSnapshot(
            event_id=mlb_event.id,
            snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
            raw_data={},
            bookmaker_count=0,
            fetch_tier=FetchTier.CLOSING.value,
            hours_until_commence=1.0,
        )
        pglite_async_session.add(snapshot)
        await pglite_async_session.flush()

        moved = await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, MOVED, now=BEFORE
        )

        assert moved
        assert mlb_event.commence_time == MOVED
        assert mlb_event.status is EventStatus.SCHEDULED
        assert snapshot.hours_until_commence == pytest.approx(49.0)
        assert snapshot.fetch_tier != FetchTier.CLOSING.value

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
    async def test_the_lone_mlb_schedule_may_move_to_a_played_start_in_the_past(
        self, pglite_async_session, mlb_event
    ) -> None:
        assert await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, IN_PLAY, now=MOVED
        )
        assert mlb_event.commence_time == IN_PLAY
