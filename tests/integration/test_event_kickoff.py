"""Integration tests for the kick-off move gated by each sport's schedule authority."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_core.models import Event, EventExternalId, EventStatus, OddsSnapshot
from odds_core.vocabulary import Source
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage import event_kickoff
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY, move_kickoff

from tests.test_helpers import TWIN_KICKOFF

MOVED = TWIN_KICKOFF + timedelta(days=2)


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

        moved = await move_kickoff(pglite_async_session, mlb_event, Source.MLB_STATS_API, MOVED)

        assert moved
        assert mlb_event.commence_time == MOVED
        assert mlb_event.status is EventStatus.SCHEDULED
        assert snapshot.hours_until_commence == pytest.approx(49.0)
        assert snapshot.fetch_tier != FetchTier.CLOSING.value

    @pytest.mark.asyncio
    async def test_a_price_source_never_moves_an_mlb_event(
        self, pglite_async_session, mlb_event
    ) -> None:
        moved = await move_kickoff(pglite_async_session, mlb_event, Source.BETFAIR, MOVED)

        assert not moved
        assert mlb_event.commence_time == TWIN_KICKOFF
        assert mlb_event.status is EventStatus.POSTPONED

    @pytest.mark.asyncio
    async def test_a_source_outside_the_epl_order_never_moves(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        for source in set(Source) - set(KICKOFF_AUTHORITY["soccer_epl"]):
            assert not await move_kickoff(pglite_async_session, stored_epl_event, source, MOVED)
        assert stored_epl_event.commence_time == TWIN_KICKOFF

    @pytest.mark.asyncio
    async def test_odds_api_yields_to_oddsportal_on_an_epl_event(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        pglite_async_session.add(
            EventExternalId(
                source=Source.ODDSPORTAL,
                external_id="op-1",
                event_id=stored_epl_event.surrogate_id,
            )
        )
        await pglite_async_session.flush()

        assert not await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDS_API, MOVED
        )
        assert await move_kickoff(pglite_async_session, stored_epl_event, Source.ODDSPORTAL, MOVED)
        assert stored_epl_event.commence_time == MOVED

    @pytest.mark.asyncio
    async def test_no_start_or_the_same_start_is_no_move(
        self, pglite_async_session, mlb_event
    ) -> None:
        assert not await move_kickoff(pglite_async_session, mlb_event, Source.MLB_STATS_API, None)
        assert not await move_kickoff(
            pglite_async_session, mlb_event, Source.MLB_STATS_API, TWIN_KICKOFF
        )

    @pytest.mark.asyncio
    async def test_a_lower_authority_yields_to_a_higher_one_holding_an_id(
        self, pglite_async_session, stored_epl_event, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            event_kickoff,
            "KICKOFF_AUTHORITY",
            {"soccer_epl": (Source.ESPN, Source.ODDS_API)},
        )

        assert await move_kickoff(pglite_async_session, stored_epl_event, Source.ODDS_API, MOVED)

        pglite_async_session.add(
            EventExternalId(
                source=Source.ESPN, external_id="espn-1", event_id=stored_epl_event.surrogate_id
            )
        )
        await pglite_async_session.flush()
        later = MOVED + timedelta(hours=1)
        assert not await move_kickoff(
            pglite_async_session, stored_epl_event, Source.ODDS_API, later
        )
        assert await move_kickoff(pglite_async_session, stored_epl_event, Source.ESPN, later)
        assert stored_epl_event.commence_time == later
