"""Integration tests for merging op_live_ twins into Odds API events."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_core.api_models import EventsResponse
from odds_core.match_brief_models import MatchBrief
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot, utc_now
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_lambda.event_sync import EventSyncService
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlmodel import SQLModel

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    StubOddsClient,
    make_epl_event,
)

CHILD_MODELS = (OddsSnapshot, Odds, Prediction, PaperTrade, MatchBrief)


async def _event_ids(session, model) -> set[str]:
    return set((await session.execute(select(model.event_id))).scalars().all())


class TestAdoptLiveTwins:
    @pytest.mark.asyncio
    async def test_moves_every_row_and_deletes_twin(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(minutes=30))
        session.add(api_event)
        await session.flush()

        adopted = await OddsWriter(session).adopt_live_twins([api_event])

        assert adopted == 1
        for model in CHILD_MODELS:
            assert await _event_ids(session, model) == {API_EVENT_ID}, model
        assert await session.get(Event, TWIN_EVENT_ID) is None

    @pytest.mark.asyncio
    async def test_covers_every_table_referencing_events(self) -> None:
        referencing = {
            table.name
            for table in SQLModel.metadata.tables.values()
            for fk in table.foreign_keys
            if fk.column.table is Event.__table__
        }
        assert {m.__tablename__ for m in CHILD_MODELS} <= referencing

    @pytest.mark.asyncio
    async def test_carries_result_when_api_event_has_none(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        twin.completed_at = utc_now()
        api_event = make_epl_event(API_EVENT_ID)
        session.add(api_event)
        await session.flush()

        await OddsWriter(session).adopt_live_twins([api_event])

        stored = await session.get(Event, API_EVENT_ID)
        assert stored.status == EventStatus.FINAL
        assert (stored.home_score, stored.away_score) == (2, 1)
        assert stored.completed_at is not None

    @pytest.mark.asyncio
    async def test_keeps_api_result_when_present(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        api_event = make_epl_event(
            API_EVENT_ID, status=EventStatus.FINAL, home_score=3, away_score=0
        )
        session.add(api_event)
        await session.flush()

        await OddsWriter(session).adopt_live_twins([api_event])

        stored = await session.get(Event, API_EVENT_ID)
        assert (stored.home_score, stored.away_score) == (3, 0)

    @pytest.mark.asyncio
    async def test_no_twin_is_a_noop(self, pglite_async_session) -> None:
        api_event = make_epl_event(API_EVENT_ID)
        pglite_async_session.add(api_event)
        await pglite_async_session.flush()

        assert await OddsWriter(pglite_async_session).adopt_live_twins([api_event]) == 0

    @pytest.mark.asyncio
    async def test_twin_outside_window_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(hours=3))
        session.add(api_event)
        await session.flush()

        assert await OddsWriter(session).adopt_live_twins([api_event]) == 0
        assert await session.get(Event, TWIN_EVENT_ID) is not None
        assert await _event_ids(session, OddsSnapshot) == {TWIN_EVENT_ID}

    @pytest.mark.asyncio
    async def test_different_teams_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, away_team="Liverpool")
        session.add(api_event)
        await session.flush()

        assert await OddsWriter(session).adopt_live_twins([api_event]) == 0
        assert await session.get(Event, TWIN_EVENT_ID) is not None

    @pytest.mark.asyncio
    async def test_event_sync_adopts_twin_of_existing_api_event(
        self, live_twin_with_rows, mock_session_factory, test_engine
    ) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID)
        session.add(api_event)
        await session.commit()

        client = StubOddsClient(
            None,
            EventsResponse(
                events=[make_epl_event(API_EVENT_ID)],
                response_time_ms=1,
                quota_remaining=None,
                timestamp=utc_now(),
            ),
        )
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        await service.sync_sport("soccer_epl")

        session.expire_all()
        assert await session.get(Event, TWIN_EVENT_ID) is None
        assert await _event_ids(session, PaperTrade) == {API_EVENT_ID}


class TestFindOrCreateEventPrefersApiEvent:
    @pytest.mark.asyncio
    async def test_prefers_non_live_candidate(self, pglite_async_session) -> None:
        # The API id sorts after the twin id, so ordering alone would pick the twin.
        pglite_async_session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await pglite_async_session.flush()

        event_id, created = await OddsWriter(pglite_async_session).find_or_create_event(
            "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl", "EPL"
        )

        assert (event_id, created) == (API_EVENT_ID, False)
