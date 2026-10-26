"""load_market_state's known_at filter against a database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.candidate_rules import load_market_state

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
MAX_AGE = timedelta(minutes=15)
BOOKS = [
    three_way_bookmaker("bet365", (2.0, 3.0, 4.0)),
    betfair_exchange_bookmaker((2.0, 4.0, 4.0)),
]


class TestLookback:
    async def test_snapshot_older_than_the_max_age_is_not_loaded(
        self, pglite_async_session
    ) -> None:
        event = make_epl_event("evt", KICKOFF)
        pglite_async_session.add(event)
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                event.id,
                T - MAX_AGE - timedelta(seconds=1),
                BOOKS,
                "oddsportal_live",
            ),
        )
        await pglite_async_session.commit()

        assert not (await load_market_state(pglite_async_session, event, T, MAX_AGE)).has_input


class TestKnownAt:
    @pytest.fixture
    async def event(self, pglite_async_session):
        session = pglite_async_session
        event = make_epl_event("evt", KICKOFF)
        session.add(event)
        # Stamped 10 minutes before t, but stored 5 minutes after it.
        late = make_tagged_snapshot(
            event.id,
            T - timedelta(minutes=10),
            BOOKS,
            "oddsportal_live",
        )
        late.created_at = T + timedelta(minutes=5)
        await store_tagged(session, late)
        await session.commit()
        return event

    async def test_none_ignores_creation_time(self, pglite_async_session, event) -> None:
        state = await load_market_state(pglite_async_session, event, T, MAX_AGE)
        assert state.has_input and "bet365" in state.retail

    async def test_snapshot_stored_after_known_at_is_excluded(
        self, pglite_async_session, event
    ) -> None:
        state = await load_market_state(pglite_async_session, event, T, MAX_AGE, known_at=T)
        assert not state.has_input

    async def test_snapshot_stored_by_known_at_is_included(
        self, pglite_async_session, event
    ) -> None:
        known_at = T + timedelta(minutes=5)
        state = await load_market_state(pglite_async_session, event, T, MAX_AGE, known_at=known_at)
        assert state.has_input and "bet365" in state.retail
