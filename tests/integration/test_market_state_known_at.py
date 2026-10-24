"""load_market_state's known_at filter against a database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.candidate_rules import load_market_state

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

pytestmark = pytest.mark.integration

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)


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
            [three_way_bookmaker("bet365", (2.0, 3.0, 4.0))],
            "oddsportal_live",
        )
        late.created_at = T + timedelta(minutes=5)
        session.add(late)
        await session.commit()
        return event

    async def test_none_ignores_creation_time(self, pglite_async_session, event) -> None:
        state = await load_market_state(pglite_async_session, event, T)
        assert "bet365" in state.retail

    async def test_snapshot_stored_after_known_at_is_excluded(
        self, pglite_async_session, event
    ) -> None:
        state = await load_market_state(pglite_async_session, event, T, known_at=T)
        assert state.retail == {}

    async def test_snapshot_stored_by_known_at_is_included(
        self, pglite_async_session, event
    ) -> None:
        known_at = T + timedelta(minutes=5)
        state = await load_market_state(pglite_async_session, event, T, known_at=known_at)
        assert "bet365" in state.retail
