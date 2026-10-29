"""Readers see one snapshot time per fetch, though an Odds API fetch stores a snapshot per market."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from odds_core.models import Event
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.shaped_snapshots import SnapshotShape
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tests.integration.test_odds_api_writers import _book, _listed
from tests.test_helpers import make_epl_event

KICKOFF = datetime.now(UTC).replace(microsecond=0) + timedelta(days=2)
SHAPE = SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, None)


async def _store(session: AsyncSession, event: Event, at: datetime) -> None:
    envelope: dict[str, Any] = {**_listed("apihex1", KICKOFF), "bookmakers": [_book("bet365")]}
    await OddsWriter(session).store_envelope(
        event.id, envelope, snapshot_time=at, validate=False, shape=SHAPE
    )
    await session.flush()


@pytest.fixture
async def event_with_two_fetches(pglite_async_session: AsyncSession) -> Event:
    session = pglite_async_session
    event = make_epl_event(1, KICKOFF)
    session.add(event)
    await session.flush()
    await _store(session, event, KICKOFF - timedelta(hours=30))
    await _store(session, event, KICKOFF - timedelta(hours=2))
    await session.commit()
    return event


@pytest.mark.asyncio
class TestSplitSnapshotReaders:
    async def test_snapshot_count_is_one_per_fetch(
        self, pglite_async_session: AsyncSession, event_with_two_fetches: Event
    ) -> None:
        reader = OddsReader(pglite_async_session)

        assert await reader.get_snapshot_count_for_event(1) == 2

    async def test_min_snapshots_counts_fetches(
        self, pglite_async_session: AsyncSession, event_with_two_fetches: Event
    ) -> None:
        reader = OddsReader(pglite_async_session)
        window = (KICKOFF - timedelta(days=1), KICKOFF + timedelta(days=1))

        assert len(await reader.get_events_by_date_range(*window, min_snapshots=2)) == 1
        assert await reader.get_events_by_date_range(*window, min_snapshots=3) == []

    async def test_odds_history_returns_only_snapshots_of_the_requested_market(
        self, pglite_async_engine: Any, event_with_two_fetches: Event
    ) -> None:
        from odds_mcp.server import get_odds_history

        factory = async_sessionmaker(pglite_async_engine, expire_on_commit=False)
        with patch("odds_mcp.server.async_session_maker", factory):
            result = await get_odds_history(1, "totals")

        assert result["snapshot_count"] == 2
        assert len(result["snapshots"]) == 2
        assert all(s["odds"] for s in result["snapshots"])
