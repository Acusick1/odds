"""Empty-schedule behaviour of the fetch-mlb-probables job body."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.collection_calendar import EmptyCollectionError, FixtureCalendar
from odds_lambda.jobs import fetch_mlb_probables


@pytest.fixture
def calendar() -> Iterator[AsyncMock]:
    with (
        patch.object(fetch_mlb_probables, "event_calendar", new_callable=AsyncMock) as mock,
        patch.object(fetch_mlb_probables, "async_session_maker", MagicMock()),
    ):
        yield mock


@pytest.fixture
def empty_fetcher() -> Iterator[MagicMock]:
    fetcher = MagicMock()
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    fetcher.fetch_dates = AsyncMock(return_value=[])
    with patch.object(fetch_mlb_probables, "MlbStatsFetcher", return_value=fetcher):
        yield fetcher


def _scheduled(count: int) -> FixtureCalendar:
    return FixtureCalendar(
        next_kickoff=datetime.now(UTC) + timedelta(days=1) if count else None,
        kickoffs_in_horizon=count,
    )


@pytest.mark.usefixtures("empty_fetcher")
class TestEmptySchedule:
    @pytest.mark.asyncio
    async def test_empty_with_scheduled_events_raises(self, calendar: AsyncMock) -> None:
        calendar.return_value = _scheduled(4)
        now = datetime.now(UTC)

        with pytest.raises(EmptyCollectionError, match="MLB Stats API returned no games"):
            await fetch_mlb_probables._fetch_and_store(now, [now.date()])

        assert calendar.await_args.args[1] == "baseball_mlb"
        assert calendar.await_args.kwargs["horizon"] == timedelta(days=3)

    @pytest.mark.asyncio
    async def test_empty_off_season_passes(self, calendar: AsyncMock) -> None:
        calendar.return_value = _scheduled(0)
        now = datetime.now(UTC)

        assert await fetch_mlb_probables._fetch_and_store(now, [now.date()]) == 0
