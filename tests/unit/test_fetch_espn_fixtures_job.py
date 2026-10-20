"""Failure behaviour of the fetch-espn-fixtures job body."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.collection_calendar import EmptyCollectionError, FixtureCalendar
from odds_lambda.espn_fixture_fetcher import EspnUpcomingFetchError
from odds_lambda.jobs import fetch_espn_fixtures


def _fetcher(*, upcoming: Any = None, upcoming_error: Exception | None = None) -> MagicMock:
    fetcher = MagicMock()
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    fetcher.fetch_season = AsyncMock(return_value=[])
    fetcher.fetch_upcoming = AsyncMock(return_value=upcoming or [], side_effect=upcoming_error)
    return fetcher


@pytest.fixture
def calendar() -> Iterator[AsyncMock]:
    with (
        patch.object(fetch_espn_fixtures, "fixture_calendar", new_callable=AsyncMock) as mock,
        patch.object(fetch_espn_fixtures, "async_session_maker", MagicMock()),
    ):
        yield mock


class TestFetchAndUpsertFailures:
    @pytest.mark.asyncio
    async def test_all_upcoming_requests_failing_propagates(self, calendar: AsyncMock) -> None:
        fetcher = _fetcher(upcoming_error=EspnUpcomingFetchError("every competition"))

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=fetcher):
            with pytest.raises(EspnUpcomingFetchError):
                await fetch_espn_fixtures._fetch_and_upsert(2026)

    @pytest.mark.asyncio
    async def test_empty_upcoming_with_scheduled_events_raises(self, calendar: AsyncMock) -> None:
        calendar.return_value = FixtureCalendar(
            next_kickoff=datetime.now(UTC) + timedelta(days=3), kickoffs_in_horizon=10
        )

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=_fetcher()):
            with pytest.raises(EmptyCollectionError, match="no upcoming fixtures"):
                await fetch_espn_fixtures._fetch_and_upsert(2026)

    @pytest.mark.asyncio
    async def test_empty_upcoming_without_scheduled_events_passes(
        self, calendar: AsyncMock
    ) -> None:
        calendar.return_value = FixtureCalendar(next_kickoff=None, kickoffs_in_horizon=0)

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=_fetcher()):
            assert await fetch_espn_fixtures._fetch_and_upsert(2026) == (0, 0, 0)
