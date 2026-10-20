"""Failure behaviour of the fetch-espn-fixtures job body."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.epl_data_models import EspnFixtureRecord
from odds_lambda.collection_calendar import EmptyCollectionError, FixtureCalendar
from odds_lambda.espn_fixture_fetcher import (
    EspnSeasonFetchError,
    EspnUpcomingFetchError,
    UpcomingFixtures,
)
from odds_lambda.jobs import fetch_espn_fixtures


def _record(competition: str) -> EspnFixtureRecord:
    return EspnFixtureRecord(
        date=datetime.now(UTC) + timedelta(days=1),
        team="Arsenal",
        opponent="Chelsea",
        competition=competition,
        match_round="",
        home_away="home",
        score_team="",
        score_opponent="",
        status="Scheduled",
        state="pre",
        season="2026-27",
    )


def _fetcher(
    *,
    past: list[EspnFixtureRecord] | None = None,
    upcoming: UpcomingFixtures | None = None,
    upcoming_error: Exception | None = None,
    season_error: Exception | None = None,
) -> MagicMock:
    fetcher = MagicMock()
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    fetcher.fetch_season = AsyncMock(return_value=past or [], side_effect=season_error)
    fetcher.fetch_upcoming = AsyncMock(
        return_value=upcoming or UpcomingFixtures(records=[], failed_leagues=[]),
        side_effect=upcoming_error,
    )
    return fetcher


@pytest.fixture
def calendar() -> Iterator[AsyncMock]:
    with (
        patch.object(fetch_espn_fixtures, "fixture_calendar", new_callable=AsyncMock) as mock,
        patch.object(fetch_espn_fixtures, "async_session_maker", MagicMock()),
    ):
        yield mock


@pytest.fixture
def upsert() -> Iterator[AsyncMock]:
    with patch.object(fetch_espn_fixtures, "_upsert", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda records: len(records)
        yield mock


@pytest.fixture
def warning() -> Iterator[AsyncMock]:
    with patch.object(fetch_espn_fixtures, "send_job_warning", new_callable=AsyncMock) as mock:
        yield mock


def _scheduled(count: int) -> FixtureCalendar:
    return FixtureCalendar(
        next_kickoff=datetime.now(UTC) + timedelta(days=3) if count else None,
        kickoffs_in_horizon=count,
    )


@pytest.mark.usefixtures("upsert", "warning")
class TestFetchAndUpsertFailures:
    @pytest.mark.asyncio
    async def test_upcoming_failure_propagates_after_upserting_past(
        self, calendar: AsyncMock, upsert: AsyncMock
    ) -> None:
        past = [_record("Premier League")]
        fetcher = _fetcher(past=past, upcoming_error=EspnUpcomingFetchError("eng.1"))

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=fetcher):
            with pytest.raises(EspnUpcomingFetchError):
                await fetch_espn_fixtures._fetch_and_upsert(2026)

        upsert.assert_awaited_once_with(past)

    @pytest.mark.asyncio
    async def test_season_failure_propagates_after_upserting_upcoming(
        self, calendar: AsyncMock, upsert: AsyncMock
    ) -> None:
        upcoming = [_record("Premier League")]
        fetcher = _fetcher(
            upcoming=UpcomingFixtures(records=upcoming, failed_leagues=[]),
            season_error=EspnSeasonFetchError("all teams"),
        )

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=fetcher):
            with pytest.raises(EspnSeasonFetchError):
                await fetch_espn_fixtures._fetch_and_upsert(2026)

        upsert.assert_awaited_once_with(upcoming)

    @pytest.mark.asyncio
    async def test_empty_premier_league_with_scheduled_events_raises_after_upsert(
        self, calendar: AsyncMock, upsert: AsyncMock
    ) -> None:
        calendar.return_value = _scheduled(10)
        cup = _record("League Cup")
        fetcher = _fetcher(upcoming=UpcomingFixtures(records=[cup], failed_leagues=[]))

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=fetcher):
            with pytest.raises(EmptyCollectionError, match="no upcoming Premier League"):
                await fetch_espn_fixtures._fetch_and_upsert(2026)

        upsert.assert_awaited_once_with([cup])

    @pytest.mark.asyncio
    async def test_empty_without_scheduled_events_passes(self, calendar: AsyncMock) -> None:
        calendar.return_value = _scheduled(0)

        with patch.object(fetch_espn_fixtures, "EspnFixtureFetcher", return_value=_fetcher()):
            assert await fetch_espn_fixtures._fetch_and_upsert(2026) == (0, 0, 0)

    @pytest.mark.asyncio
    async def test_cup_failures_warn_and_continue(
        self, calendar: AsyncMock, warning: AsyncMock
    ) -> None:
        upcoming = UpcomingFixtures(
            records=[_record("Premier League")], failed_leagues=["eng.fa", "uefa.europa"]
        )

        with patch.object(
            fetch_espn_fixtures, "EspnFixtureFetcher", return_value=_fetcher(upcoming=upcoming)
        ):
            assert await fetch_espn_fixtures._fetch_and_upsert(2026) == (0, 1, 1)

        warning.assert_awaited_once()
        assert "eng.fa, uefa.europa" in warning.await_args.args[1]
        calendar.assert_not_awaited()
