"""The fetch-injuries job: its schedule, dormancy and report storage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import AlertConfig, SchedulerConfig
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event, EventStatus
from odds_lambda.injury_fetcher import InjuryRecord
from odds_lambda.jobs import fetch_injuries
from odds_lambda.scheduling.jobs import (
    JobContext,
    get_job_function,
    make_compound_job_name,
    resolve_target_sports,
)
from sqlalchemy import select

NBA = "basketball_nba"


def _close(a: datetime, b: datetime) -> bool:
    return abs(a - b) < timedelta(seconds=1)


class SessionMaker:
    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self) -> SessionMaker:
        return self

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


def _record(team: str = "Boston Celtics", game_date: date = date(2026, 10, 31)) -> InjuryRecord:
    return InjuryRecord(
        report_time=datetime(2026, 10, 31, 22, 0, tzinfo=UTC),
        game_date=game_date,
        game_time_et="07:00 (ET)",
        matchup="BOS@ORL",
        team=team,
        player_name="Tatum, Jayson",
        status=InjuryStatus.OUT,
        reason="Ankle",
    )


class TestRegistration:
    def test_is_an_nba_only_job_in_the_default_bootstrap(self) -> None:
        assert "fetch-injuries" in SchedulerConfig().bootstrap_jobs
        assert resolve_target_sports("fetch-injuries", ["soccer_epl", NBA]) == [NBA]
        assert get_job_function("fetch-injuries") is fetch_injuries.main

    def test_heartbeat_expectation_matches_the_compound_name(self) -> None:
        name = make_compound_job_name("fetch-injuries", NBA)

        assert name == "fetch-injuries-nba"
        assert name in AlertConfig().heartbeat_expectations


class TestSchedule:
    @pytest.fixture
    def booked(self) -> Iterator[Any]:
        @asynccontextmanager
        async def alert_context(name: str) -> AsyncIterator[None]:
            yield

        with (
            patch.object(fetch_injuries, "self_schedule", new_callable=AsyncMock) as book,
            patch.object(fetch_injuries, "job_alert_context", alert_context),
            patch.object(fetch_injuries, "collect_report", new_callable=AsyncMock) as collect,
        ):
            yield SimpleNamespace(book=book, collect=collect)

    @staticmethod
    def _next_kickoff(kickoff: datetime | None) -> Any:
        async def lookup(sport_key: str, *, now: datetime) -> datetime | None:
            return kickoff if kickoff is not None and kickoff > now else None

        return patch("odds_lambda.scheduling.decision.get_next_kickoff", lookup)

    @pytest.mark.asyncio
    async def test_books_the_two_hour_run_before_a_distant_game(self, booked: Any) -> None:
        kickoff = datetime.now(UTC) + timedelta(hours=5)

        with self._next_kickoff(kickoff):
            await fetch_injuries.main(JobContext(sport=NBA))

        call = booked.book.await_args.kwargs
        assert call["job_name"] == "fetch-injuries-nba"
        assert _close(call["next_time"], kickoff - timedelta(hours=2))

    @pytest.mark.asyncio
    async def test_a_late_run_books_the_half_hour_run(self, booked: Any) -> None:
        """Past the two-hour lead the later one is booked, not a catch-up of the first."""
        kickoff = datetime.now(UTC) + timedelta(hours=1, minutes=40)

        with self._next_kickoff(kickoff):
            await fetch_injuries.main(JobContext(sport=NBA))

        assert _close(booked.book.await_args.kwargs["next_time"], kickoff - timedelta(minutes=30))

    @pytest.mark.asyncio
    async def test_books_before_it_fetches(self, booked: Any) -> None:
        order: list[str] = []
        booked.book.side_effect = lambda **_: order.append("book")
        booked.collect.side_effect = lambda *_: order.append("fetch")

        with self._next_kickoff(datetime.now(UTC) + timedelta(hours=5)):
            await fetch_injuries.main(JobContext(sport=NBA))

        assert order == ["book", "fetch"]

    @pytest.mark.asyncio
    async def test_dormant_with_no_game_books_a_check_in_only(self, booked: Any) -> None:
        with self._next_kickoff(None):
            await fetch_injuries.main(JobContext(sport=NBA))

        booked.collect.assert_not_awaited()
        assert booked.book.await_args.kwargs["next_time"] > datetime.now(UTC) + timedelta(hours=5)

    @pytest.mark.asyncio
    async def test_other_sports_are_refused(self, booked: Any) -> None:
        with pytest.raises(ValueError, match="supports basketball_nba"):
            await fetch_injuries.main(JobContext(sport="soccer_epl"))

        booked.book.assert_not_awaited()


class TestCollectReport:
    NOW = datetime(2026, 10, 31, 22, 4, tzinfo=UTC)

    @pytest.fixture(autouse=True)
    def database(self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
        monkeypatch.setattr(
            fetch_injuries, "async_session_maker", SessionMaker(pglite_async_session)
        )
        monkeypatch.setattr(
            "odds_lambda.collection_calendar.async_session_maker",
            SessionMaker(pglite_async_session),
        )

    @staticmethod
    def _fetched(records: list[InjuryRecord] | None) -> Any:
        found = (datetime(2026, 10, 31, 22, 0, tzinfo=UTC), records) if records else None
        return patch.object(
            fetch_injuries,
            "fetch_latest_injury_report_isolated",
            new_callable=AsyncMock,
            return_value=found,
        )

    @staticmethod
    def _event(kickoff: datetime, home: str, away: str) -> Event:
        return Event(
            sport_key=NBA,
            sport_title="NBA",
            commence_time=kickoff,
            home_team=home,
            away_team=away,
            status=EventStatus.SCHEDULED,
        )

    @pytest.mark.asyncio
    async def test_stores_the_report_linked_to_its_event(self, pglite_async_session: Any) -> None:
        event = self._event(
            datetime(2026, 10, 31, 23, 0, tzinfo=UTC), "Orlando Magic", "Boston Celtics"
        )
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        with (
            self._fetched([_record()]),
            patch.object(fetch_injuries, "send_job_warning", new_callable=AsyncMock) as warn,
        ):
            rows = await fetch_injuries.collect_report(self.NOW)

        stored = (await pglite_async_session.execute(select(InjuryReport))).scalar_one()
        assert rows == 1
        assert stored.event_id == event.id
        warn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_warns_about_a_team_with_no_event(self, pglite_async_session: Any) -> None:
        with (
            self._fetched([_record(team="LA Clippers")]),
            patch.object(fetch_injuries, "send_job_warning", new_callable=AsyncMock) as warn,
        ):
            await fetch_injuries.collect_report(self.NOW)

        warn.assert_awaited_once()
        assert "Los Angeles Clippers 2026-10-31" in warn.await_args.args[1]

    @pytest.mark.asyncio
    async def test_no_report_with_a_game_soon_warns_without_failing(
        self, pglite_async_session: Any
    ) -> None:
        pglite_async_session.add(
            self._event(self.NOW + timedelta(hours=2), "Orlando Magic", "Boston Celtics")
        )
        await pglite_async_session.commit()

        with (
            self._fetched(None),
            patch.object(fetch_injuries, "send_job_warning", new_callable=AsyncMock) as warn,
        ):
            assert await fetch_injuries.collect_report(self.NOW) == 0

        warn.assert_awaited_once()
        assert "no report" in warn.await_args.args[1]

    @pytest.mark.asyncio
    async def test_no_report_without_a_game_soon_is_quiet(self, pglite_async_session: Any) -> None:
        pglite_async_session.add(
            self._event(self.NOW + timedelta(hours=20), "Orlando Magic", "Boston Celtics")
        )
        await pglite_async_session.commit()

        with (
            self._fetched(None),
            patch.object(fetch_injuries, "send_job_warning", new_callable=AsyncMock) as warn,
        ):
            assert await fetch_injuries.collect_report(self.NOW) == 0

        warn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_old_report_with_a_game_soon_is_stored_and_warned_about(
        self, pglite_async_session: Any
    ) -> None:
        pglite_async_session.add(
            self._event(self.NOW + timedelta(hours=2), "Orlando Magic", "Boston Celtics")
        )
        await pglite_async_session.commit()
        stale_slot = self.NOW - timedelta(hours=9)
        found = (stale_slot, [_record()])

        with (
            patch.object(
                fetch_injuries,
                "fetch_latest_injury_report_isolated",
                new_callable=AsyncMock,
                return_value=found,
            ),
            patch.object(fetch_injuries, "send_job_warning", new_callable=AsyncMock) as warn,
        ):
            assert await fetch_injuries.collect_report(self.NOW) == 1

        assert any("9:00:00 old" in call.args[1] for call in warn.await_args_list)
