"""Creating events for played NBA games from ESPN's scoreboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.models import Event, EventExternalId, EventStatus
from odds_core.vocabulary import Source
from odds_lambda.espn_nba_schedule import EspnNbaGame, SyncReport, sync_games
from sqlalchemy import select

from tests.integration.test_polymarket_odds import SessionMaker

TIPOFF = datetime(2026, 10, 22, 2, 30, tzinfo=UTC)


def game(
    status: EventStatus | None = EventStatus.FINAL,
    espn_id: str = "401909842",
    commence_time: datetime = TIPOFF,
) -> EspnNbaGame:
    final = status is EventStatus.FINAL
    return EspnNbaGame(
        espn_id=espn_id,
        commence_time=commence_time,
        home_team="Los Angeles Clippers",
        away_team="Sacramento Kings",
        status=status,
        home_score=121 if final else None,
        away_score=96 if final else None,
    )


@pytest.fixture
def patch_session(monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
    monkeypatch.setattr(
        "odds_lambda.espn_nba_schedule.async_session_maker", SessionMaker(pglite_async_session)
    )


class TestSyncGames:
    async def test_creates_a_final_event_with_the_espn_id(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        report = SyncReport()

        await sync_games([game()], report, apply=True)

        event = (await pglite_async_session.execute(select(Event))).scalar_one()
        held = await pglite_async_session.get(EventExternalId, (Source.ESPN, "401909842"))
        assert (report.created, report.finalised, report.errors) == (1, 1, [])
        assert (event.sport_key, event.status, event.home_score, event.away_score) == (
            "basketball_nba",
            EventStatus.FINAL,
            121,
            96,
        )
        assert held is not None and held.event_id == event.id

    async def test_an_unfinished_game_stays_scheduled(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        await sync_games([game(status=None)], SyncReport(), apply=True)

        event = (await pglite_async_session.execute(select(Event))).scalar_one()
        assert event.status is EventStatus.SCHEDULED

    async def test_rerun_matches_the_event_it_made(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        await sync_games([game()], SyncReport(), apply=True)
        report = SyncReport()

        await sync_games([game()], report, apply=True)

        assert (report.created, report.matched, report.finalised) == (0, 1, 0)
        assert len((await pglite_async_session.execute(select(Event))).scalars().all()) == 1

    @pytest.mark.parametrize(
        "status", [EventStatus.POSTPONED, EventStatus.CANCELLED], ids=["postponed", "cancelled"]
    )
    async def test_postponed_and_cancelled_games_set_the_event_status(
        self, pglite_async_session: Any, patch_session: None, status: EventStatus
    ) -> None:
        await sync_games([game(status=None)], SyncReport(), apply=True)
        report = SyncReport()

        await sync_games([game(status=status)], report, apply=True)

        event = (await pglite_async_session.execute(select(Event))).scalar_one()
        assert (event.status, report.rescheduled) == (status, 1)

    async def test_a_rescheduled_game_moves_an_event_only_espn_knows(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        await sync_games([game(status=EventStatus.POSTPONED)], SyncReport(), apply=True)
        later = datetime.now(UTC) + timedelta(days=30)

        await sync_games([game(status=None, commence_time=later)], SyncReport(), apply=True)

        event = (await pglite_async_session.execute(select(Event))).scalar_one()
        assert (event.commence_time, event.status) == (later, EventStatus.SCHEDULED)

    async def test_dry_run_writes_nothing(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        report = SyncReport()

        await sync_games([game()], report, apply=False)

        assert report.created == 1
        assert (await pglite_async_session.execute(select(Event))).first() is None


class TestSyncUpcoming:
    async def test_covers_three_days_back_and_three_weeks_ahead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import date

        from odds_lambda import espn_nba_schedule

        seen: list[tuple[date, date, bool]] = []

        async def fake(since: date, until: date, *, apply: bool) -> SyncReport:
            seen.append((since, until, apply))
            return SyncReport()

        monkeypatch.setattr(espn_nba_schedule, "sync_schedule", fake)

        await espn_nba_schedule.sync_upcoming(datetime(2026, 11, 1, 6, tzinfo=UTC))

        assert seen == [(date(2026, 10, 29), date(2026, 11, 22), True)]


class TestUnsettledNbaEvents:
    async def test_health_monitor_covers_nba_but_not_the_previous_season(
        self, pglite_async_session: Any
    ) -> None:
        from odds_lambda.health_monitor import UNSETTLED_AGE, UNSETTLED_SPORTS
        from odds_lambda.storage.unsettled_events import find_unsettled_events

        pglite_async_session.add(
            Event(
                id=1,
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2026, 4, 20, 1, 10, tzinfo=UTC),
                home_team="Los Angeles Clippers",
                away_team="Sacramento Kings",
                status=EventStatus.SCHEDULED,
            )
        )
        await pglite_async_session.commit()

        stuck = await find_unsettled_events(
            pglite_async_session,
            "basketball_nba",
            UNSETTLED_AGE,
            datetime(2026, 10, 31, tzinfo=UTC),
        )

        assert "basketball_nba" in UNSETTLED_SPORTS
        assert stuck == []
