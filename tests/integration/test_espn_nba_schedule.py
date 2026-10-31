"""Creating events for played NBA games from ESPN's scoreboard."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.models import Event, EventExternalId, EventStatus
from odds_core.vocabulary import Source
from odds_lambda.espn_nba_schedule import EspnNbaGame, SyncReport, sync_games
from sqlalchemy import select

from tests.integration.test_polymarket_odds import SessionMaker

TIPOFF = datetime(2026, 10, 22, 2, 30, tzinfo=UTC)


def game(final: bool = True, espn_id: str = "401909842") -> EspnNbaGame:
    return EspnNbaGame(
        espn_id=espn_id,
        commence_time=TIPOFF,
        home_team="Los Angeles Clippers",
        away_team="Sacramento Kings",
        final=final,
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
        await sync_games([game(final=False)], SyncReport(), apply=True)

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

    async def test_dry_run_writes_nothing(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        report = SyncReport()

        await sync_games([game()], report, apply=False)

        assert report.created == 1
        assert (await pglite_async_session.execute(select(Event))).first() is None
