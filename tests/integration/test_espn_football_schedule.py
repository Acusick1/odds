"""Storing ESPN's football schedule as events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.models import Event, EventExternalId, EventStatus
from odds_core.vocabulary import Source
from odds_lambda.espn_football_schedule import (
    LEAGUES,
    EspnFootballGame,
    SyncReport,
    sync_games,
)
from odds_lambda.storage.event_resolver import ListedGame, resolve_event
from sqlalchemy import select

from tests.integration.test_polymarket_odds import SessionMaker

LEAGUE = LEAGUES["soccer_spain_la_liga"]


def kickoff() -> datetime:
    return (datetime.now(UTC) + timedelta(days=5)).replace(minute=0, second=0, microsecond=0)


def game(
    *,
    espn_id: str = "401882806",
    commence_time: datetime | None = None,
    status: EventStatus | None = None,
    upcoming: bool = True,
    away: str = "Deportivo La Coruña",
) -> EspnFootballGame:
    return EspnFootballGame(
        espn_id=espn_id,
        commence_time=commence_time or kickoff(),
        home_team="Alavés",
        away_team=away,
        status=status,
        upcoming=upcoming,
    )


def event(commence_time: datetime, *, status: EventStatus = EventStatus.SCHEDULED) -> Event:
    return Event(
        id=7,
        sport_key=LEAGUE.sport_key,
        sport_title=LEAGUE.sport_title,
        commence_time=commence_time,
        home_team="Alavés",
        away_team="Deportivo La Coruña",
        status=status,
    )


@pytest.fixture
def patch_session(monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
    monkeypatch.setattr(
        "odds_lambda.espn_football_schedule.async_session_maker",
        SessionMaker(pglite_async_session),
    )


class TestSyncGames:
    async def test_creates_a_scheduled_event_with_the_espn_id(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        report = SyncReport()

        await sync_games(LEAGUE, [game()], report, apply=True)

        stored = (await pglite_async_session.execute(select(Event))).scalar_one()
        held = await pglite_async_session.get(EventExternalId, (Source.ESPN, "401882806"))
        assert (stored.sport_key, stored.status, report.created) == (
            "soccer_spain_la_liga",
            EventStatus.SCHEDULED,
            1,
        )
        assert held is not None and held.event_id == stored.id

    async def test_a_second_run_matches_the_event(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        await sync_games(LEAGUE, [game()], SyncReport(), apply=True)
        report = SyncReport()

        await sync_games(LEAGUE, [game()], report, apply=True)

        assert (report.created, report.matched) == (0, 1)
        assert len((await pglite_async_session.execute(select(Event))).scalars().all()) == 1

    async def test_an_event_another_source_created_is_matched_not_duplicated(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        pglite_async_session.add(event(start))
        await pglite_async_session.flush()
        report = SyncReport()

        await sync_games(LEAGUE, [game(commence_time=start)], report, apply=True)

        assert (report.created, report.matched) == (0, 1)

    async def test_a_started_match_creates_nothing(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        report = SyncReport()

        await sync_games(LEAGUE, [game(upcoming=False)], report, apply=True)

        assert (report.created, report.skipped) == (0, 1)
        assert (await pglite_async_session.execute(select(Event))).first() is None

    async def test_a_postponed_match_postpones_the_event_it_has(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        pglite_async_session.add(event(start))
        await pglite_async_session.flush()
        report = SyncReport()

        await sync_games(
            LEAGUE,
            [game(commence_time=start, status=EventStatus.POSTPONED, upcoming=False)],
            report,
            apply=True,
        )

        stored = (await pglite_async_session.execute(select(Event))).scalar_one()
        assert (stored.status, report.rescheduled) == (EventStatus.POSTPONED, 1)

    async def test_a_postponed_match_without_an_event_creates_none(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        await sync_games(
            LEAGUE,
            [game(status=EventStatus.POSTPONED, upcoming=False)],
            SyncReport(),
            apply=True,
        )

        assert (await pglite_async_session.execute(select(Event))).first() is None

    async def test_a_rescheduled_match_moves_its_event_and_ends_the_postponement(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        await sync_games(LEAGUE, [game(commence_time=start)], SyncReport(), apply=True)
        await sync_games(
            LEAGUE,
            [game(commence_time=start, status=EventStatus.POSTPONED, upcoming=False)],
            SyncReport(),
            apply=True,
        )
        later = start + timedelta(days=30)

        await sync_games(LEAGUE, [game(commence_time=later)], SyncReport(), apply=True)

        stored = (await pglite_async_session.execute(select(Event))).scalar_one()
        assert (stored.commence_time, stored.status) == (later, EventStatus.SCHEDULED)

    async def test_a_dry_run_writes_nothing_and_counts_the_events_it_would_create(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        pglite_async_session.add(event(start))
        await pglite_async_session.flush()
        report = SyncReport()

        await sync_games(
            LEAGUE,
            [game(commence_time=start), game(espn_id="2", commence_time=start + timedelta(days=7))],
            report,
            apply=False,
        )

        assert (report.created, report.matched) == (1, 1)
        assert len((await pglite_async_session.execute(select(Event))).scalars().all()) == 1
        assert await pglite_async_session.get(EventExternalId, (Source.ESPN, "2")) is None


class TestPostponementFirstSeen:
    async def test_a_postponement_seen_first_attaches_the_espn_id_so_a_later_reschedule_moves_the_event(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        pglite_async_session.add(event(start))
        await pglite_async_session.flush()

        await sync_games(
            LEAGUE,
            [game(commence_time=start, status=EventStatus.POSTPONED, upcoming=False)],
            SyncReport(),
            apply=True,
        )
        held = await pglite_async_session.get(EventExternalId, (Source.ESPN, "401882806"))
        assert held is not None and held.event_id == 7

        later = start + timedelta(days=1)
        report = SyncReport()
        await sync_games(LEAGUE, [game(commence_time=later)], report, apply=True)

        stored = (await pglite_async_session.execute(select(Event))).scalars().all()
        assert report.created == 0
        assert [(e.commence_time, e.status) for e in stored] == [(later, EventStatus.SCHEDULED)]

    async def test_a_status_only_update_for_no_matching_event_creates_none_and_moves_none(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        start = kickoff()
        pglite_async_session.add(event(start))
        await pglite_async_session.flush()

        await sync_games(
            LEAGUE,
            [
                game(
                    commence_time=start + timedelta(days=9),
                    status=EventStatus.POSTPONED,
                    upcoming=False,
                )
            ],
            SyncReport(),
            apply=True,
        )
        await sync_games(
            LEAGUE,
            [game(espn_id="2", status=EventStatus.CANCELLED, upcoming=False, away="Levante")],
            SyncReport(),
            apply=True,
        )

        stored = (await pglite_async_session.execute(select(Event))).scalars().all()
        assert [(e.commence_time, e.status) for e in stored] == [(start, EventStatus.SCHEDULED)]


class TestStaleListingAfterPostponement:
    async def test_a_stale_oddsportal_listing_does_not_move_a_postponed_event_back_to_the_past(
        self, pglite_async_session: Any, patch_session: None
    ) -> None:
        now = datetime.now(UTC)
        old = now - timedelta(hours=2)
        later = now + timedelta(days=1)
        await sync_games(LEAGUE, [game(commence_time=old)], SyncReport(), apply=True)
        stored = (await pglite_async_session.execute(select(Event))).scalar_one()
        pglite_async_session.add(
            EventExternalId(source=Source.ODDSPORTAL, external_id="op1", event_id=stored.id)
        )
        await pglite_async_session.flush()
        await sync_games(LEAGUE, [game(commence_time=later)], SyncReport(), apply=True)
        assert stored.commence_time == later

        await resolve_event(
            pglite_async_session,
            ListedGame(
                source=Source.ODDSPORTAL,
                sport_key=LEAGUE.sport_key,
                sport_title=LEAGUE.sport_title,
                home_team=stored.home_team,
                away_team=stored.away_team,
                commence_time=old,
                external_id="op1",
            ),
        )

        assert stored.commence_time == later
