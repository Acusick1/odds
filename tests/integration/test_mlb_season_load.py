"""Loading played MLB seasons as FINAL events from a recorded MLB Stats API schedule."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.mlb_season_load import LoadReport, load_seasons
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.test_polymarket_odds import SessionMaker
from tests.test_helpers import add_event, game_pk_of, hold_game_pks, with_game_pk

LOADED = 8


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def mlb_event(
    home: str,
    away: str,
    commence_time: datetime,
    status: EventStatus = EventStatus.FINAL,
    scores: tuple[int, int] | None = None,
) -> Event:
    return Event(
        sport_key="baseball_mlb",
        sport_title="MLB",
        home_team=home,
        away_team=away,
        commence_time=commence_time,
        status=status,
        home_score=scores[0] if scores else None,
        away_score=scores[1] if scores else None,
    )


@pytest.fixture
def patch_session(monkeypatch: pytest.MonkeyPatch, pglite_async_session: AsyncSession) -> None:
    monkeypatch.setattr(
        "odds_lambda.mlb_season_load.async_session_maker", SessionMaker(pglite_async_session)
    )


async def events(session: AsyncSession) -> list[Event]:
    return list((await session.execute(select(Event).order_by(Event.commence_time))).scalars())


async def load(fetcher: Any, apply: bool = True) -> LoadReport:
    return await load_seasons([2024], apply=apply, fetcher=fetcher)


@pytest.mark.usefixtures("patch_session")
class TestLoadSeasons:
    async def test_creates_one_final_event_per_game_with_its_game_pk(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        report = await load(recorded_mlb_fetcher)

        stored = await events(pglite_async_session)
        assert (report.games, report.created, report.finalised, report.errors) == (
            LOADED,
            LOADED,
            LOADED,
            [],
        )
        assert report.games_by_season == {2024: LOADED}
        assert {e.status for e in stored} == {EventStatus.FINAL}
        assert sorted([await game_pk_of(pglite_async_session, e) for e in stored]) == [
            745843,
            745844,
            745848,
            746656,
            746658,
            746816,
            746942,
            775300,
        ]
        world_series = stored[-1]
        assert (
            world_series.home_team,
            world_series.away_team,
            world_series.commence_time,
            world_series.home_score,
            world_series.away_score,
        ) == ("Los Angeles Dodgers", "New York Yankees", at("2024-10-26T00:08"), 6, 3)

    async def test_doubleheader_games_get_events_of_their_own(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        await load(recorded_mlb_fetcher)

        mets = [e for e in await events(pglite_async_session) if e.away_team == "Detroit Tigers"]
        assert [(e.commence_time, e.home_score, e.away_score) for e in mets] == [
            (at("2024-04-04T16:10"), 3, 6),
            (at("2024-04-04T20:21"), 2, 1),
        ]

    async def test_rerun_matches_the_events_it_made(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        await load(recorded_mlb_fetcher)

        report = await load(recorded_mlb_fetcher)

        assert (report.created, report.matched, report.finalised, report.score_conflicts) == (
            0,
            LOADED,
            0,
            [],
        )
        assert len(await events(pglite_async_session)) == LOADED

    async def test_an_event_holding_the_game_pk_is_reused(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        held = await add_event(
            pglite_async_session,
            with_game_pk(
                mlb_event(
                    "Los Angeles Dodgers", "New York Yankees", at("2024-10-26T00:10"), scores=(6, 3)
                ),
                775300,
            ),
        )
        await hold_game_pks(pglite_async_session)
        await pglite_async_session.commit()

        report = await load(recorded_mlb_fetcher)

        assert (report.created, report.matched) == (LOADED - 1, 1)
        assert len(await events(pglite_async_session)) == LOADED
        await pglite_async_session.refresh(held)
        assert held.commence_time == at("2024-10-26T00:08")

    async def test_an_event_without_a_game_pk_is_matched_and_settled(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        listed = await add_event(
            pglite_async_session,
            mlb_event(
                "Cleveland Guardians",
                "New York Yankees",
                at("2024-04-13T16:15"),
                status=EventStatus.SCHEDULED,
            ),
        )
        await pglite_async_session.commit()

        report = await load(recorded_mlb_fetcher)

        await pglite_async_session.refresh(listed)
        assert (report.created, report.matched) == (LOADED - 1, 1)
        assert await game_pk_of(pglite_async_session, listed) == 746656
        assert (listed.status, listed.commence_time, listed.home_score, listed.away_score) == (
            EventStatus.FINAL,
            at("2024-04-13T16:10"),
            2,
            3,
        )

    async def test_a_different_stored_score_is_reported_and_kept(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        held = await add_event(
            pglite_async_session,
            with_game_pk(
                mlb_event(
                    "Chicago White Sox", "Atlanta Braves", at("2024-06-27T20:10"), scores=(0, 1)
                ),
                746816,
            ),
        )
        await hold_game_pks(pglite_async_session)
        await pglite_async_session.commit()

        report = await load(recorded_mlb_fetcher)

        await pglite_async_session.refresh(held)
        assert len(report.score_conflicts) == 1
        assert "game 746816" in report.score_conflicts[0]
        assert (held.home_score, held.away_score) == (0, 1)

    async def test_dry_run_writes_nothing(
        self, pglite_async_session: AsyncSession, recorded_mlb_fetcher: Any
    ) -> None:
        report = await load(recorded_mlb_fetcher, apply=False)

        assert (report.created, report.finalised) == (LOADED, LOADED)
        assert await events(pglite_async_session) == []


@pytest.mark.usefixtures("patch_session")
class TestSyncMlbSeasonsCommand:
    async def test_dry_run_prints_the_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        pglite_async_session: AsyncSession,
        recorded_mlb_fetcher: Any,
    ) -> None:
        from odds_cli.commands import events as events_cli

        monkeypatch.setattr(
            "odds_lambda.mlb_season_load.MlbStatsFetcher", lambda: recorded_mlb_fetcher
        )

        await events_cli._sync_mlb_seasons([2024], apply=False)

        out = " ".join(capsys.readouterr().out.split())
        assert f"Final MLB games: {LOADED} (2024: {LOADED}); events created: {LOADED}" in out
        assert "Dry run" in out
        assert await events(pglite_async_session) == []
