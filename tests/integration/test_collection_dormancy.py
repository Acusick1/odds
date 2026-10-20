"""A collector sleeps through an off-season and wakes when the next season is published."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.epl_data_models import EspnFixture
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import Event, EventStatus
from odds_lambda import collection_calendar
from odds_lambda.collection_calendar import decide_collection_run, next_fixture_kickoff
from odds_lambda.jobs.fetch_oddsportal import CADENCE
from odds_lambda.scheduling.decision import DORMANCY_HORIZON_DAYS

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
HORIZON = timedelta(days=DORMANCY_HORIZON_DAYS)
OPENER = datetime(2026, 8, 21, 19, 0, tzinfo=UTC)


class SessionMaker:
    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self) -> SessionMaker:
        return self

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


@pytest.fixture(autouse=True)
def use_test_database(monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
    maker = SessionMaker(pglite_async_session)
    monkeypatch.setattr(collection_calendar, "async_session_maker", maker)
    monkeypatch.setattr("odds_core.database.async_session_maker", maker)


def _event(
    kickoff: datetime, *, sport_key: str, status: EventStatus = EventStatus.SCHEDULED
) -> Event:
    return Event(
        id=f"{sport_key}-{kickoff.isoformat()}",
        sport_key=sport_key,
        sport_title=sport_key,
        commence_time=kickoff,
        home_team="Home",
        away_team="Away",
        status=status,
    )


def _espn(
    kickoff: datetime, *, status: str = "Scheduled", competition: str = "Premier League"
) -> EspnFixture:
    return EspnFixture(
        date=kickoff,
        team="Home",
        opponent="Away",
        competition=competition,
        match_round="1",
        home_away="home",
        score_team="",
        score_opponent="",
        status=status,
        state="pre",
        season="2026-27",
    )


def _mlb_game(
    game_pk: int, kickoff: datetime, fetched: datetime, **fields: Any
) -> MlbProbablePitchers:
    return MlbProbablePitchers(
        game_pk=game_pk,
        commence_time=kickoff,
        fetched_at=fetched,
        home_team="A",
        away_team="B",
        game_type="R",
        **fields,
    )


class TestNextFixtureKickoff:
    async def test_an_empty_database_has_no_fixture(self) -> None:
        assert await next_fixture_kickoff("soccer_epl", now=NOW) is None

    async def test_the_published_epl_schedule_is_a_fixture(self, pglite_async_session: Any) -> None:
        pglite_async_session.add(_espn(OPENER))
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("soccer_epl", now=NOW) == OPENER

    async def test_the_earlier_of_stored_events_and_published_schedule_wins(
        self, pglite_async_session: Any
    ) -> None:
        stored = OPENER - timedelta(days=3)
        pglite_async_session.add_all([_espn(OPENER), _event(stored, sport_key="soccer_epl")])
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("soccer_epl", now=NOW) == stored

    @pytest.mark.parametrize(
        "fixture",
        [
            _espn(OPENER, status="Postponed"),
            _espn(OPENER, competition="FA Cup"),
            _espn(NOW - timedelta(days=1)),
        ],
        ids=["postponed", "other-competition", "past"],
    )
    async def test_these_espn_rows_are_not_fixtures(
        self, pglite_async_session: Any, fixture: EspnFixture
    ) -> None:
        pglite_async_session.add(fixture)
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("soccer_epl", now=NOW) is None

    async def test_the_epl_schedule_is_not_an_mlb_fixture(self, pglite_async_session: Any) -> None:
        pglite_async_session.add(_espn(OPENER))
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("baseball_mlb", now=NOW) is None

    async def test_the_mlb_stats_api_schedule_is_a_fixture(self, pglite_async_session: Any) -> None:
        first_pitch = NOW + timedelta(days=2)
        pglite_async_session.add(_mlb_game(1, first_pitch, NOW - timedelta(hours=3)))
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("baseball_mlb", now=NOW) == first_pitch

    async def test_a_stale_mlb_schedule_is_not_a_fixture(self, pglite_async_session: Any) -> None:
        pglite_async_session.add(_mlb_game(1, NOW + timedelta(days=2), NOW - timedelta(hours=50)))
        await pglite_async_session.commit()

        assert await next_fixture_kickoff("baseball_mlb", now=NOW) is None


class TestSeasonCycle:
    async def test_a_published_season_beyond_the_horizon_is_slept_through(
        self, pglite_async_session: Any
    ) -> None:
        pglite_async_session.add(_espn(OPENER))
        await pglite_async_session.commit()

        decision = await decide_collection_run(["soccer_epl"], CADENCE, overnight=(22, 6), now=NOW)

        assert decision.should_execute is False
        assert decision.next_execution == NOW + timedelta(hours=CADENCE.no_game)

    async def test_a_chain_checks_in_until_the_season_enters_the_horizon_then_runs(
        self, pglite_async_session: Any
    ) -> None:
        pglite_async_session.add(_espn(OPENER))
        await pglite_async_session.commit()

        moment = NOW
        check_ins = 0
        while True:
            decision = await decide_collection_run(
                ["soccer_epl"], CADENCE, overnight=(22, 6), now=moment
            )
            if decision.should_execute:
                break
            check_ins += 1
            assert decision.next_execution is not None
            moment = decision.next_execution

        assert moment == OPENER - HORIZON
        assert check_ins > 0

    async def test_the_mlb_chain_wakes_when_the_schedule_job_publishes_games(
        self, pglite_async_session: Any
    ) -> None:
        asleep = await decide_collection_run(["baseball_mlb"], CADENCE, overnight=(5, 14), now=NOW)
        pglite_async_session.add(_mlb_game(1, NOW + timedelta(days=2), NOW + timedelta(hours=5)))
        await pglite_async_session.commit()
        awake = await decide_collection_run(
            ["baseball_mlb"], CADENCE, overnight=(5, 14), now=NOW + timedelta(hours=6)
        )

        assert asleep.should_execute is False
        assert awake.should_execute is True
