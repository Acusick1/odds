"""A collector sleeps through an off-season and wakes when the next season is published."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.epl_data_models import EspnFixture
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import Event, EventStatus
from odds_core.vocabulary import Market
from odds_lambda import collection_calendar
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.betfair.kickoff import next_betfair_kickoff
from odds_lambda.collection_calendar import decide_collection_run, next_fixture_kickoff
from odds_lambda.jobs import fetch_betfair_exchange
from odds_lambda.jobs.fetch_oddsportal import CADENCE
from odds_lambda.scheduling.decision import DORMANCY_HORIZON_DAYS

from tests.test_helpers import make_tagged_snapshot, store_tagged

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


class TestBetfairKickoff:
    """The Betfair job also runs before an earlier Betfair start (#640)."""

    FIRST_PITCH = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    BETFAIR_START = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
    LAST_RUN = datetime(2026, 10, 24, 22, 47, tzinfo=UTC)

    async def _seed(
        self,
        session: Any,
        *starts: str | None,
        sport_key: str = "baseball_mlb",
        totals_start: str | None = None,
    ) -> None:
        event = _event(self.FIRST_PITCH, sport_key=sport_key)
        session.add(event)
        if sport_key == "baseball_mlb":
            session.add(_mlb_game(849802, self.FIRST_PITCH, self.LAST_RUN - timedelta(hours=6)))
        match_odds = SPORT_CONFIG[sport_key].match_odds.market  # type: ignore[index]
        snapshots = [(minutes, start, match_odds) for minutes, start in enumerate(starts)]
        if totals_start is not None:
            snapshots.append((len(starts), totals_start, Market.TOTALS))
        for minutes, start, market in snapshots:
            if market is Market.TOTALS:
                names = ("Over 2.5 Goals", "Under 2.5 Goals")
            elif market is Market.H2H:
                names = ("Home", "Away")
            else:
                names = ("Home", "The Draw", "Away")
            meta = {
                "inplay": False,
                "market_start_time": start,
                "runners": [{"runner_name": name, "best_back": 2.5} for name in names],
            }
            snapshot_time = self.LAST_RUN - timedelta(minutes=30 - minutes)
            snapshot = make_tagged_snapshot(
                event.id,
                snapshot_time,
                [
                    {
                        "key": "betfair_exchange",
                        "markets": [{"key": market.value, "outcomes": []}],
                        "betfair_meta": meta,
                    }
                ],
                "betfair_api",
            )
            await store_tagged(session, snapshot)
        await session.commit()

    @staticmethod
    async def _decide(sport_key: str, now: datetime, *, betfair: bool = True) -> datetime | None:
        decision = await decide_collection_run(
            [sport_key],
            fetch_betfair_exchange.CADENCE,
            overnight=None,
            now=now,
            stored_kickoff=next_betfair_kickoff if betfair else None,
            leads=[fetch_betfair_exchange.CLOSE_LEAD],
        )
        return decision.next_execution

    async def test_a_close_run_lands_before_an_earlier_betfair_start(
        self, pglite_async_session: Any
    ) -> None:
        await self._seed(pglite_async_session, self.BETFAIR_START.isoformat())

        assert await self._decide("baseball_mlb", self.LAST_RUN) == self.BETFAIR_START - timedelta(
            minutes=5
        )

    async def test_the_commence_time_close_run_is_kept_after_the_betfair_start(
        self, pglite_async_session: Any
    ) -> None:
        await self._seed(pglite_async_session, self.BETFAIR_START.isoformat())

        after_start = self.BETFAIR_START + timedelta(minutes=1)
        assert await next_betfair_kickoff("baseball_mlb", now=after_start) == self.FIRST_PITCH
        last_cadence_run = self.FIRST_PITCH - timedelta(minutes=14)
        assert await self._decide("baseball_mlb", last_cadence_run) == self.FIRST_PITCH - timedelta(
            minutes=5
        )

    async def test_a_totals_snapshot_does_not_move_the_kickoff(
        self, pglite_async_session: Any
    ) -> None:
        await self._seed(
            pglite_async_session,
            None,
            sport_key="soccer_epl",
            totals_start=self.BETFAIR_START.isoformat(),
        )

        assert await next_betfair_kickoff("soccer_epl", now=self.LAST_RUN) == self.FIRST_PITCH

    async def test_the_latest_snapshot_decides_the_betfair_start(
        self, pglite_async_session: Any
    ) -> None:
        later = (self.FIRST_PITCH + timedelta(hours=1)).isoformat()
        await self._seed(pglite_async_session, self.BETFAIR_START.isoformat(), later)

        assert await next_betfair_kickoff("baseball_mlb", now=self.LAST_RUN) == self.FIRST_PITCH

    @pytest.mark.parametrize(
        "start",
        [
            None,
            "not-a-time",
            (FIRST_PITCH + timedelta(minutes=75)).isoformat(),
            (FIRST_PITCH - timedelta(hours=3, minutes=1)).isoformat(),
        ],
        ids=["no_start", "malformed_start", "later_start", "start_beyond_the_bound"],
    )
    async def test_the_kickoff_stays_the_commence_time(
        self, pglite_async_session: Any, start: str | None
    ) -> None:
        await self._seed(pglite_async_session, start)

        assert await next_betfair_kickoff("baseball_mlb", now=self.LAST_RUN) == self.FIRST_PITCH

    @pytest.mark.parametrize("minutes_before", [60, 20, 4], ids=["pregame", "closing", "close_run"])
    async def test_a_betfair_start_at_kickoff_keeps_the_timing(
        self, pglite_async_session: Any, minutes_before: int
    ) -> None:
        await self._seed(pglite_async_session, self.FIRST_PITCH.isoformat(), sport_key="soccer_epl")
        now = self.FIRST_PITCH - timedelta(minutes=minutes_before)

        assert await self._decide("soccer_epl", now) == await self._decide(
            "soccer_epl", now, betfair=False
        )
