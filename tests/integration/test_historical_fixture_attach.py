"""A historical Odds API listing attaches to the FINAL fixture football-data.co.uk loaded (#779)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_analytics import backfill_executor
from odds_analytics.backfill_executor import BackfillExecutor, BackfillResult
from odds_analytics.backfill_plan import Call, Fixture, Manifest, Target
from odds_core.models import DataQualityLog, Event, EventExternalId, EventStatus, OddsSnapshot
from odds_core.sports import SportKey
from odds_core.time import utc_isoformat
from odds_core.vocabulary import Source
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.live_events import DroppedGame, LiveEventFinder, SkipReason
from odds_lambda.mlb_game_resolver import GameRow, ResolutionRule, ResolvedGame
from odds_lambda.storage.event_resolver import (
    REFUSED_ID_ISSUE,
    ListedGame,
    ResolvedEvent,
    resolve_event,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import (
    LEAGUE_FIXTURES,
    LeagueFixture,
    ReplayHttpSession,
    SessionFactory,
    load_fduk_fixture,
)

# Valencia v Real Madrid, postponed from 3 Nov 2024 by the Valencia floods and played 3 Jan 2025.
VALENCIA_REAL = LeagueFixture(
    "soccer_spain_la_liga",
    "La Liga - Spain",
    "Valencia",
    "Real Madrid",
    "Valencia",
    "Real Madrid",
    datetime(2025, 1, 3, 20, 0, tzinfo=UTC),
)
VALENCIA_REAL_ORIGINAL = datetime(2024, 11, 3, 20, 0, tzinfo=UTC)

MLB = "baseball_mlb"
YANKEES, RED_SOX = "New York Yankees", "Boston Red Sox"


def manifest(sport: SportKey) -> Manifest:
    return Manifest.model_validate(
        {
            "sport": sport,
            "seasons": {
                "2024-25": {"start": "2024-07-01T00:00:00Z", "end": "2025-07-01T00:00:00Z"}
            },
            "markets": ["h2h"],
            "bookmakers": ["pinnacle"],
            "coverage": [
                {"min_lead_minutes": 10, "max_lead_minutes": 120, "max_input_age_minutes": 10}
            ],
            "snapshot_grid": [{"start": "2020-06-06T00:00:00Z", "step_minutes": 10}],
            "probe": {
                "snapshots": ["2024-09-14T12:00:00Z"],
                "min_fixture_lead_hours": 12,
                "required_bookmakers": {},
                "units_per_call": 10,
            },
        }
    )


def listing(
    api_id: str, fixture: LeagueFixture, start: datetime, snapshot_time: datetime
) -> dict[str, Any]:
    stamp = snapshot_time.isoformat().replace("+00:00", "Z")
    return {
        "id": api_id,
        "sport_key": fixture.sport,
        "sport_title": fixture.sport_title,
        "commence_time": start.isoformat().replace("+00:00", "Z"),
        "home_team": fixture.api_home,
        "away_team": fixture.api_away,
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": stamp,
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": stamp,
                        "outcomes": [
                            {"name": fixture.api_home, "price": 2.1},
                            {"name": fixture.api_away, "price": 3.6},
                            {"name": "Draw", "price": 3.4},
                        ],
                    }
                ],
            }
        ],
    }


async def run_call(
    session: AsyncSession,
    sport: SportKey,
    at: datetime,
    listings: list[dict[str, Any]],
    serves: tuple[Event, ...] = (),
) -> BackfillResult:
    """One planned call at ``at`` whose response lists ``listings``, through the real client and
    executor."""
    return await run_calls(session, sport, [at], listings, serves)


async def run_calls(
    session: AsyncSession,
    sport: SportKey,
    times: list[datetime],
    listings: list[dict[str, Any]],
    serves: tuple[Event, ...] = (),
) -> BackfillResult:
    """One run of planned calls at ``times``, each answered with its planned snapshot listing
    ``listings``."""
    m = manifest(sport)
    http = ReplayHttpSession(
        {
            utc_isoformat(m.request_time(t)): {"timestamp": utc_isoformat(t), "data": listings}
            for t in times
        },
        {"x-requests-last": "10", "x-requests-remaining": "1000", "x-requests-used": "10"},
    )
    client = TheOddsAPIClient(api_key="k", base_url="http://example.invalid")
    client.session = http  # type: ignore[assignment]
    band = m.coverage[0]
    calls = [
        Call(
            t,
            tuple(
                Target(Fixture(event.id, event.commence_time), band, t, t)
                for event in serves
                if event.id is not None
            ),
        )
        for t in times
    ]
    async with BackfillExecutor(
        m,
        client=client,
        session_factory=SessionFactory(session),
        skip_existing=False,
        rate_limit_seconds=0,
    ) as executor:
        return await executor.execute(calls)


async def events_of(session: AsyncSession, sport: str) -> list[Event]:
    rows = await session.execute(select(Event).where(Event.sport_key == sport).order_by(Event.id))
    return list(rows.scalars().all())


async def odds_api_ids(session: AsyncSession, event: Event) -> list[str]:
    rows = await session.execute(
        select(EventExternalId.external_id).where(
            EventExternalId.event_id == event.id, EventExternalId.source == Source.ODDS_API
        )
    )
    return list(rows.scalars().all())


async def snapshot_times(session: AsyncSession, event: Event) -> set[datetime]:
    rows = await session.execute(
        select(OddsSnapshot.snapshot_time).where(
            OddsSnapshot.event_id == event.id, OddsSnapshot.source == Source.ODDS_API
        )
    )
    return set(rows.scalars().all())


@pytest.mark.asyncio
class TestHistoricalFixtureAttach:
    @pytest.mark.parametrize("league", LEAGUE_FIXTURES, ids=lambda f: f.sport)
    async def test_listing_attaches_to_the_loaded_final_fixture(
        self, pglite_async_session: AsyncSession, league: LeagueFixture
    ) -> None:
        session = pglite_async_session
        loaded = await load_fduk_fixture(session, league)
        at = league.kickoff - timedelta(hours=1)

        result = await run_call(
            session, league.sport, at, [listing("api-1", league, league.kickoff, at)], (loaded,)
        )

        assert [e.id for e in await events_of(session, league.sport)] == [loaded.id]
        assert await odds_api_ids(session, loaded) == ["api-1"]
        assert await snapshot_times(session, loaded) == {at}
        assert (loaded.status, loaded.commence_time) == (EventStatus.FINAL, league.kickoff)
        assert (result.fixtures_stored, result.dropped_games) == (1, {})

    async def test_listing_off_the_kickoff_by_up_to_two_hours_attaches(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        fixture = LEAGUE_FIXTURES[1]
        loaded = await load_fduk_fixture(session, fixture)
        at = fixture.kickoff - timedelta(days=1)
        listed_start = fixture.kickoff + timedelta(hours=2)

        await run_call(session, fixture.sport, at, [listing("api-1", fixture, listed_start, at)])

        assert [e.id for e in await events_of(session, fixture.sport)] == [loaded.id]
        assert await odds_api_ids(session, loaded) == ["api-1"]
        assert loaded.commence_time == fixture.kickoff

    async def test_listing_with_no_loaded_fixture_creates_no_event(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        fixture = LEAGUE_FIXTURES[0]
        loaded = await load_fduk_fixture(session, fixture)
        playoff = LeagueFixture(
            fixture.sport,
            fixture.sport_title,
            "Sunderland",
            "Coventry",
            "Sunderland",
            "Coventry City",
            datetime(2025, 5, 13, 19, 0, tzinfo=UTC),
        )
        at = playoff.kickoff - timedelta(hours=1)

        result = await run_call(
            session, fixture.sport, at, [listing("api-9", playoff, playoff.kickoff, at)]
        )

        assert [e.id for e in await events_of(session, fixture.sport)] == [loaded.id]
        assert [(d.game.home_team, d.reason) for d in result.dropped_games] == [
            ("Sunderland", SkipReason.NO_EVENT)
        ]
        assert result.fixtures_stored == 0

    async def test_postponed_match_attaches_only_at_the_kickoff_it_was_played(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        fixture = VALENCIA_REAL
        loaded = await load_fduk_fixture(session, fixture)
        before = VALENCIA_REAL_ORIGINAL - timedelta(days=4)
        after = fixture.kickoff - timedelta(hours=1)

        first = await run_call(
            session, fixture.sport, before, [listing("vr", fixture, VALENCIA_REAL_ORIGINAL, before)]
        )
        await run_call(
            session,
            fixture.sport,
            after,
            [listing("vr", fixture, fixture.kickoff, after)],
            (loaded,),
        )
        rerun = await run_call(
            session, fixture.sport, before, [listing("vr", fixture, VALENCIA_REAL_ORIGINAL, before)]
        )

        assert [e.id for e in await events_of(session, fixture.sport)] == [loaded.id]
        assert await odds_api_ids(session, loaded) == ["vr"]
        assert await snapshot_times(session, loaded) == {after}
        assert (loaded.status, loaded.commence_time) == (EventStatus.FINAL, fixture.kickoff)
        for result in (first, rerun):
            assert [d.reason for d in result.dropped_games] == [SkipReason.NO_EVENT]

    async def test_an_id_another_event_holds_is_reported_as_refused(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        fixture = VALENCIA_REAL
        duplicate = await resolve_event(
            session,
            ListedGame(
                source=Source.ODDS_API,
                sport_key=fixture.sport,
                sport_title=fixture.sport_title,
                home_team=fixture.api_home,
                away_team=fixture.api_away,
                commence_time=VALENCIA_REAL_ORIGINAL,
                external_id="vr",
            ),
        )
        loaded = await load_fduk_fixture(session, fixture)
        at = fixture.kickoff - timedelta(hours=1)

        result = await run_call(
            session, fixture.sport, at, [listing("vr", fixture, fixture.kickoff, at)], (loaded,)
        )

        [dropped] = result.dropped_games
        assert dropped.reason is SkipReason.ID_REFUSED
        assert dropped.detail == (
            f"odds_api id vr is held by event {duplicate.event.id}, "
            f"not the matched event {loaded.id}"
        )
        assert await odds_api_ids(session, loaded) == []
        assert await snapshot_times(session, loaded) == set()
        assert await snapshot_times(session, duplicate.event) == set()
        refusals = await session.execute(
            select(DataQualityLog.event_id).where(DataQualityLog.issue_type == REFUSED_ID_ISSUE)
        )
        assert refusals.scalars().all() == [loaded.id]

    async def test_a_game_dropped_by_several_calls_is_reported_once(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = pglite_async_session
        fixture = LEAGUE_FIXTURES[0]
        await load_fduk_fixture(session, fixture)
        playoff = LeagueFixture(
            fixture.sport,
            fixture.sport_title,
            "Sunderland",
            "Coventry",
            "Sunderland",
            "Coventry City",
            datetime(2025, 5, 13, 19, 0, tzinfo=UTC),
        )
        start = playoff.kickoff - timedelta(hours=3)
        times = [start + timedelta(minutes=10 * i) for i in range(3)]
        alerted: list[list[DroppedGame]] = []

        async def alert(job_name: str, dropped: list[DroppedGame]) -> bool:
            alerted.append(list(dropped))
            return True

        monkeypatch.setattr(backfill_executor, "alert_dropped_games", alert)

        result = await run_calls(
            session, fixture.sport, times, [listing("api-9", playoff, playoff.kickoff, start)]
        )

        assert result.calls_made == 3
        assert [(d.game.home_team, d.reason, n) for d, n in result.dropped_games.items()] == [
            ("Sunderland", SkipReason.NO_EVENT, 3)
        ]
        assert [len(games) for games in alerted] == [1]

    async def test_relisting_a_fixture_keeps_one_event_and_one_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        fixture = LEAGUE_FIXTURES[2]
        loaded = await load_fduk_fixture(session, fixture)
        early = fixture.kickoff - timedelta(hours=2)
        late = fixture.kickoff - timedelta(minutes=30)

        for at in (early, late, late):
            await run_call(
                session, fixture.sport, at, [listing("api-1", fixture, fixture.kickoff, at)]
            )

        assert [e.id for e in await events_of(session, fixture.sport)] == [loaded.id]
        assert await odds_api_ids(session, loaded) == ["api-1"]
        assert await snapshot_times(session, loaded) == {
            early,
            late,
        }


@pytest.mark.asyncio
class TestHistoricalMlbAttach:
    async def test_a_game_attaches_to_the_event_holding_its_game_pk(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        played = datetime(2025, 6, 2, 23, 5, tzinfo=UTC)
        stored = await resolve_event(
            session,
            ListedGame(
                source=Source.MLB_STATS_API,
                sport_key=MLB,
                sport_title="MLB",
                home_team=YANKEES,
                away_team=RED_SOX,
                commence_time=played,
                mlb_game=ResolvedGame(777, ResolutionRule.SINGLE_GAME, played),
            ),
        )
        listed = GameRow(YANKEES, RED_SOX, played + timedelta(minutes=10))
        unknown = GameRow(YANKEES, RED_SOX, played + timedelta(days=1))
        finder = LiveEventFinder(
            MLB,
            "MLB",
            {
                listed: ResolvedGame(777, ResolutionRule.SINGLE_GAME, played),
                unknown: ResolvedGame(778, ResolutionRule.SINGLE_GAME, unknown.start),
            },
        )

        attached = await finder.attach_historical(session, listed, Source.ODDS_API, "api-1")
        dropped = await finder.attach_historical(session, unknown, Source.ODDS_API, "api-2")

        assert isinstance(attached, ResolvedEvent)
        assert attached.event.id == stored.event.id
        assert isinstance(dropped, DroppedGame) and dropped.reason is SkipReason.NO_EVENT
        assert [e.id for e in await events_of(session, MLB)] == [stored.event.id]
