"""Integration tests for live MLB events keyed to their MLB Stats API gamePk (#612)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from odds_core.config import get_settings
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.health_monitor import HealthMonitor
from odds_lambda.jobs import fetch_betfair_exchange, fetch_oddsportal
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME, LeagueScrape, ingest_league
from odds_lambda.live_events import DroppedGame, SkipReason, alert_dropped_games
from odds_lambda.mlb_game_resolver import GameRow, ResolutionRule, ResolvedGame
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.oddsportal_adapter import MatchOdds
from odds_lambda.storage.event_merge import ConflictingGamePks, EventMerger
from odds_lambda.storage.writers import OddsWriter
from odds_lambda.tier_utils import calculate_tier_from_timestamps
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import make_epl_event

MLB = "baseball_mlb"
YANKEES, RED_SOX = "New York Yankees", "Boston Red Sox"
CUBS, CARDINALS = "Chicago Cubs", "St. Louis Cardinals"
# 23:10 UTC is 19:10 US Eastern on the same date, so the MLB date is the UTC date.
EVENING = (datetime.now(UTC) + timedelta(days=1)).replace(
    hour=23, minute=10, second=0, microsecond=0
)
AFTERNOON = EVENING.replace(hour=17, minute=5)


def _game(
    pk: int,
    start: datetime,
    home: str,
    away: str,
    game_number: int = 1,
    *,
    tbd: bool = False,
    state: MlbGameState = MlbGameState.SCHEDULED,
) -> MlbGameResult:
    return MlbGameResult(
        game_pk=pk,
        start_time=start,
        home_team=home,
        away_team=away,
        is_final=False,
        home_score=None,
        away_score=None,
        game_number=game_number,
        start_time_tbd=tbd,
        official_date=start.date(),
        state=state,
        schedule_date=start.date(),
    )


def _resolved(pk: int) -> ResolvedGame:
    return ResolvedGame(pk, ResolutionRule.SINGLE_GAME)


class FakeMlbStatsFetcher:
    """Stands in for MlbStatsFetcher: a scripted schedule, or an outage."""

    schedule: list[MlbGameResult] = []
    down: bool = False
    requests: int = 0

    async def __aenter__(self) -> FakeMlbStatsFetcher:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def fetch_results(self, start: date, end: date) -> list[MlbGameResult]:
        type(self).requests += 1
        if self.down:
            raise httpx.ConnectError("statsapi.mlb.com unreachable")
        return list(self.schedule)


@pytest.fixture
def mlb_schedule(monkeypatch: pytest.MonkeyPatch) -> type[FakeMlbStatsFetcher]:
    monkeypatch.setattr(FakeMlbStatsFetcher, "schedule", [])
    monkeypatch.setattr(FakeMlbStatsFetcher, "down", False)
    monkeypatch.setattr(FakeMlbStatsFetcher, "requests", 0)
    monkeypatch.setattr("odds_lambda.live_events.MlbStatsFetcher", FakeMlbStatsFetcher)
    return FakeMlbStatsFetcher


@pytest.fixture
def job_session(
    pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncSession:
    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return pglite_async_session

        async def __aexit__(self, *args: object) -> bool:
            return False

    for module in (fetch_oddsportal, fetch_betfair_exchange):
        monkeypatch.setattr(module, "async_session_maker", _Maker())
    return pglite_async_session


def _op_match(home: str, away: str, start: datetime) -> MatchOdds:
    return MatchOdds(
        market="home_away",
        home_team=home,
        away_team=away,
        match_date=start,
        match_link=f"https://www.oddsportal.com/baseball/usa/mlb/{home}-{away}-{start:%H%M}/",
        scraped_date=datetime.now(UTC),
        raw_data={
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home, "price_decimal": 1.8},
                                {"name": away, "price_decimal": 2.1},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        },
        bookmaker_count=1,
    )


def _book(event_name: str, start: datetime, home: str, away: str) -> BetfairBook:
    def runner(selection_id: int, name: str, back: float) -> BetfairRunner:
        return BetfairRunner(selection_id, name, back, back + 0.02, 100.0, 100.0, back)

    return BetfairBook(
        market_id=f"1.{event_name}",
        betfair_event_id=event_name,
        betfair_event_name=event_name,
        market_start_time=start,
        market_status="OPEN",
        inplay=False,
        total_matched=5000.0,
        runners=[runner(1, away, 2.1), runner(2, home, 1.8)],
    )


class FakeBetfairClient:
    def __init__(self, books: list[BetfairBook]) -> None:
        self.books = books

    def list_events(self, sport_cfg: Any, lookahead_hours: int) -> list[str]:
        return [book.betfair_event_id for book in self.books]

    def list_match_odds_books(self, sport_cfg: Any, events: list[str]) -> list[BetfairBook]:
        return self.books


async def _mlb_events(session: AsyncSession) -> list[Event]:
    result = await session.execute(
        select(Event).where(Event.sport_key == MLB).order_by(Event.commence_time)
    )
    return list(result.scalars().all())


async def _snapshot_event_ids(session: AsyncSession) -> list[str]:
    return list((await session.execute(select(OddsSnapshot.event_id))).scalars().all())


async def _ingest_op(*matches: MatchOdds) -> fetch_oddsportal.IngestionStats:
    return await ingest_league(
        LEAGUE_SPEC_BY_NAME["mlb"], scraped=LeagueScrape(matches=list(matches), events_listed=1)
    )


async def _ingest_betfair(*books: BetfairBook) -> fetch_betfair_exchange.IngestionStats:
    return await fetch_betfair_exchange.ingest_sport(
        SPORT_CONFIG[MLB],
        client=FakeBetfairClient(list(books)),  # type: ignore[arg-type]
        lookahead_hours=24,
        snapshot_time=datetime.now(UTC),
    )


class TestLiveMlbEventsCarryGamePk:
    async def test_resolved_game_creates_an_event_with_its_game_pk(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        mlb_schedule.schedule = [_game(813001, EVENING, YANKEES, RED_SOX)]

        stats = await _ingest_op(_op_match(YANKEES, RED_SOX, EVENING))

        [event] = await _mlb_events(job_session)
        assert event.mlb_game_pk == 813001
        assert (stats.events_created, stats.snapshots_stored, stats.games_unresolved) == (1, 1, 0)

    async def test_second_source_with_the_same_game_pk_lands_on_the_same_event(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        mlb_schedule.schedule = [_game(813001, EVENING, YANKEES, RED_SOX)]
        await _ingest_op(_op_match(YANKEES, RED_SOX, EVENING + timedelta(minutes=1)))

        # Betfair lists the game 3 h later than OddsPortal: outside the ±2 h team match.
        stats = await _ingest_betfair(
            _book(
                "Boston Red Sox (B Bello) @ New York Yankees (G Cole)",
                EVENING + timedelta(hours=3),
                YANKEES,
                RED_SOX,
            )
        )

        [event] = await _mlb_events(job_session)
        assert event.mlb_game_pk == 813001
        assert (stats.events_matched, stats.events_created) == (1, 0)
        assert await _snapshot_event_ids(job_session) == [event.id, event.id]

    async def test_doubleheader_creates_one_event_per_game_from_betfair_game_numbers(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        mlb_schedule.schedule = [
            _game(824101, AFTERNOON, CUBS, CARDINALS, game_number=1),
            _game(824102, EVENING, CUBS, CARDINALS, game_number=2),
        ]

        stats = await _ingest_betfair(
            _book("St. Louis Cardinals @ Chicago Cubs Game 1", AFTERNOON, CUBS, CARDINALS),
            _book("St. Louis Cardinals @ Chicago Cubs Game 2", EVENING, CUBS, CARDINALS),
        )

        events = await _mlb_events(job_session)
        assert sorted(e.mlb_game_pk for e in events) == [824101, 824102]
        assert stats.events_created == 2
        assert {e.home_team for e in events} == {CUBS}

    async def test_unresolved_game_is_skipped_and_counted(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        mlb_schedule.schedule = [_game(813001, EVENING, YANKEES, RED_SOX)]

        stats = await _ingest_op(
            _op_match(YANKEES, RED_SOX, EVENING), _op_match(CUBS, CARDINALS, EVENING)
        )

        [event] = await _mlb_events(job_session)
        assert event.home_team == YANKEES
        assert stats.games_unresolved == 1
        assert stats.snapshots_stored == 1
        assert mlb_schedule.requests == 1

    async def test_schedule_outage_matches_existing_events_but_creates_none(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        job_session.add(
            Event(
                id="op_live_NEWYAN_BOSRED_existing",
                sport_key=MLB,
                sport_title="MLB",
                commence_time=EVENING,
                home_team=YANKEES,
                away_team=RED_SOX,
                status=EventStatus.SCHEDULED,
            )
        )
        await job_session.commit()
        mlb_schedule.down = True

        stats = await _ingest_op(
            _op_match(YANKEES, RED_SOX, EVENING + timedelta(minutes=5)),
            _op_match(CUBS, CARDINALS, EVENING),
        )

        assert [e.id for e in await _mlb_events(job_session)] == ["op_live_NEWYAN_BOSRED_existing"]
        assert await _snapshot_event_ids(job_session) == ["op_live_NEWYAN_BOSRED_existing"]
        assert stats.games_skipped_schedule_unavailable == 1
        assert stats.errors == ["MLB schedule unavailable: no new events created"]

    async def test_epl_neither_reads_the_schedule_nor_sets_a_game_pk(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        kickoff = EVENING.replace(hour=15, minute=0)
        match = MatchOdds(
            market="1x2",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=kickoff,
            match_link="https://www.oddsportal.com/football/x/",
            scraped_date=datetime.now(UTC),
            raw_data={"bookmakers": [], "source": "oddsportal_live"},
            bookmaker_count=1,
        )

        stats = await ingest_league(
            LEAGUE_SPEC_BY_NAME["england-premier-league"],
            scraped=LeagueScrape(matches=[match], events_listed=1),
        )

        event = (await job_session.execute(select(Event))).scalar_one()
        assert event.mlb_game_pk is None
        assert stats.events_created == 1
        assert mlb_schedule.requests == 0


class TestFindOrCreateGame:
    async def test_mlb_event_without_a_game_pk_is_refused(
        self, pglite_async_session: AsyncSession
    ) -> None:
        with pytest.raises(ValueError, match="needs its gamePk"):
            await OddsWriter(pglite_async_session).find_or_create_event(
                YANKEES, RED_SOX, EVENING, MLB, "MLB"
            )

    async def test_claims_an_event_with_no_game_pk_within_two_hours(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(
            Event(
                id="bfe",
                sport_key=MLB,
                sport_title="MLB",
                commence_time=EVENING,
                home_team=YANKEES,
                away_team=RED_SOX,
            )
        )
        await session.flush()

        event_id, created = await OddsWriter(session).find_or_create_event(
            YANKEES,
            RED_SOX,
            EVENING + timedelta(minutes=30),
            MLB,
            "MLB",
            mlb_game=_resolved(813001),
        )

        assert (event_id, created) == ("bfe", False)
        assert (await session.get(Event, "bfe")).mlb_game_pk == 813001

    async def test_never_takes_an_event_that_holds_another_game_pk(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        writer = OddsWriter(session)
        first_id, _ = await writer.find_or_create_event(
            CUBS, CARDINALS, AFTERNOON, MLB, "MLB", mlb_game=_resolved(824101)
        )
        await session.flush()

        second_id, created = await writer.find_or_create_event(
            CUBS,
            CARDINALS,
            AFTERNOON + timedelta(minutes=90),
            MLB,
            "MLB",
            mlb_game=_resolved(824102),
        )

        assert created is True
        assert second_id != first_id

    async def test_game_pk_taken_concurrently_lands_on_the_holder(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                Event(
                    id="holder",
                    sport_key=MLB,
                    sport_title="MLB",
                    commence_time=EVENING + timedelta(hours=4),
                    home_team=YANKEES,
                    away_team=RED_SOX,
                    mlb_game_pk=813001,
                ),
                Event(
                    id="unassigned",
                    sport_key=MLB,
                    sport_title="MLB",
                    commence_time=EVENING,
                    home_team=YANKEES,
                    away_team=RED_SOX,
                ),
            ]
        )
        await session.commit()
        writer = OddsWriter(session)
        lookup = writer._event_by_game_pk
        calls: list[int] = []

        # The first lookup runs before a concurrent run commits the holder.
        async def lookup_after_first(game_pk: int) -> Event | None:
            calls.append(game_pk)
            return None if len(calls) == 1 else await lookup(game_pk)

        monkeypatch.setattr(writer, "_event_by_game_pk", lookup_after_first)

        event_id, created = await writer.find_or_create_event(
            YANKEES, RED_SOX, EVENING, MLB, "MLB", mlb_game=_resolved(813001)
        )

        assert (event_id, created) == ("holder", False)
        assert (await session.get(Event, "unassigned")).mlb_game_pk is None


def _mlb_event(event_id: str, commence: datetime, game_pk: int | None, **fields: Any) -> Event:
    return Event(
        id=event_id,
        sport_key=MLB,
        sport_title="MLB",
        commence_time=commence,
        home_team=CUBS,
        away_team=CARDINALS,
        mlb_game_pk=game_pk,
        **fields,
    )


class TestMergeGuard:
    async def test_merge_refuses_events_with_different_game_pks(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        keep = _mlb_event("g1", AFTERNOON, 824101)
        other = _mlb_event("g2", AFTERNOON + timedelta(minutes=30), 824102)
        session.add_all([keep, other])
        await session.flush()

        with pytest.raises(ConflictingGamePks, match="824101, 824102"):
            await EventMerger(session).merge(keep, [other])

    async def test_merge_carries_the_duplicates_game_pk_to_the_kept_event(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        keep = _mlb_event("zz9hexapi", AFTERNOON, None)
        duplicate = _mlb_event("op_live_x", AFTERNOON, 824101)
        session.add_all([keep, duplicate])
        await session.flush()

        await EventMerger(session).merge(keep, [duplicate])
        await session.flush()

        assert (await session.get(Event, "zz9hexapi")).mlb_game_pk == 824101
        assert await session.get(Event, "op_live_x") is None

    async def test_adopt_twins_refuses_twins_of_different_games(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        api_event = _mlb_event("zz9hexapi", AFTERNOON, None)
        session.add_all(
            [
                api_event,
                _mlb_event("op_live_a", AFTERNOON, 824101),
                _mlb_event("op_live_b", AFTERNOON + timedelta(minutes=40), 824102),
            ]
        )
        await session.flush()

        with pytest.raises(ConflictingGamePks):
            await EventMerger(session).adopt_twins([api_event])

    async def test_twin_groups_leave_out_events_of_different_games(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("g1", AFTERNOON, 824101),
                _mlb_event("g2", AFTERNOON + timedelta(minutes=40), 824102),
                make_epl_event("op_live_ARS_CHE"),
                make_epl_event("zz9hexapi"),
            ]
        )
        await session.flush()

        groups = await EventMerger(session).find_twin_groups()

        assert [[e.id for e in g] for g in groups] == [["op_live_ARS_CHE", "zz9hexapi"]]


class TestMissingGamePkCheck:
    async def test_warns_only_on_events_created_since_game_pks_were_assigned(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        now = datetime.now(UTC)
        session.add_all(
            [
                _mlb_event(
                    "before_assignment",
                    now - timedelta(days=9),
                    None,
                    created_at=now - timedelta(days=10),
                ),
                _mlb_event(
                    "first_assigned",
                    now - timedelta(days=4),
                    824101,
                    created_at=now - timedelta(days=5),
                ),
                _mlb_event(
                    "missing", now - timedelta(days=2), None, created_at=now - timedelta(days=3)
                ),
                _mlb_event(
                    "too_new", now + timedelta(hours=6), None, created_at=now - timedelta(hours=2)
                ),
            ]
        )
        await session.flush()

        issues = await HealthMonitor(session, get_settings()).check_event_identity(now)

        assert issues == ["1 baseball_mlb events have no gamePk: missing"]

    async def test_silent_before_any_event_holds_a_game_pk(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        old = datetime.now(UTC) - timedelta(days=3)
        session.add(_mlb_event("legacy", old, None, created_at=old))
        await session.flush()

        assert await HealthMonitor(session, get_settings()).check_event_identity() == []


class TestDiscoveryKeepsGamePk:
    async def test_re_discovered_event_keeps_its_game_pk(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_mlb_event("zz9hexapi", AFTERNOON, 824101))
        await session.commit()

        await OddsWriter(session).bulk_upsert_events([_mlb_event("zz9hexapi", AFTERNOON, None)])
        await session.commit()

        stored = await session.get(Event, "zz9hexapi", populate_existing=True)
        assert stored.mlb_game_pk == 824101


def _null_event(
    event_id: str, commence: datetime, home: str = CUBS, away: str = CARDINALS, **fields: Any
) -> Event:
    fields.setdefault("status", EventStatus.SCHEDULED)
    return Event(
        id=event_id,
        sport_key=MLB,
        sport_title="MLB",
        commence_time=commence,
        home_team=home,
        away_team=away,
        **fields,
    )


async def _snapshots(session: AsyncSession) -> list[tuple[str, float | None]]:
    rows = await session.execute(
        select(OddsSnapshot.event_id, OddsSnapshot.hours_until_commence).order_by(OddsSnapshot.id)
    )
    return [(event_id, hours) for event_id, hours in rows.all()]


class TestDoubleheaderClaims:
    async def test_tbd_game_two_listed_first_leaves_game_ones_event_to_game_one(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        game_one_id = OddsWriter._build_event_id(CUBS, CARDINALS, AFTERNOON)
        job_session.add(_null_event(game_one_id, AFTERNOON))
        await job_session.commit()
        placeholder = AFTERNOON + timedelta(minutes=6)
        mlb_schedule.schedule = [
            _game(901, AFTERNOON, CUBS, CARDINALS, 1),
            _game(902, placeholder, CUBS, CARDINALS, 2, tbd=True),
        ]

        stats = await _ingest_betfair(
            _book("St. Louis Cardinals @ Chicago Cubs Game 2", placeholder, CUBS, CARDINALS),
            _book("St. Louis Cardinals @ Chicago Cubs Game 1", AFTERNOON, CUBS, CARDINALS),
        )

        events = {e.mlb_game_pk: e.id for e in await _mlb_events(job_session)}
        assert events[901] == game_one_id
        assert events[902] == OddsWriter._build_event_id(CUBS, CARDINALS, placeholder)
        assert [event_id for event_id, _ in await _snapshots(job_session)] == [
            events[902],
            events[901],
        ]
        assert stats.errors == []

    async def test_oddsportal_doubleheader_claims_each_existing_event_for_its_game(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        game_two_listed = AFTERNOON + timedelta(hours=3, minutes=31)
        job_session.add_all(
            [
                _null_event(OddsWriter._build_event_id(CUBS, CARDINALS, AFTERNOON), AFTERNOON),
                _null_event(
                    OddsWriter._build_event_id(CUBS, CARDINALS, game_two_listed), game_two_listed
                ),
            ]
        )
        await job_session.commit()
        mlb_schedule.schedule = [
            _game(901, AFTERNOON, CUBS, CARDINALS, 1),
            _game(902, AFTERNOON + timedelta(minutes=6), CUBS, CARDINALS, 2, tbd=True),
        ]

        stats = await _ingest_op(
            _op_match(CUBS, CARDINALS, game_two_listed), _op_match(CUBS, CARDINALS, AFTERNOON)
        )

        assert [e.mlb_game_pk for e in await _mlb_events(job_session)] == [901, 902]
        assert stats.events_created == 0

    async def test_id_taken_by_another_game_creates_the_event_under_a_suffixed_id(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        taken_id = OddsWriter._build_event_id(CUBS, CARDINALS, AFTERNOON)
        job_session.add(_null_event(taken_id, AFTERNOON + timedelta(hours=5), mlb_game_pk=999))
        await job_session.commit()
        mlb_schedule.schedule = [_game(901, AFTERNOON, CUBS, CARDINALS)]

        stats = await _ingest_op(_op_match(CUBS, CARDINALS, AFTERNOON))

        game = next(e for e in await _mlb_events(job_session) if e.mlb_game_pk == 901)
        assert game.id == f"{taken_id}_901"
        assert (stats.events_created, stats.errors) == (1, [])

    async def test_savepoint_keeps_the_runs_earlier_writes(
        self,
        job_session: AsyncSession,
        mlb_schedule: type[FakeMlbStatsFetcher],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        job_session.add_all(
            [
                _null_event(
                    "holder", EVENING + timedelta(hours=4), YANKEES, RED_SOX, mlb_game_pk=813001
                ),
                _null_event("unassigned", EVENING, YANKEES, RED_SOX),
            ]
        )
        await job_session.commit()
        mlb_schedule.schedule = [
            _game(824101, EVENING, CUBS, CARDINALS),
            _game(813001, EVENING, YANKEES, RED_SOX),
        ]
        lookup = OddsWriter._event_by_game_pk
        raced: set[int] = set()

        # The first lookup of 813001 runs before a concurrent run commits its holder.
        async def racy(self: OddsWriter, game_pk: int) -> Event | None:
            if game_pk == 813001 and game_pk not in raced:
                raced.add(game_pk)
                return None
            return await lookup(self, game_pk)

        monkeypatch.setattr(OddsWriter, "_event_by_game_pk", racy)

        stats = await _ingest_op(
            _op_match(CUBS, CARDINALS, EVENING), _op_match(YANKEES, RED_SOX, EVENING)
        )

        job_session.expire_all()
        event_ids = [event_id for event_id, _ in await _snapshots(job_session)]
        assert len(event_ids) == 2
        assert event_ids[1] == "holder"
        [(_, holder_hours)] = [row for row in await _snapshots(job_session) if row[0] == "holder"]
        expected = (EVENING - datetime.now(UTC)).total_seconds() / 3600
        assert holder_hours == pytest.approx(expected, abs=0.1)
        assert stats.errors == []


class TestRescheduledGame:
    async def test_postponed_then_replayed_game_moves_its_event_to_the_played_start(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        postponed_slot = EVENING - timedelta(days=1)
        job_session.add(
            _null_event(
                "old_slot",
                postponed_slot,
                YANKEES,
                RED_SOX,
                mlb_game_pk=813001,
                status=EventStatus.POSTPONED,
            )
        )
        await job_session.commit()
        mlb_schedule.schedule = [
            _game(813001, postponed_slot, YANKEES, RED_SOX, state=MlbGameState.POSTPONED),
            _game(813001, EVENING, YANKEES, RED_SOX),
        ]

        await _ingest_op(_op_match(YANKEES, RED_SOX, EVENING + timedelta(minutes=1)))

        [event] = await _mlb_events(job_session)
        assert (event.id, event.commence_time, event.status) == (
            "old_slot",
            EVENING,
            EventStatus.SCHEDULED,
        )
        [(_, hours)] = await _snapshots(job_session)
        expected = (EVENING - datetime.now(UTC)).total_seconds() / 3600
        assert hours == pytest.approx(expected, abs=0.1)

    async def test_move_re_times_the_events_stored_snapshots(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        listed = EVENING - timedelta(hours=4)
        taken_at = listed - timedelta(hours=27.6)
        job_session.add(_null_event("moved", listed, YANKEES, RED_SOX, mlb_game_pk=813001))
        job_session.add(
            OddsSnapshot(
                event_id="moved",
                snapshot_time=taken_at,
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                hours_until_commence=27.6,
                fetch_tier=calculate_tier_from_timestamps(taken_at, listed).value,
            )
        )
        await job_session.commit()
        mlb_schedule.schedule = [_game(813001, EVENING, YANKEES, RED_SOX)]

        await _ingest_op(_op_match(YANKEES, RED_SOX, EVENING))

        stored = (
            await job_session.execute(
                select(OddsSnapshot).where(OddsSnapshot.snapshot_time == taken_at)
            )
        ).scalar_one()
        assert stored.hours_until_commence == pytest.approx(31.6)
        assert stored.fetch_tier == calculate_tier_from_timestamps(taken_at, EVENING).value

    async def test_postponed_slots_closing_snapshots_are_re_tiered(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        postponed_slot = EVENING - timedelta(days=1)
        job_session.add(
            _null_event(
                "old_slot",
                postponed_slot,
                YANKEES,
                RED_SOX,
                mlb_game_pk=813001,
                status=EventStatus.POSTPONED,
            )
        )
        seeded = [
            OddsSnapshot(
                event_id="old_slot",
                snapshot_time=postponed_slot - timedelta(minutes=minutes),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                hours_until_commence=minutes / 60,
                fetch_tier=FetchTier.CLOSING.value,
            )
            for minutes in (20, 5)
        ]
        job_session.add_all(seeded)
        await job_session.commit()
        seeded_ids = [snapshot.id for snapshot in seeded]
        mlb_schedule.schedule = [
            _game(813001, postponed_slot, YANKEES, RED_SOX, state=MlbGameState.POSTPONED),
            _game(813001, EVENING, YANKEES, RED_SOX),
        ]

        await _ingest_op(_op_match(YANKEES, RED_SOX, EVENING))

        pre_slot = (
            (
                await job_session.execute(
                    select(OddsSnapshot)
                    .where(OddsSnapshot.id.in_(seeded_ids))
                    .order_by(OddsSnapshot.snapshot_time)
                )
            )
            .scalars()
            .all()
        )
        assert [s.hours_until_commence for s in pre_slot] == pytest.approx(
            [24 + 20 / 60, 24 + 5 / 60]
        )
        assert [s.fetch_tier for s in pre_slot] == [FetchTier.EARLY.value] * 2

    async def test_tbd_start_leaves_the_listed_commence_time(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        listed = AFTERNOON + timedelta(hours=3, minutes=30)
        mlb_schedule.schedule = [
            _game(901, AFTERNOON, CUBS, CARDINALS, 1, state=MlbGameState.FINAL),
            _game(902, AFTERNOON + timedelta(minutes=6), CUBS, CARDINALS, 2, tbd=True),
        ]

        await _ingest_op(_op_match(CUBS, CARDINALS, listed))

        [event] = await _mlb_events(job_session)
        assert (event.mlb_game_pk, event.commence_time) == (902, listed)


class TestScheduleOutage:
    async def test_doubleheader_attaches_only_unlabelled_rows_to_their_own_game(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        job_session.add_all(
            [
                _null_event("game_one", AFTERNOON, mlb_game_pk=901),
                _null_event("game_two", EVENING, mlb_game_pk=902),
            ]
        )
        await job_session.commit()
        mlb_schedule.down = True
        placeholder = AFTERNOON + timedelta(minutes=6)

        betfair = await _ingest_betfair(
            _book("St. Louis Cardinals @ Chicago Cubs Game 1", AFTERNOON, CUBS, CARDINALS),
            _book("St. Louis Cardinals @ Chicago Cubs Game 2", placeholder, CUBS, CARDINALS),
        )
        oddsportal = await _ingest_op(
            _op_match(CUBS, CARDINALS, AFTERNOON), _op_match(CUBS, CARDINALS, EVENING)
        )

        assert betfair.games_skipped_schedule_unavailable == 2
        assert oddsportal.games_skipped_schedule_unavailable == 0
        assert [event_id for event_id, _ in await _snapshots(job_session)] == [
            "game_one",
            "game_two",
        ]

    async def test_two_events_in_the_window_attach_nothing(
        self, job_session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]
    ) -> None:
        job_session.add_all(
            [
                _null_event("a", AFTERNOON),
                _null_event("b", AFTERNOON + timedelta(minutes=50)),
            ]
        )
        await job_session.commit()
        mlb_schedule.down = True

        stats = await _ingest_op(_op_match(CUBS, CARDINALS, AFTERNOON + timedelta(minutes=20)))

        assert stats.games_skipped_schedule_unavailable == 1
        assert await _snapshots(job_session) == []


class TestDroppedGamesAlert:
    async def test_warns_per_job_with_the_count_and_matchups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = AsyncMock(return_value=True)
        monkeypatch.setattr("odds_lambda.live_events.send_recorded_alert", sent)
        dropped = [
            DroppedGame(
                GameRow(CUBS, CARDINALS, AFTERNOON, 2),
                SkipReason.UNRESOLVED,
                "game_number_not_found",
            ),
            DroppedGame(GameRow(YANKEES, RED_SOX, EVENING), SkipReason.SCHEDULE_UNAVAILABLE),
        ]

        assert await alert_dropped_games("fetch-betfair-exchange-mlb", dropped)

        alert_type, severity, message = sent.await_args.args
        assert (alert_type, severity) == ("mlb_games_dropped:fetch-betfair-exchange-mlb", "warning")
        assert "2 listed MLB games" in message
        assert (
            f"St. Louis Cardinals @ Chicago Cubs game 2 {AFTERNOON:%Y-%m-%d %H:%M}Z "
            "(unresolved: game_number_not_found)" in message
        )
        assert "Boston Red Sox @ New York Yankees" in message
        assert sent.await_args.kwargs["every"] == timedelta(
            minutes=get_settings().alerts.stale_reminder_minutes
        )

    async def test_silent_when_nothing_was_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = AsyncMock()
        monkeypatch.setattr("odds_lambda.live_events.send_recorded_alert", sent)

        assert not await alert_dropped_games("fetch-oddsportal-mlb", [])
        sent.assert_not_awaited()
