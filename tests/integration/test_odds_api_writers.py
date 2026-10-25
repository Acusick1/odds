"""The Odds API writers resolve games through the event resolver and store the new shape (#687)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_core.api_models import (
    EventsResponse,
    HistoricalOddsResponse,
    OddsResponse,
    create_scheduled_event,
)
from odds_core.models import (
    Event,
    EventExternalId,
    EventStatus,
    Odds,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda import odds_api_events
from odds_lambda.event_sync import EventSyncService
from odds_lambda.ingestion import OddsIngestionService
from odds_lambda.jobs import fetch_scores
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.odds_api_events import find_event_by_odds_api_id, odds_api_id_of
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import StubOddsClient, make_epl_event

KICKOFF = (datetime.now(UTC) + timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
MLB = "baseball_mlb"


class _Factory:
    """A session factory that hands every caller the test session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> _Factory:
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *args: object) -> bool:
        return False


def _listed(
    api_id: str,
    start: datetime = KICKOFF,
    home: str = "Arsenal",
    away: str = "Chelsea",
    sport_key: str = "soccer_epl",
    sport_title: str = "EPL",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": api_id,
        "sport_key": sport_key,
        "sport_title": sport_title,
        "commence_time": start.isoformat(),
        "home_team": home,
        "away_team": away,
        **extra,
    }


def _book(key: str, home: str = "Arsenal", away: str = "Chelsea") -> dict[str, Any]:
    return {
        "key": key,
        "title": key,
        "last_update": datetime.now(UTC).isoformat(),
        "markets": [
            {
                "key": "1x2",
                "outcomes": [
                    {"name": home, "price": 2.0},
                    {"name": "Draw", "price": 3.5},
                    {"name": away, "price": 4.0},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "price": 1.9, "point": 2.5},
                    {"name": "Under", "price": 1.95, "point": 2.5},
                ],
            },
        ],
    }


def _sync(session: AsyncSession, listed: list[dict[str, Any]], sport_key: str = "soccer_epl"):
    events = [create_scheduled_event(item) for item in listed]
    client = StubOddsClient(
        OddsResponse([], [], 0, None, datetime.now(UTC)),
        EventsResponse(events, 0, None, datetime.now(UTC)),
    )
    service = EventSyncService(client=client, session_factory=_Factory(session))  # type: ignore[arg-type]
    return service.sync_sport(sport_key)


async def _events(session: AsyncSession) -> list[Event]:
    return list((await session.execute(select(Event).order_by(Event.surrogate_id))).scalars())


async def _odds_api_ids(session: AsyncSession) -> dict[int, str]:
    rows = await session.execute(
        select(EventExternalId.event_id, EventExternalId.external_id).where(
            EventExternalId.source == Source.ODDS_API
        )
    )
    return dict(rows.all())


@pytest.fixture
def mlb_schedule(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[MlbGameResult]]:
    schedule: list[MlbGameResult] = []

    class _Fetcher:
        async def __aenter__(self) -> _Fetcher:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def fetch_results(self, start: date, end: date) -> list[MlbGameResult]:
            return list(schedule)

    monkeypatch.setattr("odds_lambda.live_events.MlbStatsFetcher", _Fetcher)
    yield schedule


def _mlb_game(pk: int, start: datetime, home: str, away: str) -> MlbGameResult:
    return MlbGameResult(
        game_pk=pk,
        start_time=start,
        home_team=home,
        away_team=away,
        is_final=False,
        home_score=None,
        away_score=None,
        game_number=1,
        start_time_tbd=False,
        official_date=start.date(),
        state=MlbGameState.SCHEDULED,
        schedule_date=start.date(),
    )


@pytest.mark.asyncio
class TestEventSync:
    async def test_new_game_creates_an_event_keyed_and_recorded_by_the_api_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await _sync(pglite_async_session, [_listed("apihex1")])

        assert (result.inserted, result.updated) == (1, 0)
        (event,) = await _events(pglite_async_session)
        assert event.id == "apihex1"
        assert event.status == EventStatus.SCHEDULED
        assert await _odds_api_ids(pglite_async_session) == {event.surrogate_id: "apihex1"}

    async def test_game_another_source_stored_attaches_to_that_event(
        self, pglite_async_session: AsyncSession
    ) -> None:
        stored = make_epl_event("op_live_ARS_CHE_x", KICKOFF)
        pglite_async_session.add(stored)
        await pglite_async_session.flush()

        result = await _sync(pglite_async_session, [_listed("apihex1", KICKOFF)])

        assert (result.inserted, result.updated) == (0, 1)
        (event,) = await _events(pglite_async_session)
        assert event.id == "op_live_ARS_CHE_x"
        assert await _odds_api_ids(pglite_async_session) == {event.surrogate_id: "apihex1"}

    async def test_event_keyed_by_the_api_id_is_found_after_the_api_moves_it(
        self, pglite_async_session: AsyncSession
    ) -> None:
        pglite_async_session.add(make_epl_event("apihex1", KICKOFF))
        await pglite_async_session.flush()
        moved = KICKOFF + timedelta(days=3)

        result = await _sync(pglite_async_session, [_listed("apihex1", moved)])

        assert (result.inserted, result.updated) == (0, 1)
        (event,) = await _events(pglite_async_session)
        assert event.id == "apihex1"
        assert event.commence_time == moved
        assert await _odds_api_ids(pglite_async_session) == {event.surrogate_id: "apihex1"}

    async def test_postponed_event_the_api_lists_at_a_new_start_is_scheduled_again(
        self, pglite_async_session: AsyncSession
    ) -> None:
        stored = make_epl_event("apihex1", KICKOFF, status=EventStatus.POSTPONED)
        pglite_async_session.add(stored)
        await pglite_async_session.flush()

        await _sync(pglite_async_session, [_listed("apihex1", KICKOFF + timedelta(days=5))])

        (event,) = await _events(pglite_async_session)
        assert event.status == EventStatus.SCHEDULED
        assert event.commence_time == KICKOFF + timedelta(days=5)

    async def test_relisting_keeps_status_scores_and_teams(
        self, pglite_async_session: AsyncSession
    ) -> None:
        past = datetime.now(UTC) - timedelta(days=1)
        pglite_async_session.add(
            make_epl_event(
                "apihex1",
                past,
                status=EventStatus.FINAL,
                home_score=2,
                away_score=1,
                completed_at=past + timedelta(hours=2),
            )
        )
        await pglite_async_session.flush()

        await _sync(pglite_async_session, [_listed("apihex1", past, home="Arsenal FC")])

        (event,) = await _events(pglite_async_session)
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 1)
        assert event.commence_time == past
        assert event.home_team == "Arsenal"

    async def test_listing_the_same_game_twice_makes_one_event(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await _sync(pglite_async_session, [_listed("apihex1"), _listed("apihex1")])

        assert (result.inserted, result.updated) == (1, 0)
        assert len(await _events(pglite_async_session)) == 1

    async def test_failing_game_is_rolled_back_alone_and_the_sync_raises(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = odds_api_events.resolve_odds_api_event

        async def _resolve(finder: Any, session: AsyncSession, event: Event) -> Any:
            if event.id == "bad":
                raise RuntimeError("resolver down")
            return await real(finder, session, event)

        monkeypatch.setattr(odds_api_events, "resolve_odds_api_event", _resolve)

        with pytest.raises(RuntimeError, match="1 soccer_epl events failed to sync: bad"):
            await _sync(
                pglite_async_session,
                [_listed("bad", home="Fulham", away="Leeds"), _listed("good")],
            )

        assert [e.id for e in await _events(pglite_async_session)] == ["good"]

    async def test_mlb_game_attaches_to_the_event_holding_its_game_pk(
        self, pglite_async_session: AsyncSession, mlb_schedule: list[MlbGameResult]
    ) -> None:
        start = KICKOFF.replace(hour=23)
        held = Event(
            id="op_live_NYY_BOS_x",
            sport_key=MLB,
            sport_title="MLB",
            commence_time=start,
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(held)
        await pglite_async_session.flush()
        held.assign_game_pk(824101)
        mlb_schedule.append(_mlb_game(824101, start, "New York Yankees", "Boston Red Sox"))

        result = await _sync(
            pglite_async_session,
            [
                _listed(
                    "apimlb1",
                    start,
                    "New York Yankees",
                    "Boston Red Sox",
                    sport_key=MLB,
                    sport_title="MLB",
                )
            ],
            MLB,
        )

        assert (result.inserted, result.updated) == (0, 1)
        (event,) = await _events(pglite_async_session)
        assert event.id == "op_live_NYY_BOS_x"
        assert event.mlb_game_pk == 824101
        assert await _odds_api_ids(pglite_async_session) == {event.surrogate_id: "apimlb1"}

    async def test_mlb_game_the_schedule_lacks_creates_no_event(
        self, pglite_async_session: AsyncSession, mlb_schedule: list[MlbGameResult]
    ) -> None:
        start = KICKOFF.replace(hour=23)
        mlb_schedule.append(_mlb_game(900, start, "Chicago Cubs", "St. Louis Cardinals"))

        result = await _sync(
            pglite_async_session,
            [
                _listed(
                    "apimlb2",
                    start,
                    "New York Yankees",
                    "Boston Red Sox",
                    sport_key=MLB,
                    sport_title="MLB",
                )
            ],
            MLB,
        )

        assert (result.inserted, result.updated) == (0, 0)
        assert await _events(pglite_async_session) == []


@pytest.mark.asyncio
class TestIngestion:
    async def _ingest(
        self,
        session: AsyncSession,
        mock_settings: Any,
        listed: dict[str, Any],
        bookmakers: list[dict[str, Any]],
    ) -> Any:
        payload = {**listed, "bookmakers": bookmakers}
        response = OddsResponse(
            events=[create_scheduled_event(listed)],
            raw_events_data=[payload],
            response_time_ms=10,
            quota_remaining=100,
            timestamp=datetime.now(UTC),
        )
        mock_settings.data_collection.bookmakers["soccer_epl"] = ["bet365", "williamhill"]
        service = OddsIngestionService(
            client=StubOddsClient(response),  # type: ignore[arg-type]
            settings=mock_settings,
            session_factory=_Factory(session),
        )
        return await service.ingest_sport("soccer_epl")

    async def test_envelope_becomes_one_snapshot_per_market_with_prices_and_payload(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session

        result = await self._ingest(
            session, mock_settings, _listed("apihex1"), [_book("bet365"), _book("williamhill")]
        )

        assert result.failures == []
        snapshots = (
            (await session.execute(select(OddsSnapshot).order_by(OddsSnapshot.id))).scalars().all()
        )
        assert [s.market for s in snapshots] == [Market.ONE_X_TWO, Market.TOTALS]
        assert {s.source for s in snapshots} == {Source.ODDS_API}
        assert {s.time_basis for s in snapshots} == {TimeBasis.OBSERVED}
        assert {s.api_request_id for s in snapshots} == {None}
        assert len({s.payload_id for s in snapshots}) == 1
        for snapshot in snapshots:
            assert {m["key"] for b in snapshot.raw_data["bookmakers"] for m in b["markets"]} == {
                "1x2" if snapshot.market is Market.ONE_X_TWO else "totals"
            }
        first, second = snapshots
        prices = (await session.execute(select(Price))).scalars().all()
        assert sorted(p.snapshot_id for p in prices) == sorted([first.id] * 6 + [second.id] * 4)
        odds = (await session.execute(select(Odds))).scalars().all()
        assert {o.snapshot_id for o in odds} == {first.id}
        assert len(odds) == 10
        payload = await session.get(SnapshotPayload, first.payload_id)
        assert payload is not None and payload.payload[0]["id"] == "apihex1"

    async def test_game_another_source_stored_gets_its_snapshot_on_that_event(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event("op_live_ARS_CHE_x", KICKOFF))
        await session.flush()

        await self._ingest(session, mock_settings, _listed("apihex1"), [_book("bet365")])

        events = await _events(session)
        assert [e.id for e in events] == ["op_live_ARS_CHE_x"]
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert {s.event_id for s in snapshots} == {"op_live_ARS_CHE_x"}
        assert await _odds_api_ids(session) == {events[0].surrogate_id: "apihex1"}

    async def test_repeated_fetch_at_one_time_stores_each_market_once(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session
        listed = _listed("apihex1")
        response = OddsResponse(
            events=[create_scheduled_event(listed)],
            raw_events_data=[{**listed, "bookmakers": [_book("bet365")]}],
            response_time_ms=10,
            quota_remaining=100,
            timestamp=datetime.now(UTC),
        )
        mock_settings.data_collection.bookmakers["soccer_epl"] = ["bet365", "williamhill"]
        service = OddsIngestionService(
            client=StubOddsClient(response),  # type: ignore[arg-type]
            settings=mock_settings,
            session_factory=_Factory(session),
        )

        await service.ingest_sport("soccer_epl")
        await service.ingest_sport("soccer_epl")

        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert len(snapshots) == 2
        assert len((await session.execute(select(Odds))).scalars().all()) == 5


@pytest.mark.asyncio
class TestScores:
    async def test_id_is_found_by_the_recorded_external_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _sync(session, [_listed("apihex1")])
        stored = make_epl_event("op_live_ARS_CHE_y", KICKOFF + timedelta(days=9), "Fulham", "Leeds")
        session.add(stored)
        await session.flush()
        await _sync(session, [_listed("apihex2", KICKOFF + timedelta(days=9), "Fulham", "Leeds")])

        found = await find_event_by_odds_api_id(session, "apihex2")

        assert found is not None and found.id == "op_live_ARS_CHE_y"
        assert await odds_api_id_of(session, "op_live_ARS_CHE_y") == "apihex2"

    async def test_event_stored_before_ids_were_recorded_is_found_by_its_key(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event("apihex1"))
        await session.flush()

        found = await find_event_by_odds_api_id(session, "apihex1")

        assert found is not None and found.id == "apihex1"
        assert await odds_api_id_of(session, "apihex1") == "apihex1"
        assert await find_event_by_odds_api_id(session, "nothing") is None

    async def test_job_settles_the_event_the_api_id_names(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = pglite_async_session
        past = datetime.now(UTC) - timedelta(hours=6)
        session.add(make_epl_event("op_live_ARS_CHE_z", past))
        await session.flush()
        await _sync(session, [_listed("apihex1", past)])
        scores = [
            {
                "id": "apihex1",
                "completed": True,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "scores": [
                    {"name": "Arsenal", "score": "2"},
                    {"name": "Chelsea", "score": "1"},
                ],
            },
            {"id": "unknown", "completed": True, "scores": []},
        ]

        class _Client:
            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def get_scores(self, **kwargs: Any) -> Any:
                return type("Response", (), {"scores_data": scores})()

        monkeypatch.setattr(fetch_scores, "TheOddsAPIClient", _Client)
        monkeypatch.setattr(fetch_scores, "async_session_maker", _Factory(session))

        await fetch_scores._fetch_and_update_scores(["soccer_epl"])

        event = await session.get(Event, "op_live_ARS_CHE_z", populate_existing=True)
        assert event is not None
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 1)


@pytest.mark.asyncio
class TestBackfill:
    async def test_snapshot_is_stored_on_the_event_holding_the_api_id_as_an_upper_bound(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        past = datetime(2024, 1, 15, 19, 0, tzinfo=UTC)
        session.add(make_epl_event("op_live_ARS_CHE_h", past))
        await session.flush()
        await _sync(session, [_listed("apihex1", past)])
        listed = _listed("apihex1", past)
        event = create_scheduled_event(listed)
        client = AsyncMock()
        response = HistoricalOddsResponse(
            events=[event],
            raw_events_data=[{**listed, "bookmakers": [_book("bet365")]}],
            response_time_ms=1,
            quota_remaining=5,
            timestamp=past - timedelta(hours=1),
        )

        client.get_historical_odds = AsyncMock(return_value=response)
        plan = {
            "games": [
                {
                    "event_id": "op_live_ARS_CHE_h",
                    "odds_api_id": "apihex1",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "snapshots": [(past - timedelta(hours=1)).isoformat()],
                }
            ]
        }

        async with BackfillExecutor(
            client=client,  # type: ignore[arg-type]
            session_factory=_Factory(session),
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            result = await executor.execute_plan(plan)

        assert result.successful_snapshots == 1
        assert [e.id for e in await _events(session)] == ["op_live_ARS_CHE_h"]
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert {(s.event_id, s.source, s.time_basis) for s in snapshots} == {
            ("op_live_ARS_CHE_h", Source.ODDS_API, TimeBasis.UPPER_BOUND)
        }
        assert {s.market for s in snapshots} == {Market.ONE_X_TWO, Market.TOTALS}
