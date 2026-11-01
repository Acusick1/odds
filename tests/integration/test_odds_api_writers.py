"""The Odds API writers resolve games through the event resolver and store the new shape (#687)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_analytics.backfill_plan import Call, Fixture, Manifest, Target
from odds_core.api_models import (
    ApiEvent,
    EventsResponse,
    OddsResponse,
    parse_api_event,
)
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    FetchLog,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.time import parse_api_datetime
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda import odds_api_events
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.event_sync import EventSyncService
from odds_lambda.ingestion import OddsIngestionService
from odds_lambda.jobs import fetch_scores
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.odds_api_events import find_event_by_odds_api_id
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import StubOddsClient, game_pk_of, make_epl_event

KICKOFF = (datetime.now(UTC) + timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
MLB = "baseball_mlb"
FIXTURES = Path(__file__).parents[1] / "fixtures"


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
    events = [parse_api_event(item) for item in listed]
    client = StubOddsClient(
        OddsResponse([], [], 0, None, None, datetime.now(UTC)),
        EventsResponse(events, 0, None, None, datetime.now(UTC)),
    )
    service = EventSyncService(client=client, session_factory=_Factory(session))  # type: ignore[arg-type]
    return service.sync_sport(sport_key)


async def _events(session: AsyncSession) -> list[Event]:
    return list((await session.execute(select(Event).order_by(Event.id))).scalars())


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


async def _holding_api_id(session: AsyncSession, event: Event, api_id: str) -> Event:
    """Store ``event`` holding ``api_id`` as its Odds API id."""
    session.add(event)
    await session.flush()
    session.add(EventExternalId(source=Source.ODDS_API, external_id=api_id, event_id=event.id))
    await session.flush()
    return event


@pytest.mark.asyncio
class TestEventSync:
    async def test_new_game_creates_an_event_recorded_by_the_api_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await _sync(pglite_async_session, [_listed("apihex1")])

        assert (result.inserted, result.updated) == (1, 0)
        (event,) = await _events(pglite_async_session)
        assert event.status == EventStatus.SCHEDULED
        assert await _odds_api_ids(pglite_async_session) == {event.id: "apihex1"}

    async def test_game_another_source_stored_attaches_to_that_event(
        self, pglite_async_session: AsyncSession
    ) -> None:
        stored = make_epl_event(1, KICKOFF)
        pglite_async_session.add(stored)
        await pglite_async_session.flush()

        result = await _sync(pglite_async_session, [_listed("apihex1", KICKOFF)])

        assert (result.inserted, result.updated) == (0, 1)
        (event,) = await _events(pglite_async_session)
        assert event.id == 1
        assert await _odds_api_ids(pglite_async_session) == {event.id: "apihex1"}

    async def test_event_holding_the_api_id_is_found_after_the_api_moves_it(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await _holding_api_id(pglite_async_session, make_epl_event(1, KICKOFF), "apihex1")
        moved = KICKOFF + timedelta(days=3)

        result = await _sync(pglite_async_session, [_listed("apihex1", moved)])

        assert (result.inserted, result.updated) == (0, 1)
        (event,) = await _events(pglite_async_session)
        assert event.id == 1
        assert event.commence_time == moved
        assert await _odds_api_ids(pglite_async_session) == {event.id: "apihex1"}

    async def test_postponed_event_the_api_lists_at_a_new_start_is_scheduled_again(
        self, pglite_async_session: AsyncSession
    ) -> None:
        stored = make_epl_event(1, KICKOFF, status=EventStatus.POSTPONED)
        await _holding_api_id(pglite_async_session, stored, "apihex1")

        await _sync(pglite_async_session, [_listed("apihex1", KICKOFF + timedelta(days=5))])

        (event,) = await _events(pglite_async_session)
        assert event.status == EventStatus.SCHEDULED
        assert event.commence_time == KICKOFF + timedelta(days=5)

    async def test_relisting_keeps_status_scores_and_teams(
        self, pglite_async_session: AsyncSession
    ) -> None:
        past = datetime.now(UTC) - timedelta(days=1)
        await _holding_api_id(
            pglite_async_session,
            make_epl_event(
                1,
                past,
                status=EventStatus.FINAL,
                home_score=2,
                away_score=1,
                completed_at=past + timedelta(hours=2),
            ),
            "apihex1",
        )

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

        async def _resolve(finder: Any, session: AsyncSession, event: ApiEvent) -> Any:
            if event.odds_api_id == "bad":
                raise RuntimeError("resolver down")
            return await real(finder, session, event)

        monkeypatch.setattr(odds_api_events, "resolve_odds_api_event", _resolve)

        with pytest.raises(RuntimeError, match="1 soccer_epl events failed to sync: bad"):
            await _sync(
                pglite_async_session,
                [_listed("bad", home="Fulham", away="Leeds"), _listed("good")],
            )

        assert [e.home_team for e in await _events(pglite_async_session)] == ["Arsenal"]

    async def test_mlb_game_attaches_to_the_event_holding_its_game_pk(
        self, pglite_async_session: AsyncSession, mlb_schedule: list[MlbGameResult]
    ) -> None:
        start = KICKOFF.replace(hour=23)
        held = Event(
            id=1,
            sport_key=MLB,
            sport_title="MLB",
            commence_time=start,
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(held)
        await pglite_async_session.flush()
        pglite_async_session.add(
            EventExternalId(source=Source.MLB_STATS_API, external_id="824101", event_id=held.id)
        )
        await pglite_async_session.flush()
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
        assert event.id == 1
        assert await game_pk_of(pglite_async_session, event) == 824101
        assert await _odds_api_ids(pglite_async_session) == {event.id: "apimlb1"}

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
            events=[parse_api_event(listed)],
            raw_events_data=[payload],
            response_time_ms=10,
            quota_remaining=100,
            quota_used=400,
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
        assert len({s.payload_id for s in snapshots}) == 1
        first, second = snapshots
        prices = (await session.execute(select(Price))).scalars().all()
        assert sorted(p.snapshot_id for p in prices) == sorted([first.id] * 6 + [second.id] * 4)
        payload = await session.get(SnapshotPayload, first.payload_id)
        assert payload is not None and payload.payload[0]["id"] == "apihex1"

    async def test_game_another_source_stored_gets_its_snapshot_on_that_event(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event(1, KICKOFF))
        await session.flush()

        await self._ingest(session, mock_settings, _listed("apihex1"), [_book("bet365")])

        events = await _events(session)
        assert [e.id for e in events] == [1]
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert {s.event_id for s in snapshots} == {1}
        assert await _odds_api_ids(session) == {events[0].id: "apihex1"}

    async def test_dropped_mlb_game_is_not_a_failure_and_writes_no_payload(
        self,
        pglite_async_session: AsyncSession,
        mock_settings: Any,
        mlb_schedule: list[MlbGameResult],
    ) -> None:
        session = pglite_async_session
        start = KICKOFF.replace(hour=23)
        listed = _listed(
            "apimlb3",
            start,
            "New York Yankees",
            "Boston Red Sox",
            sport_key=MLB,
            sport_title="MLB",
        )
        mlb_schedule.append(_mlb_game(900, start, "Chicago Cubs", "St. Louis Cardinals"))
        mock_settings.data_collection.bookmakers[MLB] = ["bet365"]
        response = OddsResponse(
            events=[parse_api_event(listed)],
            raw_events_data=[{**listed, "bookmakers": [_book("bet365")]}],
            response_time_ms=10,
            quota_remaining=100,
            quota_used=400,
            timestamp=datetime.now(UTC),
        )
        service = OddsIngestionService(
            client=StubOddsClient(response),  # type: ignore[arg-type]
            settings=mock_settings,
            session_factory=_Factory(session),
        )

        result = await service.ingest_sport(MLB)

        assert result.failures == []
        assert len(result.dropped_games) == 1
        assert result.success
        assert (await session.execute(select(SnapshotPayload))).scalars().all() == []
        log = (await session.execute(select(FetchLog))).scalar_one()
        assert log.success is True

    async def test_payload_is_written_once_for_a_run_of_events(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session
        listed = [
            _listed("apihex1"),
            _listed("apihex2", KICKOFF + timedelta(days=1), "Fulham", "Leeds"),
        ]
        response = OddsResponse(
            events=[parse_api_event(item) for item in listed],
            raw_events_data=[
                {**item, "bookmakers": [_book("bet365", item["home_team"], item["away_team"])]}
                for item in listed
            ],
            response_time_ms=10,
            quota_remaining=100,
            quota_used=400,
            timestamp=datetime.now(UTC),
        )
        mock_settings.data_collection.bookmakers["soccer_epl"] = ["bet365"]
        service = OddsIngestionService(
            client=StubOddsClient(response),  # type: ignore[arg-type]
            settings=mock_settings,
            session_factory=_Factory(session),
        )

        await service.ingest_sport("soccer_epl")

        assert len((await session.execute(select(SnapshotPayload))).scalars().all()) == 1
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert len(snapshots) == 4
        assert len({s.payload_id for s in snapshots}) == 1

    async def test_repeated_fetch_at_one_time_stores_each_market_once(
        self, pglite_async_session: AsyncSession, mock_settings: Any
    ) -> None:
        session = pglite_async_session
        listed = _listed("apihex1")
        response = OddsResponse(
            events=[parse_api_event(listed)],
            raw_events_data=[{**listed, "bookmakers": [_book("bet365")]}],
            response_time_ms=10,
            quota_remaining=100,
            quota_used=400,
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
        assert len((await session.execute(select(Price))).scalars().all()) == 5


@pytest.mark.asyncio
class TestScores:
    async def test_id_is_found_by_the_recorded_external_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _sync(session, [_listed("apihex1")])
        stored = make_epl_event(2, KICKOFF + timedelta(days=9), "Fulham", "Leeds")
        session.add(stored)
        await session.flush()
        await _sync(session, [_listed("apihex2", KICKOFF + timedelta(days=9), "Fulham", "Leeds")])

        found = await find_event_by_odds_api_id(session, "apihex2")

        assert found is not None and found.id == 2

    async def test_reissued_api_id_replaces_the_recorded_one_and_is_logged(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _sync(session, [_listed("oldid")])

        await _sync(session, [_listed("newid")])

        (event,) = await _events(session)
        assert await _odds_api_ids(session) == {event.id: "newid"}
        found = await find_event_by_odds_api_id(session, "newid")
        assert found is not None and found.id == event.id
        replaced = (
            (
                await session.execute(
                    select(DataQualityLog).where(
                        DataQualityLog.issue_type == "external_id_replaced"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert [log.raw_data["old_external_id"] for log in replaced] == ["oldid"]

    async def test_an_event_without_a_recorded_api_id_is_not_found(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event(1))
        await session.flush()

        assert await find_event_by_odds_api_id(session, "apihex1") is None

    async def test_job_settles_the_event_the_api_id_names(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = pglite_async_session
        past = datetime.now(UTC) - timedelta(hours=6)
        session.add(make_epl_event(1, past))
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

        result = await fetch_scores.update_scores("soccer_epl")

        assert result == fetch_scores.ScoresUpdate(listed=2, updated=1)

        event = await session.get(Event, 1, populate_existing=True)
        assert event is not None
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 1)


class _RecordedSession:
    """Serves one recorded historical response through the client's HTTP layer."""

    def __init__(self, payload: dict[str, Any], headers: dict[str, str]) -> None:
        self._payload = payload
        self._headers = headers
        self.status = 200
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: Any) -> _RecordedSession:
        self.requests.append({"url": url, **params})
        return self

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Any:
        return self._payload

    async def __aenter__(self) -> _RecordedSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _manifest() -> Manifest:
    return Manifest.model_validate(
        {
            "sport": "soccer_epl",
            "seasons": {
                "2025-26": {"start": "2025-07-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"}
            },
            "markets": [{"key": "h2h"}, {"key": "totals", "line": 2.5}],
            "regions": ["uk"],
            "coverage": [
                {"min_lead_minutes": 1, "max_lead_minutes": 120, "max_input_age_minutes": 10}
            ],
            "snapshot_grid": [{"start": "2020-06-06T00:00:00Z", "step_minutes": 10}],
            "probe": {
                "snapshots": ["2026-04-18T10:00:00Z"],
                "min_fixture_lead_hours": 12,
                "required_bookmakers": {},
                "units_per_call": 20,
            },
        }
    )


@pytest.mark.asyncio
class TestBackfill:
    @pytest.mark.parametrize("reverse", [False, True], ids=["listed", "reversed"])
    async def test_one_call_stores_every_fixture_it_returns_at_the_snapshot_time(
        self, pglite_async_session: AsyncSession, reverse: bool
    ) -> None:
        session = pglite_async_session
        recorded = json.loads((FIXTURES / "odds_api_historical_epl.json").read_text())
        listed = [{k: v for k, v in e.items() if k != "bookmakers"} for e in recorded["data"]]
        kickoffs = [parse_api_datetime(e["commence_time"]) for e in listed]
        session.add(make_epl_event(1, kickoffs[0]))
        session.add(make_epl_event(2, kickoffs[1], "Liverpool", "Everton"))
        await session.flush()
        await _sync(session, listed)

        manifest = _manifest()
        requested = datetime(2026, 4, 18, 10, 0, tzinfo=UTC)
        band = manifest.coverage[0]
        call = Call(
            requested,
            tuple(
                Target(Fixture(event_id, kickoff), band, requested, requested)
                for event_id, kickoff in ((1, kickoffs[0]), (2, kickoffs[1]))
            ),
        )
        # The fixtures' lock order must not depend on the order the API lists them in.
        served = recorded | {"data": recorded["data"][::-1] if reverse else recorded["data"]}
        http = _RecordedSession(
            served,
            {"x-requests-last": "20", "x-requests-remaining": "12345", "x-requests-used": "655"},
        )
        client = TheOddsAPIClient(api_key="k", base_url="http://example.invalid")
        client.session = http  # type: ignore[assignment]

        async with BackfillExecutor(
            manifest, client=client, session_factory=_Factory(session), rate_limit_seconds=0
        ) as executor:
            result = await executor.execute([call])

        assert [r["date"] for r in http.requests] == ["2026-04-18T10:00:00Z"]
        assert http.requests[0]["regions"] == "uk"
        assert "bookmakers" not in http.requests[0]
        assert http.requests[0]["markets"] == "h2h,totals"
        assert [e.id for e in await _events(session)] == [1, 2]
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert {(s.event_id, s.snapshot_time, s.source, s.time_basis) for s in snapshots} == {
            (
                event_id,
                parse_api_datetime(recorded["timestamp"]),
                Source.ODDS_API,
                TimeBasis.OBSERVED,
            )
            for event_id in (1, 2)
        }
        assert {s.market for s in snapshots} == {Market.ONE_X_TWO, Market.TOTALS}
        logs = (await session.execute(select(FetchLog))).scalars().all()
        assert [
            (log.sport_key, log.events_count, log.api_quota_remaining, log.api_quota_used)
            for log in logs
        ] == [("soccer_epl", 2, 12345, 655)]
        assert (result.calls_made, result.units_used, result.fixtures_stored) == (1, 20, 2)
        assert result.quota_remaining == 12345

        async with BackfillExecutor(
            manifest, client=client, session_factory=_Factory(session), rate_limit_seconds=0
        ) as executor:
            rerun = await executor.execute([call])

        assert (rerun.calls_made, rerun.calls_skipped) == (0, 1)
        assert len(http.requests) == 1
