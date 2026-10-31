"""Unit tests for BackfillExecutor."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from odds_analytics import backfill_executor
from odds_analytics.backfill_executor import BackfillExecutor, CallProgress
from odds_analytics.backfill_plan import Call, Fixture, Manifest, Target
from odds_core.api_models import HistoricalOddsResponse, parse_api_event
from odds_core.models import OddsSnapshot
from odds_core.vocabulary import Market, Source, TimeBasis
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import add_event, make_epl_event

T0 = datetime(2026, 4, 18, 10, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 4, 18, 15, 0, tzinfo=UTC)


def manifest(**overrides: Any) -> Manifest:
    data: dict[str, Any] = {
        "sport": "soccer_epl",
        "seasons": {"2025-26": {"start": "2025-07-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"}},
        "markets": [{"key": "h2h"}, {"key": "totals", "line": 2.5}],
        "bookmakers": ["pinnacle", "williamhill"],
        "coverage": [{"min_lead_minutes": 1, "max_lead_minutes": 120, "max_input_age_minutes": 10}],
        "snapshot_grid": [{"start": "2020-06-06T00:00:00Z", "step_minutes": 10}],
        "probe": {
            "snapshots": ["2026-04-18T10:00:00Z"],
            "min_fixture_lead_hours": 12,
            "required_bookmakers": {},
            "units_per_call": 20,
        },
    }
    return Manifest.model_validate(data | overrides)


def call(at: datetime, *event_ids: int) -> Call:
    m = manifest()
    band = m.coverage[0]
    return Call(at, tuple(Target(Fixture(i, KICKOFF), band, at, at) for i in event_ids))


def raw_event(api_id: str) -> dict[str, Any]:
    return {
        "id": api_id,
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "commence_time": KICKOFF.isoformat(),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [],
    }


def response(*api_ids: str, units: int | None = 20, quota: int | None = 1000) -> Any:
    raws = [raw_event(i) for i in api_ids]
    return HistoricalOddsResponse(
        events=[parse_api_event(r) for r in raws],
        raw_events_data=raws,
        response_time_ms=5,
        quota_remaining=quota,
        quota_used=4000,
        snapshot_time=T0 - timedelta(minutes=5),
        units_used=units,
    )


class FakeClient:
    quota_remaining: int | None = 950
    quota_used: int | None = 4050

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    async def get_historical_odds(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Factory:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> Factory:
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *args: object) -> bool:
        return False


class FakeWriter:
    stored: list[dict[str, Any]] = []
    fetch_logs: list[Any] = []
    fail_store: bool = False

    def __init__(self, session: AsyncSession) -> None:
        pass

    async def store_envelope(self, **kwargs: Any) -> Any:
        if self.fail_store:
            raise RuntimeError("store failed")
        self.stored.append(kwargs)
        return SimpleNamespace(snapshots_written=2)

    async def log_fetch(self, fetch_log: Any) -> Any:
        self.fetch_logs.append(fetch_log)
        return fetch_log


@pytest.fixture
def writer(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakeWriter]]:
    """Every returned game resolves to the event whose id is its position in the response."""
    FakeWriter.stored = []
    FakeWriter.fetch_logs = []
    FakeWriter.fail_store = False

    async def finder(sport_key: str, events: Any) -> None:
        return None

    async def resolve(finder: Any, session: Any, event: Any) -> Any:
        return SimpleNamespace(event=SimpleNamespace(id=int(event.odds_api_id[-1])))

    async def payload(session: Any, raw: Any) -> int:
        return 7

    monkeypatch.setattr(backfill_executor, "OddsWriter", FakeWriter)
    monkeypatch.setattr(backfill_executor, "odds_api_finder", finder)
    monkeypatch.setattr(backfill_executor, "resolve_odds_api_event", resolve)
    monkeypatch.setattr(backfill_executor, "store_payload", payload)
    yield FakeWriter


def executor(client: FakeClient, session: AsyncSession, **kwargs: Any) -> BackfillExecutor:
    return BackfillExecutor(
        manifest(),
        client=client,  # type: ignore[arg-type]
        session_factory=Factory(session),
        rate_limit_seconds=0,
        **kwargs,
    )


async def seed_snapshot(
    session: AsyncSession,
    event_id: int,
    at: datetime,
    source: Source,
    markets: tuple[Market, ...] = (Market.ONE_X_TWO, Market.TOTALS),
) -> None:
    for market in markets:
        session.add(
            OddsSnapshot(
                event_id=event_id,
                snapshot_time=at,
                source=source,
                market=market,
                time_basis=TimeBasis.OBSERVED,
            )
        )
    await session.flush()


@pytest.mark.asyncio
class TestBackfillExecutor:
    async def test_the_request_comes_from_the_manifest(self, test_session, writer) -> None:
        client = FakeClient(response("e1"))

        async with executor(client, test_session, skip_existing=False) as ex:
            await ex.execute([call(T0, 1)])

        assert client.requests == [
            {
                "sport": "soccer_epl",
                "date": "2026-04-18T10:00:00Z",
                "markets": ["h2h", "totals"],
                "regions": None,
                "bookmakers": ["pinnacle", "williamhill"],
            }
        ]

    async def test_stores_every_returned_event_as_observed_at_the_snapshot_time(
        self, test_session, writer
    ) -> None:
        client = FakeClient(response("e1", "e2", units=20, quota=900))
        progress: list[CallProgress] = []

        async with executor(client, test_session, skip_existing=False) as ex:
            result = await ex.execute([call(T0, 1, 2)], progress.append)

        assert [(s["event_id"], s["snapshot_time"]) for s in writer.stored] == [
            (1, T0 - timedelta(minutes=5)),
            (2, T0 - timedelta(minutes=5)),
        ]
        assert {(s["shape"].source, s["shape"].time_basis) for s in writer.stored} == {
            (Source.ODDS_API, TimeBasis.OBSERVED)
        }
        assert [
            (f.sport_key, f.api_quota_remaining, f.api_quota_used) for f in writer.fetch_logs
        ] == [("soccer_epl", 900, 4000)]
        assert (result.fixtures_stored, result.snapshots_stored) == (2, 4)
        assert progress == [CallProgress(T0, "fetched", 2, 20, 900)]

    async def test_units_and_quota_follow_the_calls(self, test_session, writer) -> None:
        client = FakeClient(
            response("e1", units=20, quota=980), response("e1", units=20, quota=960)
        )

        async with executor(client, test_session, skip_existing=False) as ex:
            result = await ex.execute([call(T0, 1), call(T0 + timedelta(minutes=10), 1)])

        assert (result.calls_made, result.units_used, result.quota_remaining) == (2, 40, 960)

    async def test_skips_a_call_whose_fixtures_all_have_its_snapshot(
        self, test_session, writer
    ) -> None:
        for event_id in (1, 2):
            await add_event(test_session, make_epl_event(event_id, KICKOFF))
            await seed_snapshot(test_session, event_id, T0 - timedelta(minutes=5), Source.ODDS_API)
        client = FakeClient()

        async with executor(client, test_session) as ex:
            result = await ex.execute([call(T0, 1, 2)])

        assert (result.calls_made, result.calls_skipped) == (0, 1)
        assert client.requests == []

    async def test_calls_when_one_fixture_lacks_the_snapshot(self, test_session, writer) -> None:
        for event_id in (1, 2):
            await add_event(test_session, make_epl_event(event_id, KICKOFF))
        await seed_snapshot(test_session, 1, T0 - timedelta(minutes=5), Source.ODDS_API)
        client = FakeClient(response("e1", "e2"))

        async with executor(client, test_session) as ex:
            result = await ex.execute([call(T0, 1, 2)])

        assert (result.calls_made, result.calls_skipped) == (1, 0)

    async def test_calls_when_a_requested_market_is_missing(self, test_session, writer) -> None:
        await add_event(test_session, make_epl_event(1, KICKOFF))
        await seed_snapshot(
            test_session, 1, T0 - timedelta(minutes=5), Source.ODDS_API, (Market.ONE_X_TWO,)
        )
        client = FakeClient(response("e1"))

        async with executor(client, test_session) as ex:
            result = await ex.execute([call(T0, 1)])

        assert (result.calls_made, result.calls_skipped) == (1, 0)

    @pytest.mark.parametrize(
        ("offset_minutes", "source"),
        [(10, Source.ODDS_API), (-1, Source.ODDS_API), (5, Source.POLYMARKET)],
    )
    async def test_a_snapshot_outside_the_step_or_of_another_source_does_not_count(
        self, test_session, writer, offset_minutes: int, source: Source
    ) -> None:
        await add_event(test_session, make_epl_event(1, KICKOFF))
        await seed_snapshot(test_session, 1, T0 - timedelta(minutes=offset_minutes), source)
        client = FakeClient(response("e1"))

        async with executor(client, test_session) as ex:
            result = await ex.execute([call(T0, 1)])

        assert result.calls_made == 1

    async def test_dry_run_makes_no_call(self, test_session, writer) -> None:
        client = FakeClient()
        progress: list[CallProgress] = []

        async with executor(client, test_session, dry_run=True) as ex:
            result = await ex.execute(
                [call(T0, 1), call(T0 + timedelta(minutes=10), 1)], progress.append
            )

        assert client.requests == []
        assert (result.would_fetch, result.calls_made, result.units_used) == (2, 0, 0)
        assert [p.status for p in progress] == ["would_fetch", "would_fetch"]
        assert writer.stored == []

    async def test_an_exception_stops_the_run(self, test_session, writer) -> None:
        client = FakeClient(response("e1"), RuntimeError("quota exhausted"), response("e1"))
        progress: list[CallProgress] = []
        calls = [call(T0 + timedelta(minutes=10 * i), 1) for i in range(3)]

        async with executor(client, test_session, skip_existing=False) as ex:
            with pytest.raises(RuntimeError, match="quota exhausted"):
                await ex.execute(calls, progress.append)

        assert len(client.requests) == 2
        assert [p.status for p in progress] == ["fetched", "failed"]
        assert progress[-1].message == "quota exhausted"

    async def test_a_failed_request_records_the_last_quota(self, test_session, writer) -> None:
        client = FakeClient(RuntimeError("429 Too Many Requests"))

        async with executor(client, test_session, skip_existing=False) as ex:
            with pytest.raises(RuntimeError):
                await ex.execute([call(T0, 1), call(T0 + timedelta(minutes=10), 1)])

        assert len(client.requests) == 1
        assert [
            (log.success, log.api_quota_remaining, log.api_quota_used) for log in writer.fetch_logs
        ] == [(False, 950, 4050)]
        assert "429" in writer.fetch_logs[0].error_message

    async def test_a_failed_store_records_the_quota_and_stops(self, test_session, writer) -> None:
        writer.fail_store = True
        client = FakeClient(response("e1", quota=900), response("e1"))
        client.quota_remaining = 900

        async with executor(client, test_session, skip_existing=False) as ex:
            with pytest.raises(RuntimeError, match="store failed"):
                await ex.execute([call(T0, 1), call(T0 + timedelta(minutes=10), 1)])

        assert len(client.requests) == 1
        assert [(log.success, log.api_quota_remaining) for log in writer.fetch_logs] == [
            (False, 900)
        ]
