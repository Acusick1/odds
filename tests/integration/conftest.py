"""
Fixtures for integration tests.

This module contains shared fixtures for integration testing.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import EllipsisType
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from odds_analytics import backfill_executor
from odds_analytics.backfill_checks import ResponseCheck
from odds_analytics.backfill_executor import BackfillExecutor, BackfillResult
from odds_analytics.backfill_plan import Call, Fixture, Manifest, Target
from odds_core.config import APIConfig, Settings
from odds_core.models import Event, OddsSnapshot
from odds_core.time import utc_isoformat
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda import collection_calendar
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.jobs import fetch_betfair_exchange, fetch_oddsportal
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from scripts import ingest_football_data_uk
from scripts.ingest_football_data_uk import FdukLeague
from tests.test_helpers import (
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    FakeMlbStatsFetcher,
    ReplayHttpSession,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)


@pytest.fixture
def job_session(
    pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncSession:
    """The test session, handed to the OddsPortal and Betfair jobs as their session."""

    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return pglite_async_session

        async def __aexit__(self, *args: object) -> bool:
            return False

    for module in (fetch_oddsportal, fetch_betfair_exchange, collection_calendar):
        monkeypatch.setattr(module, "async_session_maker", _Maker())
    return pglite_async_session


@pytest.fixture
def mlb_schedule(monkeypatch: pytest.MonkeyPatch) -> type[FakeMlbStatsFetcher]:
    monkeypatch.setattr(FakeMlbStatsFetcher, "schedule", [])
    monkeypatch.setattr(FakeMlbStatsFetcher, "down", False)
    monkeypatch.setattr(FakeMlbStatsFetcher, "requests", 0)
    monkeypatch.setattr("odds_lambda.live_events.MlbStatsFetcher", FakeMlbStatsFetcher)
    return FakeMlbStatsFetcher


FDUK_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "football_data_uk"


@pytest.fixture
def fduk_epl() -> FdukLeague:
    return FdukLeague("E0", "soccer_epl", "EPL", ZoneInfo("Europe/London"))


@pytest.fixture
def fduk_session(
    pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncSession:
    """The test session, handed to the FDUK loader as its session, reading the fixture CSVs."""

    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return pglite_async_session

        async def __aexit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(ingest_football_data_uk, "async_session_maker", _Maker())
    monkeypatch.setattr(ingest_football_data_uk, "DATA_DIR", FDUK_FIXTURES)
    return pglite_async_session


@pytest.fixture
async def live_twin_with_rows(pglite_async_session):
    """An event owning a snapshot, paper trade and match brief."""
    from odds_core.match_brief_models import BriefDecision, MatchBrief
    from odds_core.paper_trade_models import PaperTrade

    session = pglite_async_session
    session.add(make_epl_event(TWIN_EVENT_ID))
    await session.flush()

    await store_tagged(
        session,
        make_tagged_snapshot(
            TWIN_EVENT_ID,
            TWIN_KICKOFF - timedelta(hours=3),
            [three_way_bookmaker("bet365", (2.5, 3.4, 2.9))],
            "oddsportal_live",
        ),
    )

    session.add_all(
        [
            PaperTrade(
                event_id=TWIN_EVENT_ID,
                market="h2h",
                selection="home",
                bookmaker="bet365",
                price_decimal=2.5,
                stake=10.0,
                bankroll_before=100.0,
            ),
            MatchBrief(
                event_id=TWIN_EVENT_ID,
                decision=BriefDecision.WATCHING,
                summary="s",
                brief_text="t",
            ),
        ]
    )
    await session.flush()
    return session


@pytest.fixture
async def stored_epl_event(pglite_async_session) -> Event:
    """One stored EPL event (Arsenal v Chelsea at ``TWIN_KICKOFF``) with no external ids."""
    event = make_epl_event(11)
    pglite_async_session.add(event)
    await pglite_async_session.flush()
    return event


@pytest.fixture
async def epl_snapshot(pglite_async_session, stored_epl_event) -> OddsSnapshot:
    """A stored Betfair 1x2 snapshot of ``stored_epl_event``, in the new shape, without prices."""
    snapshot = OddsSnapshot(
        event_id=stored_epl_event.id,
        snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
        source=Source.BETFAIR,
        market=Market.ONE_X_TWO,
        time_basis=TimeBasis.OBSERVED,
    )
    pglite_async_session.add(snapshot)
    await pglite_async_session.flush()
    return snapshot


@pytest.fixture
async def mixed_source_event(pglite_async_session):
    """An event with an Odds API snapshot and a later Polymarket snapshot carrying mid, ask and bid."""
    session = pglite_async_session
    kickoff = datetime(2026, 4, 18, 15, 0, tzinfo=UTC)
    event = make_epl_event(12, kickoff)
    session.add(event)
    api_time = kickoff - timedelta(hours=3)
    await store_tagged(
        session,
        make_tagged_snapshot(
            event.id,
            api_time,
            [
                three_way_bookmaker("bet365", (2.0, 3.5, 4.0)),
                three_way_bookmaker("pinnacle", (2.1, 3.4, 3.9)),
            ],
            None,
        ),
    )
    pm_time = kickoff - timedelta(hours=2)
    probabilities = {
        "yes_probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25},
        "best_ask": {"home": 0.52, "draw": 0.26, "away": 0.26},
        "best_bid": {"home": 0.48, "draw": 0.24, "away": 0.24},
    }
    await store_tagged(
        session,
        make_tagged_snapshot(
            event.id,
            pm_time,
            [three_way_bookmaker("polymarket", (2.0, 4.0, 4.0), polymarket_meta=probabilities)],
            "polymarket",
        ),
    )
    await session.commit()
    return event


ACQUISITION_KEY = "purchased"


class _ReplayClient(TheOddsAPIClient):
    """The real client over a recorded HTTP session."""

    def __init__(self, api_key: str, http: ReplayHttpSession) -> None:
        super().__init__(api_key=api_key, base_url="http://example.invalid")
        self._http = http

    async def __aenter__(self) -> _ReplayClient:
        self.session = self._http  # type: ignore[assignment]
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class BackfillReplay:
    """Replays recorded historical responses, one per call time, through the real client and
    ``BackfillExecutor``, each call in its own committed session as production runs it."""

    def __init__(self, engine: AsyncEngine, session: AsyncSession) -> None:
        self._engine = engine
        self._session = session
        self._responses: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self._billed: int | None | EllipsisType = ...

    def bill(self, units: int | None) -> None:
        """Bill each later call ``units`` in ``x-requests-last`` (None: send no such header)."""
        self._billed = units

    def record(
        self,
        at: datetime,
        games: list[dict[str, Any]],
        *,
        snapshot: datetime | None = None,
        previous_gap: timedelta | None = timedelta(minutes=5),
        next_gap: timedelta | None = timedelta(minutes=5),
    ) -> None:
        """The response a call at ``at`` gets: the provider's snapshot, by default 5 minutes
        earlier, with ``games`` and the neighbouring snapshots ``previous_gap`` before and
        ``next_gap`` after it (None leaves that timestamp out)."""
        taken = snapshot or at - timedelta(minutes=5)
        response: dict[str, Any] = {"timestamp": utc_isoformat(taken), "data": games}
        if previous_gap is not None:
            response["previous_timestamp"] = utc_isoformat(taken - previous_gap)
        if next_gap is not None:
            response["next_timestamp"] = utc_isoformat(taken + next_gap)
        self._responses[utc_isoformat(at)] = response

    @staticmethod
    def call(manifest: Manifest, at: datetime, *serves: Event) -> Call:
        """A planned call at ``at`` serving the first band of ``manifest`` for ``serves``."""
        band = manifest.coverage[0]
        return Call(
            at,
            tuple(
                Target(Fixture(event.id, event.commence_time), band, at, at)
                for event in serves
                if event.id is not None
            ),
        )

    @asynccontextmanager
    async def _executor(
        self, manifest: Manifest, **options: Any
    ) -> AsyncIterator[BackfillExecutor]:
        """An executor building its client from the acquisition key, as a real run does."""
        headers = {"x-requests-remaining": "4900000", "x-requests-used": "100000"}
        billed = manifest.units_per_call if self._billed is ... else self._billed
        if billed is not None:
            headers["x-requests-last"] = str(billed)
        http = ReplayHttpSession(self._responses, headers)
        settings = Settings(
            api=APIConfig(
                keys=f"live,nba-live,{ACQUISITION_KEY}", acquisition_key_index=2, _env_file=None
            ),
            database={"url": "postgresql+asyncpg://localhost/test"},
        )
        try:
            with (
                patch.object(backfill_executor, "get_settings", return_value=settings),
                patch("odds_lambda.data_fetcher.get_settings", return_value=settings),
                patch.object(
                    backfill_executor,
                    "TheOddsAPIClient",
                    lambda api_key: _ReplayClient(api_key, http),
                ),
            ):
                async with BackfillExecutor(
                    manifest,
                    session_factory=lambda: AsyncSession(self._engine, expire_on_commit=False),
                    rate_limit_seconds=0,
                    **options,
                ) as executor:
                    yield executor
        finally:
            self.requests += http.requests
            self._session.expunge_all()

    async def run(
        self, manifest: Manifest, calls: list[Call], *, skip_existing: bool = True, **options: Any
    ) -> BackfillResult:
        """Run ``calls``. Reports each request in ``requests``."""
        async with self._executor(manifest, skip_existing=skip_existing, **options) as executor:
            return await executor.execute(calls)

    async def probe(self, manifest: Manifest) -> list[ResponseCheck]:
        async with self._executor(manifest) as executor:
            return await executor.probe()


@pytest.fixture
async def replay(
    pglite_async_engine: AsyncEngine, pglite_async_session: AsyncSession
) -> AsyncGenerator[BackfillReplay, None]:
    yield BackfillReplay(pglite_async_engine, pglite_async_session)
