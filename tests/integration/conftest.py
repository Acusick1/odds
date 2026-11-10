"""
Fixtures for integration tests.

This module contains shared fixtures for integration testing.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from odds_core.models import Event, OddsSnapshot
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda import collection_calendar
from odds_lambda.jobs import fetch_betfair_exchange, fetch_oddsportal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from scripts import ingest_football_data_uk
from scripts.ingest_football_data_uk import FdukLeague
from tests.integration.backfill_replay_helpers import BackfillReplay
from tests.test_helpers import (
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    FakeMlbStatsFetcher,
    apply_odds_web_role,
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


@pytest.fixture
async def replay(
    pglite_async_engine: AsyncEngine, pglite_async_session: AsyncSession
) -> AsyncGenerator[BackfillReplay, None]:
    yield BackfillReplay(pglite_async_engine, pglite_async_session)


@pytest.fixture
async def odds_web_session(pglite_async_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """The suite's empty database with the ``odds_web`` role and its grants applied."""
    await apply_odds_web_role(pglite_async_session)
    yield pglite_async_session


CrossMarketRun = Callable[..., Awaitable[tuple[int, int | None]]]


@pytest.fixture
def add_cross_market_run(pglite_async_session: AsyncSession) -> CrossMarketRun:
    """Store one OddsPortal run of event 1's 1x2 and totals 2.5 as one payload and return the
    ids of its 1x2 and totals snapshots, stored at ``created_at`` when given; ``totals=False`` leaves the totals market out. The
    retail home price is 2.2 against an exchange fair of 0.5, so ``exchange_gap`` bets it."""
    from odds_lambda.storage.shaped_snapshots import store_payload

    from tests.test_helpers import betfair_exchange_bookmaker, totals_book

    async def add(
        minutes_before_t: float = 5,
        *,
        totals: bool = True,
        kickoff: datetime | None = None,
        created_at: datetime | None = None,
    ) -> tuple[int, int | None]:
        session = pglite_async_session
        kickoff = kickoff or datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
        if await session.get(Event, 1) is None:
            session.add(make_epl_event(1, kickoff))
            await session.flush()
        retail = three_way_bookmaker("bet365", (2.2, 3.0, 3.0))
        exchange = betfair_exchange_bookmaker((2.0, 4.0, 4.0))
        books = [retail, exchange]
        if totals:
            retail["markets"] += totals_book("bet365", {2.5: (1.9, 1.9)})["markets"]
            books.append(totals_book("betfair_exchange", {2.5: (2.0, 2.0)}))
        payload_id = await store_payload(session, [{"bookmakers": books}])
        snapshot = make_tagged_snapshot(
            1,
            kickoff - timedelta(minutes=60 + minutes_before_t),
            books,
            "oddsportal_live",
            payload_id=payload_id,
        )
        snapshot.created_at = created_at
        first = await store_tagged(session, snapshot)
        await session.commit()
        rows = (
            await session.execute(
                select(OddsSnapshot.id, OddsSnapshot.market).where(
                    OddsSnapshot.payload_id == payload_id
                )
            )
        ).all()
        by_market = {market: snapshot_id for snapshot_id, market in rows}
        assert first.id is not None
        return by_market[Market.ONE_X_TWO], by_market.get(Market.TOTALS)

    return add


AddMatch = Callable[..., Awaitable[Event]]
# When a test match was stored, before every decision and run time the tests use.
MATCH_STORED_AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def add_match(pglite_async_session: AsyncSession) -> AddMatch:
    """Store an EPL match with the given kick-off. ``score=(home, away)`` makes it FINAL with
    that result, completed at ``completed_at`` (the kick-off plus 2 hours by default);
    ``stored_at`` is its ``created_at`` and ``sport_key`` its league. Other fields go to ``make_epl_event``."""

    async def add(
        event_id: int,
        kickoff: datetime,
        home_team: str = "Arsenal",
        away_team: str = "Chelsea",
        *,
        score: tuple[int, int] | None = None,
        completed_at: datetime | None = None,
        stored_at: datetime = MATCH_STORED_AT,
        sport_key: str = "soccer_epl",
        **fields: object,
    ) -> Event:
        if score is not None:
            fields.update(
                status="final",
                home_score=score[0],
                away_score=score[1],
                completed_at=completed_at or kickoff + timedelta(hours=2),
            )
        event = make_epl_event(
            event_id, kickoff, home_team, away_team, created_at=stored_at, **fields
        )
        event.sport_key = sport_key
        pglite_async_session.add(event)
        await pglite_async_session.flush()
        return event

    return add
