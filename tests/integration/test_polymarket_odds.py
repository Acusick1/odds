"""Integration tests for the Polymarket odds backfill and live job."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
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
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.jobs import fetch_polymarket_odds
from odds_lambda.polymarket_odds import (
    MAX_HORIZON,
    MAX_STALENESS,
    Observation,
    grid_offsets,
)
from odds_lambda.polymarket_odds_backfill import backfill_sport
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.storage.event_resolver import event_lock_key
from odds_lambda.storage.shaped_snapshots import (
    SnapshotShape,
    store_payload,
    store_shaped_snapshot,
)
from sqlalchemy import select

from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    polymarket_bookmaker,
    store_tagged,
)

KICKOFF = datetime(2025, 12, 6, 15, 0, tzinfo=UTC)
SLUG = "epl-bou-che-2025-12-06"
PRICES = {"bou": 0.52, "draw": 0.26, "che": 0.22}


class FakeClient:
    def __init__(self, events: list[dict[str, Any]], series: dict[str, list[Observation]]) -> None:
        self.events = events
        self.series = series
        self.history_calls = 0

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_all_events(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return self.events

    async def get_price_series(
        self, token_id: str, start: datetime, end: datetime, fidelity: int = 1
    ) -> list[Observation]:
        self.history_calls += 1
        if token_id.startswith("boom-"):
            raise RuntimeError("history unavailable")
        return [(t, p) for t, p in self.series[token_id] if start <= t <= end]


class SessionMaker:
    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self) -> SessionMaker:
        return self

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


def full_history() -> dict[str, list[Observation]]:
    start = KICKOFF - MAX_HORIZON - 2 * MAX_STALENESS
    minutes = int((KICKOFF - start).total_seconds() // 60)
    return {
        f"yes-{suffix}": [(start + timedelta(minutes=i), p) for i in range(minutes + 1)]
        for suffix, p in PRICES.items()
    }


@pytest.fixture
async def finished_event(pglite_async_session: Any) -> Any:
    event = make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea", status=EventStatus.FINAL)
    pglite_async_session.add(event)
    await pglite_async_session.commit()
    return event


@pytest.fixture
def patch_backfill(
    monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any, gamma_event: Any
) -> FakeClient:
    client = FakeClient([gamma_event()], full_history())
    monkeypatch.setattr("odds_lambda.polymarket_odds_backfill.PolymarketClient", lambda: client)
    monkeypatch.setattr(
        "odds_lambda.polymarket_odds_backfill.async_session_maker",
        SessionMaker(pglite_async_session),
    )
    return client


class TestBackfill:
    async def test_writes_one_snapshot_per_grid_point(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        snapshots = (
            (
                await pglite_async_session.execute(
                    select(OddsSnapshot).order_by(OddsSnapshot.snapshot_time)
                )
            )
            .scalars()
            .all()
        )
        assert report.matches_attached == 1
        assert report.events_written == 1
        assert len(snapshots) == len(grid_offsets()) == report.snapshots_written
        assert {(s.source, s.market, s.time_basis) for s in snapshots} == {
            (Source.POLYMARKET, Market.ONE_X_TWO, TimeBasis.OBSERVED)
        }
        assert snapshots[-1].snapshot_time == KICKOFF
        payload = await pglite_async_session.get(SnapshotPayload, snapshots[-1].payload_id)
        assert payload is not None
        bookmakers = payload.payload["bookmakers"]
        assert [b["key"] for b in bookmakers] == ["polymarket"]
        assert [o["name"] for o in bookmakers[0]["markets"][0]["outcomes"]] == [
            "Bournemouth",
            "Draw",
            "Chelsea",
        ]
        prices = (
            (
                await pglite_async_session.execute(
                    select(Price).where(Price.snapshot_id == snapshots[-1].id)
                )
            )
            .scalars()
            .all()
        )
        assert {(p.bookmaker, p.side) for p in prices} == {("polymarket", Side.MID)}

    async def test_rerun_skips_events_with_history(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))
        calls = patch_backfill.history_calls

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert (report.events_skipped_existing, report.snapshots_written) == (1, 0)
        assert patch_backfill.history_calls == calls
        count = len((await pglite_async_session.execute(select(OddsSnapshot))).scalars().all())
        payloads = (await pglite_async_session.execute(select(SnapshotPayload))).scalars().all()
        assert count == len(payloads) == len(grid_offsets())

    async def test_an_event_with_only_live_snapshots_is_backfilled(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                finished_event.id,
                KICKOFF - timedelta(hours=7, minutes=1),
                [polymarket_bookmaker((0.5, 0.25, 0.25))],
                "polymarket",
            ),
        )
        await pglite_async_session.commit()

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert (report.events_skipped_existing, report.events_written) == (0, 1)

    async def test_a_live_snapshot_without_quotes_is_not_history(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        bookmaker = polymarket_bookmaker((0.5, 0.25, 0.25))
        bookmaker["polymarket_meta"] |= {"best_bid": {}, "best_ask": {}}
        envelope = {"bookmakers": [bookmaker]}
        payload_id = await store_payload(pglite_async_session, [envelope])
        await store_shaped_snapshot(
            pglite_async_session,
            finished_event,
            envelope,
            KICKOFF - timedelta(hours=7, minutes=1),
            SnapshotShape(Source.POLYMARKET, TimeBasis.OBSERVED, payload_id),
        )
        await pglite_async_session.commit()

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert (report.events_skipped_existing, report.events_written) == (0, 1)

    async def test_dry_run_writes_nothing(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1), dry_run=True)

        assert report.snapshots_written == len(grid_offsets())
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None
        assert (await pglite_async_session.execute(select(EventExternalId))).first() is None

    async def test_dry_run_releases_fixture_locks_between_matches(
        self,
        pglite_async_session: Any,
        finished_event: Any,
        patch_backfill: FakeClient,
        gamma_event: Any,
    ) -> None:
        fixtures = [
            ("Bournemouth", "Chelsea", "AFC Bournemouth", "Chelsea FC", "bou", "che"),
            ("Arsenal", "Everton", "Arsenal FC", "Everton FC", "ars", "eve"),
            ("Liverpool", "Fulham", "Liverpool FC", "Fulham FC", "liv", "ful"),
        ]
        fixtures.sort(key=lambda f: event_lock_key("soccer_epl", f[0], f[1]), reverse=True)
        events = []
        for index, (home, away, gamma_home, gamma_away, home_code, away_code) in enumerate(
            fixtures
        ):
            if home != "Bournemouth":
                pglite_async_session.add(
                    make_epl_event(10 + index, KICKOFF, home, away, status=EventStatus.FINAL)
                )
            slug = f"epl-{home_code}-{away_code}-2025-12-06"
            gamma = gamma_event(
                slug=slug,
                home=gamma_home,
                away=gamma_away,
                quotes={away_code: (0.20, 0.22), "draw": (0.24, 0.26), home_code: (0.52, 0.54)},
            )
            gamma["id"] = str(900 + index)
            for market in gamma["markets"]:
                market["clobTokenIds"] = market["clobTokenIds"].replace("yes-", f"yes-{home_code}-")
            patch_backfill.series.update(
                {
                    f"yes-{home_code}-{suffix}": observations
                    for suffix, observations in zip(
                        (home_code, "draw", away_code), full_history().values(), strict=True
                    )
                }
            )
            events.append(gamma)
        await pglite_async_session.commit()
        patch_backfill.events = events

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1), dry_run=True)

        assert report.errors == []
        assert (report.matches_attached, report.events_written) == (3, 3)
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None

    async def test_cutoff_is_the_earlier_kickoff(
        self,
        pglite_async_session: Any,
        finished_event: Any,
        patch_backfill: FakeClient,
        gamma_event: Any,
    ) -> None:
        early_start = KICKOFF - timedelta(minutes=15)
        patch_backfill.events = [gamma_event(start="2025-12-06T14:45:00Z")]

        await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        times = (await pglite_async_session.execute(select(OddsSnapshot.snapshot_time))).scalars()
        assert max(times) <= early_start

    async def test_event_kicking_off_before_polymarket_start_ends_at_the_event(
        self, pglite_async_session: Any, patch_backfill: FakeClient, gamma_event: Any
    ) -> None:
        early_event = make_epl_event(
            1,
            KICKOFF - timedelta(minutes=30),
            "Bournemouth",
            "Chelsea",
            status=EventStatus.FINAL,
        )
        pglite_async_session.add(early_event)
        await pglite_async_session.commit()
        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        times = (await pglite_async_session.execute(select(OddsSnapshot.snapshot_time))).scalars()
        assert report.events_written == 1
        assert max(times) == early_event.commence_time

    async def test_snapshots_carry_source_market_basis_and_mid_prices(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        snapshots = (await pglite_async_session.execute(select(OddsSnapshot))).scalars().all()
        prices = (await pglite_async_session.execute(select(Price))).scalars().all()
        assert {(s.source, s.market, s.time_basis) for s in snapshots} == {
            (Source.POLYMARKET, Market.ONE_X_TWO, TimeBasis.OBSERVED)
        }
        payloads = (await pglite_async_session.execute(select(SnapshotPayload))).scalars().all()
        assert all(s.payload_id is not None for s in snapshots)
        assert {s.payload_id for s in snapshots} == {p.id for p in payloads}
        assert len(payloads) == len(snapshots)
        assert len(prices) == 3 * len(snapshots)
        assert {(p.bookmaker, p.side) for p in prices} == {("polymarket", Side.MID)}
        home = [p for p in prices if p.outcome is PriceOutcome.HOME]
        assert {round(float(p.price), 4) for p in home} == {round(1 / 0.52, 4)}

    async def test_attaches_to_the_event_and_records_the_polymarket_id(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        held = await pglite_async_session.get(EventExternalId, (Source.POLYMARKET, "900"))
        assert held is not None
        assert held.event_id == finished_event.id

    async def test_creates_no_event_for_an_unmatched_match(
        self, pglite_async_session: Any, patch_backfill: FakeClient
    ) -> None:
        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert report.unmatched_pm_tickers == [SLUG]
        assert (await pglite_async_session.execute(select(Event))).first() is None
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None

    async def test_skips_an_event_that_is_not_finished(
        self, pglite_async_session: Any, patch_backfill: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert (report.matches_out_of_scope, report.events_written) == (1, 0)
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None

    async def test_a_failing_match_leaves_the_others_written(
        self,
        pglite_async_session: Any,
        finished_event: Any,
        patch_backfill: FakeClient,
        gamma_event: Any,
    ) -> None:
        other_kickoff = KICKOFF + timedelta(days=1)
        pglite_async_session.add(
            make_epl_event(2, other_kickoff, "Arsenal", "Everton", status=EventStatus.FINAL)
        )
        await pglite_async_session.commit()
        second = gamma_event(
            slug="epl-ars-eve-2025-12-07",
            home="Arsenal FC",
            away="Everton FC",
            start="2025-12-07T15:00:00Z",
            quotes={"eve": (0.20, 0.22), "draw": (0.24, 0.26), "ars": (0.52, 0.54)},
        )
        second["id"] = "901"
        for market in second["markets"]:
            market["clobTokenIds"] = market["clobTokenIds"].replace("yes-", "boom-")
        patch_backfill.events = [gamma_event(), second]

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        owners = set(
            (await pglite_async_session.execute(select(OddsSnapshot.event_id))).scalars().all()
        )
        assert owners == {1}
        assert (report.events_written, len(report.errors)) == (1, 1)
        assert report.errors[0].startswith("epl-ars-eve-2025-12-07:")

    async def test_event_before_since_is_out_of_scope(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        report = await backfill_sport("soccer_epl", since=KICKOFF + timedelta(days=1))

        assert (report.matches_out_of_scope, report.events_written) == (1, 0)

    async def test_event_without_a_match_is_reported(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        patch_backfill.events = []

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert report.unmatched_event_ids == [1]
        assert report.snapshots_written == 0


NBA_KICKOFF = datetime(2026, 10, 31, 2, 0, tzinfo=UTC)


def nba_history() -> dict[str, list[Observation]]:
    start = NBA_KICKOFF - MAX_HORIZON - 2 * MAX_STALENESS
    minutes = int((NBA_KICKOFF - start).total_seconds() // 60)
    return {
        token: [(start + timedelta(minutes=i), p) for i in range(minutes + 1)]
        for token, p in (("tok-away", 0.31), ("tok-home", 0.69))
    }


def make_nba_event(event_id: int, status: EventStatus = EventStatus.FINAL) -> Event:
    return Event(
        id=event_id,
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=NBA_KICKOFF,
        home_team="Golden State Warriors",
        away_team="Los Angeles Lakers",
        status=status,
        home_score=110 if status is EventStatus.FINAL else None,
        away_score=104 if status is EventStatus.FINAL else None,
    )


class TestNbaBackfill:
    @pytest.fixture
    def patch_nba(
        self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any, gamma_nba_event: Any
    ) -> FakeClient:
        client = FakeClient([gamma_nba_event()], nba_history())
        monkeypatch.setattr("odds_lambda.polymarket_odds_backfill.PolymarketClient", lambda: client)
        monkeypatch.setattr(
            "odds_lambda.polymarket_odds_backfill.async_session_maker",
            SessionMaker(pglite_async_session),
        )
        return client

    async def test_writes_h2h_snapshots_with_two_mid_prices(
        self, pglite_async_session: Any, patch_nba: FakeClient
    ) -> None:
        pglite_async_session.add(make_nba_event(1))
        await pglite_async_session.commit()

        report = await backfill_sport("basketball_nba", since=NBA_KICKOFF - timedelta(days=1))

        snapshots = (await pglite_async_session.execute(select(OddsSnapshot))).scalars().all()
        prices = (await pglite_async_session.execute(select(Price))).scalars().all()
        assert (report.events_written, report.snapshots_written) == (1, len(grid_offsets()))
        assert {(s.source, s.market, s.time_basis) for s in snapshots} == {
            (Source.POLYMARKET, Market.H2H, TimeBasis.OBSERVED)
        }
        assert len(prices) == 2 * len(snapshots)
        home = {round(1 / float(p.price), 4) for p in prices if p.outcome is PriceOutcome.HOME}
        away = {round(1 / float(p.price), 4) for p in prices if p.outcome is PriceOutcome.AWAY}
        assert (home, away) == ({0.69}, {0.31})

    async def test_rerun_adds_nothing(
        self, pglite_async_session: Any, patch_nba: FakeClient
    ) -> None:
        pglite_async_session.add(make_nba_event(1))
        await pglite_async_session.commit()
        await backfill_sport("basketball_nba", since=NBA_KICKOFF - timedelta(days=1))

        report = await backfill_sport("basketball_nba", since=NBA_KICKOFF - timedelta(days=1))

        assert (report.events_skipped_existing, report.snapshots_written) == (1, 0)

    async def test_creates_no_event_for_a_game_without_one(
        self, pglite_async_session: Any, patch_nba: FakeClient
    ) -> None:
        report = await backfill_sport("basketball_nba", since=NBA_KICKOFF - timedelta(days=1))

        assert report.unmatched_pm_tickers == ["nba-lal-gsw-2026-10-30"]
        assert (await pglite_async_session.execute(select(Event))).first() is None

    async def test_live_job_stores_a_two_outcome_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any, gamma_nba_event: Any
    ) -> None:
        client = FakeClient([gamma_nba_event()], {})
        monkeypatch.setattr(fetch_polymarket_odds, "PolymarketClient", lambda: client)
        monkeypatch.setattr(
            fetch_polymarket_odds, "async_session_maker", SessionMaker(pglite_async_session)
        )
        pglite_async_session.add(make_nba_event(1, EventStatus.SCHEDULED))
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport(
            "basketball_nba", NBA_KICKOFF - timedelta(hours=5)
        )

        prices = (await pglite_async_session.execute(select(Price))).scalars().all()
        assert stats.snapshots_stored == 1
        assert len(prices) == 6
        assert {p.outcome for p in prices} == {PriceOutcome.HOME, PriceOutcome.AWAY}


class TestLiveIngestion:
    @pytest.fixture
    def patch_live(
        self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any, gamma_event: Any
    ) -> FakeClient:
        client = FakeClient([gamma_event()], {})
        monkeypatch.setattr(fetch_polymarket_odds, "PolymarketClient", lambda: client)
        monkeypatch.setattr(
            fetch_polymarket_odds, "async_session_maker", SessionMaker(pglite_async_session)
        )
        return client

    async def test_stores_snapshot_with_both_halves(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        now = KICKOFF - timedelta(hours=5)

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)

        snapshot = (await pglite_async_session.execute(select(OddsSnapshot))).scalar_one()
        prices = (await pglite_async_session.execute(select(Price))).scalars().all()
        assert stats.snapshots_stored == 1
        assert (snapshot.source, snapshot.market) == (Source.POLYMARKET, Market.ONE_X_TWO)
        assert {p.bookmaker for p in prices} == {"polymarket"}
        assert {p.side for p in prices} == {Side.MID, Side.BACK, Side.LAY}
        assert len(prices) == 9

    async def test_skips_matches_already_started(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF + timedelta(hours=1))

        assert stats.matches_seen == 0
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None

    async def test_counts_matches_without_an_event(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.matches_seen, stats.matches_unattached, stats.snapshots_stored) == (1, 1, 0)
        assert (await pglite_async_session.execute(select(Event))).first() is None

    async def test_rejects_prices_off_sum(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        for market in patch_live.events[0]["markets"]:
            market["bestBid"], market["bestAsk"] = 0.49, 0.51

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.sum_rejected, stats.snapshots_stored) == (1, 0)

    async def test_counts_matches_without_prices(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        patch_live.events[0]["markets"][0]["bestBid"] = None

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.prices_missing, stats.snapshots_stored) == (1, 0)

    async def test_stores_quotes_in_meta(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        snapshot = (await pglite_async_session.execute(select(OddsSnapshot))).scalar_one()
        payload = await pglite_async_session.get(SnapshotPayload, snapshot.payload_id)
        assert payload is not None
        [envelope] = payload.payload
        meta = envelope["bookmakers"][0]["polymarket_meta"]
        assert meta["best_bid"]["home"] == 0.52
        assert meta["best_ask"]["home"] == 0.54

    async def test_skips_wide_spreads(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        patch_live.events[0]["markets"][0]["bestBid"] = 0.10
        patch_live.events[0]["markets"][0]["bestAsk"] = 0.30

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.wide_spread, stats.snapshots_stored) == (1, 0)

    async def test_skips_events_that_have_kicked_off_before_polymarket_start(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(
            make_epl_event(1, KICKOFF - timedelta(hours=1), "Bournemouth", "Chelsea")
        )
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport(
            "soccer_epl", KICKOFF - timedelta(minutes=30)
        )

        assert (stats.already_started, stats.snapshots_stored) == (1, 0)

    async def test_snapshot_carries_source_market_basis_payload_and_prices(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        snapshot = (await pglite_async_session.execute(select(OddsSnapshot))).scalar_one()
        prices = (await pglite_async_session.execute(select(Price))).scalars().all()
        payload = (await pglite_async_session.execute(select(SnapshotPayload))).scalar_one()
        assert (snapshot.source, snapshot.market, snapshot.time_basis) == (
            Source.POLYMARKET,
            Market.ONE_X_TWO,
            TimeBasis.OBSERVED,
        )
        assert snapshot.payload_id == payload.id
        assert sorted((p.side, p.outcome) for p in prices) == sorted(
            (side, outcome)
            for side in (Side.MID, Side.BACK, Side.LAY)
            for outcome in (PriceOutcome.HOME, PriceOutcome.DRAW, PriceOutcome.AWAY)
        )
        by_side = {p.side: float(p.price) for p in prices if p.outcome is PriceOutcome.HOME}
        assert by_side[Side.BACK] == pytest.approx(1 / 0.54)
        assert by_side[Side.LAY] == pytest.approx(1 / 0.52)
        assert by_side[Side.MID] == pytest.approx(1 / 0.53)

    async def test_records_the_polymarket_id_on_the_event(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        event = make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea")
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        held = await pglite_async_session.get(EventExternalId, (Source.POLYMARKET, "900"))
        assert held is not None
        assert held.event_id == event.id

    async def test_a_recorded_id_attaches_across_a_moved_kickoff(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        event = make_epl_event(1, KICKOFF + timedelta(hours=6), "Bournemouth", "Chelsea")
        pglite_async_session.add(event)
        await pglite_async_session.commit()
        pglite_async_session.add(
            EventExternalId(source=Source.POLYMARKET, external_id="900", event_id=event.id)
        )
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert stats.snapshots_stored == 1

    async def test_a_second_match_for_a_fixture_is_refused_and_logged_once(
        self, pglite_async_session: Any, patch_live: FakeClient, gamma_event: Any
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        duplicate = gamma_event()
        duplicate["id"] = "901"
        patch_live.events = [gamma_event(), duplicate]
        now = KICKOFF - timedelta(hours=5)

        first = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)
        second = await fetch_polymarket_odds.ingest_sport("soccer_epl", now + timedelta(minutes=15))

        refusals = (
            (
                await pglite_async_session.execute(
                    select(DataQualityLog).where(DataQualityLog.issue_type == "external_id_refused")
                )
            )
            .scalars()
            .all()
        )
        assert (first.snapshots_stored, first.matches_unattached) == (1, 1)
        assert (second.snapshots_stored, second.matches_unattached) == (1, 1)
        assert len(refusals) == 1
        assert len((await pglite_async_session.execute(select(Price))).scalars().all()) == 18

    async def test_second_run_at_the_same_time_stores_and_archives_nothing_new(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        now = KICKOFF - timedelta(hours=5)

        first = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)
        second = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)

        assert (first.snapshots_stored, second.snapshots_stored) == (1, 0)
        assert len((await pglite_async_session.execute(select(OddsSnapshot))).scalars().all()) == 1
        assert (
            len((await pglite_async_session.execute(select(SnapshotPayload))).scalars().all()) == 1
        )
        assert len((await pglite_async_session.execute(select(Price))).scalars().all()) == 9

    async def test_logs_a_successful_fetch(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        log = (await pglite_async_session.execute(select(FetchLog))).scalar_one()
        assert (log.sport_key, log.success, log.events_count) == ("soccer_epl", True, 1)

    async def test_empty_listing_with_scheduled_events_fails_and_logs(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        patch_live.events = []

        with pytest.raises(fetch_polymarket_odds.EmptyListingError):
            await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        log = (await pglite_async_session.execute(select(FetchLog))).scalar_one()
        assert log.success is False
        assert "lists no upcoming" in (log.error_message or "")

    @pytest.mark.parametrize(
        ("status", "days_out"),
        [(EventStatus.SCHEDULED, 6), (EventStatus.CANCELLED, 2), (EventStatus.POSTPONED, 2)],
    )
    async def test_empty_listing_ignores_far_or_unscheduled_events(
        self, pglite_async_session: Any, patch_live: FakeClient, status: EventStatus, days_out: int
    ) -> None:
        now = KICKOFF - timedelta(days=days_out)
        pglite_async_session.add(
            make_epl_event(1, KICKOFF, "Bournemouth", "Chelsea", status=status)
        )
        await pglite_async_session.commit()
        patch_live.events = []

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)

        assert stats.errors == []

    async def test_empty_listing_without_scheduled_events_is_not_a_failure(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        patch_live.events = []

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert stats.matches_seen == 0
        assert (await pglite_async_session.execute(select(FetchLog))).first() is None


class TestMain:
    @pytest.fixture
    def alerts(self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> list[str]:
        raised: list[str] = []

        @asynccontextmanager
        async def spy(name: str) -> AsyncIterator[None]:
            try:
                yield
            except Exception as e:
                raised.append(f"{name}: {e}")
                raise

        monkeypatch.setattr(fetch_polymarket_odds, "job_alert_context", spy)
        monkeypatch.setattr(
            fetch_polymarket_odds,
            "decide_collection_run",
            AsyncMock(
                return_value=SimpleNamespace(
                    next_execution=KICKOFF, reason="test", should_execute=True
                )
            ),
        )
        monkeypatch.setattr(fetch_polymarket_odds, "self_schedule", AsyncMock())
        monkeypatch.setattr(
            fetch_polymarket_odds, "async_session_maker", SessionMaker(pglite_async_session)
        )
        return raised

    async def test_sport_failure_alerts_after_rescheduling_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any, alerts: list[str]
    ) -> None:
        monkeypatch.setattr(
            fetch_polymarket_odds, "ingest_sport", AsyncMock(side_effect=OSError("gamma down"))
        )

        with pytest.raises(RuntimeError, match="gamma down"):
            await fetch_polymarket_odds.main(JobContext(sport="soccer_epl"))

        assert len(alerts) == 1
        assert alerts[0].startswith("fetch-polymarket-odds-epl:")
        assert fetch_polymarket_odds.self_schedule.await_count == 2
        log = (await pglite_async_session.execute(select(FetchLog))).scalar_one()
        assert log.success is False
        assert "gamma down" in (log.error_message or "")

    async def test_empty_listing_alerts(
        self, monkeypatch: pytest.MonkeyPatch, alerts: list[str]
    ) -> None:
        monkeypatch.setattr(
            fetch_polymarket_odds,
            "ingest_sport",
            AsyncMock(side_effect=fetch_polymarket_odds.EmptyListingError("no listing")),
        )

        with pytest.raises(RuntimeError, match="no listing"):
            await fetch_polymarket_odds.main(JobContext(sport="soccer_epl"))

        assert len(alerts) == 1
