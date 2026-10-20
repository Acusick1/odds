"""Integration tests for the Polymarket odds backfill and live job."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from odds_core.models import EventStatus, FetchLog, Odds, OddsSnapshot
from odds_lambda.jobs import fetch_polymarket_odds
from odds_lambda.polymarket_odds import (
    HISTORY_REQUEST_ID,
    LIVE_REQUEST_ID_PREFIX,
    MAX_HORIZON,
    MAX_STALENESS,
    Observation,
    grid_offsets,
)
from odds_lambda.polymarket_odds_backfill import backfill_sport
from odds_lambda.scheduling.jobs import JobContext
from sqlalchemy import select

from tests.test_helpers import make_epl_event

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
    event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea", status=EventStatus.FINAL)
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
        assert report.events_paired == 1
        assert report.events_written == 1
        assert len(snapshots) == len(grid_offsets()) == report.snapshots_written
        assert {s.api_request_id for s in snapshots} == {HISTORY_REQUEST_ID}
        assert snapshots[-1].snapshot_time == KICKOFF
        assert snapshots[-1].fetch_tier == "closing"
        assert snapshots[-1].hours_until_commence == 0
        assert snapshots[0].fetch_tier == "opening"
        bookmakers = snapshots[-1].raw_data["bookmakers"]
        assert [b["key"] for b in bookmakers] == ["polymarket"]
        assert bookmakers[0]["markets"][0]["key"] == "1x2"
        assert [o["name"] for o in bookmakers[0]["markets"][0]["outcomes"]] == [
            "Bournemouth",
            "Draw",
            "Chelsea",
        ]
        assert (await pglite_async_session.execute(select(Odds))).first() is None

    async def test_rerun_skips_events_with_history(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))
        calls = patch_backfill.history_calls

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert report.events_skipped_existing == 1
        assert report.snapshots_written == 0
        assert patch_backfill.history_calls == calls
        count = len((await pglite_async_session.execute(select(OddsSnapshot))).scalars().all())
        assert count == len(grid_offsets())

    async def test_dry_run_writes_nothing(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1), dry_run=True)

        assert report.snapshots_written == len(grid_offsets())
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
            "e1",
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

    async def test_twins_get_snapshots_only_on_the_api_event(
        self, pglite_async_session: Any, patch_backfill: FakeClient
    ) -> None:
        for event_id in ("op_2025-2026_BOU_CHE_2025-12-06", "a1b2c3"):
            pglite_async_session.add(
                make_epl_event(
                    event_id, KICKOFF, "Bournemouth", "Chelsea", status=EventStatus.FINAL
                )
            )
        await pglite_async_session.commit()

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        owners = set(
            (await pglite_async_session.execute(select(OddsSnapshot.event_id))).scalars().all()
        )
        assert owners == {"a1b2c3"}
        assert report.twin_events_skipped == 1

    async def test_unpaired_event_is_reported(
        self, pglite_async_session: Any, finished_event: Any, patch_backfill: FakeClient
    ) -> None:
        patch_backfill.events = []

        report = await backfill_sport("soccer_epl", since=KICKOFF - timedelta(days=1))

        assert report.unmatched_event_ids == ["e1"]
        assert report.snapshots_written == 0

    async def test_rejects_sport_without_three_way_markets(self) -> None:
        with pytest.raises(ValueError, match="three-way"):
            await backfill_sport("basketball_nba", since=KICKOFF)


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
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        now = KICKOFF - timedelta(hours=5)

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", now)

        snapshot = (await pglite_async_session.execute(select(OddsSnapshot))).scalar_one()
        odds = (await pglite_async_session.execute(select(Odds))).scalars().all()
        assert stats.snapshots_stored == 1
        assert snapshot.api_request_id == f"{LIVE_REQUEST_ID_PREFIX}{now.isoformat()}"
        assert snapshot.fetch_tier == "pregame"
        assert {o.bookmaker_key for o in odds} == {"polymarket"}
        assert {o.market_key for o in odds} == {"1x2"}
        assert len(odds) == 3

    async def test_skips_matches_already_started(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF + timedelta(hours=1))

        assert stats.matches_seen == 0
        assert (await pglite_async_session.execute(select(OddsSnapshot))).first() is None

    async def test_counts_matches_without_an_event(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.matches_seen, stats.matches_unpaired, stats.snapshots_stored) == (1, 1, 0)

    async def test_rejects_prices_off_sum(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        for market in patch_live.events[0]["markets"]:
            market["bestBid"], market["bestAsk"] = 0.49, 0.51

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.sum_rejected, stats.snapshots_stored) == (1, 0)

    async def test_counts_matches_without_prices(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        patch_live.events[0]["markets"][0]["bestBid"] = None

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.prices_missing, stats.snapshots_stored) == (1, 0)

    async def test_stores_quotes_in_meta(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        snapshot = (await pglite_async_session.execute(select(OddsSnapshot))).scalar_one()
        meta = snapshot.raw_data["bookmakers"][0]["polymarket_meta"]
        assert meta["best_bid"]["home"] == 0.52
        assert meta["best_ask"]["home"] == 0.54

    async def test_skips_wide_spreads(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()
        patch_live.events[0]["markets"][0]["bestBid"] = 0.10
        patch_live.events[0]["markets"][0]["bestAsk"] = 0.30

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        assert (stats.wide_spread, stats.snapshots_stored) == (1, 0)

    async def test_skips_events_that_have_kicked_off_before_polymarket_start(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(
            make_epl_event("e1", KICKOFF - timedelta(hours=1), "Bournemouth", "Chelsea")
        )
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport(
            "soccer_epl", KICKOFF - timedelta(minutes=30)
        )

        assert (stats.already_started, stats.snapshots_stored) == (1, 0)

    async def test_twins_get_one_snapshot_on_the_api_event(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        for event_id in ("op_2025-2026_BOU_CHE_2025-12-06", "a1b2c3"):
            pglite_async_session.add(make_epl_event(event_id, KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        stats = await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        owners = (await pglite_async_session.execute(select(OddsSnapshot.event_id))).scalars().all()
        assert (stats.snapshots_stored, list(owners)) == (1, ["a1b2c3"])

    async def test_logs_a_successful_fetch(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
        await pglite_async_session.commit()

        await fetch_polymarket_odds.ingest_sport("soccer_epl", KICKOFF - timedelta(hours=5))

        log = (await pglite_async_session.execute(select(FetchLog))).scalar_one()
        assert (log.sport_key, log.success, log.events_count) == ("soccer_epl", True, 1)

    async def test_empty_listing_with_scheduled_events_fails_and_logs(
        self, pglite_async_session: Any, patch_live: FakeClient
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea"))
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
            make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea", status=status)
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
