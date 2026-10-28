"""Integration test for replacing football-data.co.uk snapshots on re-ingest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.paper_trade_models import CandidateDecision
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.tier_utils import snapshot_tiers
from sqlalchemy import func, select

from scripts.ingest_football_data_uk import (
    FdukStats,
    ingest_row,
)

ROW = {
    "Date": "17/08/2024",
    "Time": "12:30",
    "HomeTeam": "Arsenal",
    "AwayTeam": "Wolves",
    "FTHG": "2",
    "FTAG": "0",
    "B365H": "1.30",
    "B365D": "5.50",
    "B365A": "10.00",
    "B365CH": "1.28",
    "B365CD": "5.60",
    "B365CA": "11.00",
    "PPH": "1.29",
    "PPD": "5.50",
    "PPA": "10.00",
}


class TestShape:
    @pytest.mark.asyncio
    async def test_writes_shaped_snapshots_prices_and_one_payload(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        row = {**ROW, "MaxH": "1.35", "MaxD": "5.80", "MaxA": "11.00"}

        await ingest_row(session, row, "2024-2025", FdukStats())
        await session.flush()

        snaps = await _snaps(session, FDUK_ID)
        assert [(s.source, s.market, s.time_basis) for s in snaps] == [
            (Source.FOOTBALL_DATA_UK, Market.ONE_X_TWO, TimeBasis.UPPER_BOUND),
            (Source.FOOTBALL_DATA_UK, Market.ONE_X_TWO, TimeBasis.CLOSE),
        ]
        assert snaps[0].snapshot_time == datetime(2024, 8, 16, 16, 0, tzinfo=UTC)
        assert snaps[1].snapshot_time == datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        assert snaps[0].payload_id is not None and snaps[0].payload_id == snaps[1].payload_id
        payload = await session.get(SnapshotPayload, snaps[0].payload_id)
        assert payload is not None
        assert {bk["key"] for bk in payload.payload[0]["bookmakers"]} == {
            "bet365",
            "paddypower",
            "market_max",
        }
        for snap in snaps:
            prices = (
                (await session.execute(select(Price).where(Price.snapshot_id == snap.id)))
                .scalars()
                .all()
            )
            assert {p.bookmaker for p in prices} <= {"bet365", "paddypower"}
            assert {p.side for p in prices} == {Side.BACK}
        closing = (
            (await session.execute(select(Price).where(Price.snapshot_id == snaps[1].id)))
            .scalars()
            .all()
        )
        assert {(p.bookmaker, p.outcome): p.price for p in closing}[
            ("bet365", PriceOutcome.HOME)
        ] == Decimal("1.28")

    @pytest.mark.asyncio
    async def test_issues_no_external_id_and_creates_the_event_at_the_csv_kickoff(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session

        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()

        event = await session.get(Event, FDUK_ID)
        assert event is not None
        assert event.commence_time == datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 0)
        assert event.completed_at == event.commence_time + timedelta(hours=2)
        count = await session.execute(select(func.count()).select_from(EventExternalId))
        assert count.scalar_one() == 0


class TestReplacementRefusal:
    @pytest.mark.asyncio
    async def test_snapshots_a_decision_used_as_input_are_kept(self, pglite_async_session) -> None:
        session = pglite_async_session
        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()
        stored = await _snaps(session, FDUK_ID)
        event = await session.get(Event, FDUK_ID)
        assert event is not None
        session.add(
            CandidateDecision(
                candidate_name="c",
                candidate_version=1,
                event_id=event.id,
                kickoff_at_decision=event.commence_time,
                known_at=event.commence_time,
                decision_time=event.commence_time - timedelta(hours=1),
                minutes_before_kickoff=60,
                input_snapshot_ids=[stored[0].id],
            )
        )
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, {**ROW, "B365CH": "1.50"}, "2024-2025", stats)
        await session.flush()

        assert [s.id for s in await _snaps(session, FDUK_ID)] == [s.id for s in stored]
        assert (stats.replacements_refused, stats.snapshots_updated) == (1, 0)
        issues = await session.execute(
            select(DataQualityLog.issue_type).where(DataQualityLog.event_id == FDUK_ID)
        )
        assert "fduk_replace_refused" in set(issues.scalars())


class TestReplacement:
    @pytest.mark.asyncio
    async def test_replaces_snapshots_whose_content_differs(self, pglite_async_session) -> None:
        session = pglite_async_session
        stale_kickoff = datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        session.add(_event(FDUK_ID, stale_kickoff))
        for offset in (24, 1):
            session.add(
                _snapshot(FDUK_ID, Source.FOOTBALL_DATA_UK, stale_kickoff - timedelta(hours=offset))
            )
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        snaps = await _snaps(session, FDUK_ID)
        assert [s.time_basis for s in snaps] == [TimeBasis.UPPER_BOUND, TimeBasis.CLOSE]
        assert (stats.snapshots_updated, stats.snapshots_inserted) == (2, 2)

    @pytest.mark.asyncio
    async def test_a_moved_kickoff_leaves_one_close(self, pglite_async_session) -> None:
        session = pglite_async_session
        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()
        event = await session.get(Event, FDUK_ID)
        assert event is not None
        await _hold_oddsportal_id(session, event)
        moved = event.commence_time + timedelta(hours=3)
        event.commence_time = moved
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        closes = [s for s in await _snaps(session, FDUK_ID) if s.time_basis is TimeBasis.CLOSE]
        assert [s.snapshot_time for s in closes] == [moved]
        assert stats.snapshots_updated == 2

    @pytest.mark.asyncio
    async def test_replaced_snapshots_take_their_prices_and_payload_with_them(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()
        changed = {**ROW, "B365CH": "1.50"}

        await ingest_row(session, changed, "2024-2025", FdukStats())
        await session.flush()

        prices = (
            (
                await session.execute(
                    select(Price).where(Price.bookmaker == "bet365", Price.outcome == "home")
                )
            )
            .scalars()
            .all()
        )
        assert sorted(p.price for p in prices) == [Decimal("1.30"), Decimal("1.50")]
        payloads = await session.execute(select(func.count()).select_from(SnapshotPayload))
        assert payloads.scalar_one() == 1


FDUK_ID = "fduk_2024-2025_ARS_WOL_2024-08-17"
OP_ID = "op_2024-2025_ARS_WOL_2024-08-17"


def _event(event_id: str, commence: datetime) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence,
        home_team="Arsenal",
        away_team="Wolves",
        status=EventStatus.FINAL,
        home_score=2,
        away_score=0,
        completed_at=commence + timedelta(hours=2),
    )


def _snapshot(event_id: str, source: Source, when: datetime) -> OddsSnapshot:
    """A stored snapshot that holds no prices."""
    return OddsSnapshot(event_id=event_id, snapshot_time=when, source=source)


async def _hold_oddsportal_id(session, event: Event) -> None:
    await session.flush()
    session.add(
        EventExternalId(
            source=Source.ODDSPORTAL, external_id="AbCd1234", event_id=event.surrogate_id
        )
    )
    await session.flush()


async def _snaps(session, event_id: str) -> list[OddsSnapshot]:
    result = await session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.event_id == event_id)
        .order_by(OddsSnapshot.snapshot_time)
    )
    return list(result.scalars().all())


class TestMatchedAndAmbiguousEvents:
    @pytest.mark.asyncio
    async def test_op_event_keeps_time_and_oddsportal_snapshots(self, pglite_async_session) -> None:
        session = pglite_async_session
        op_kickoff = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        op_event = _event(OP_ID, op_kickoff)
        session.add(op_event)
        await _hold_oddsportal_id(session, op_event)
        op_snapshot_time = op_kickoff - timedelta(hours=5)
        session.add(_snapshot(OP_ID, Source.ODDSPORTAL, op_snapshot_time))
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        event = await session.get(Event, OP_ID)
        assert event is not None and event.commence_time == op_kickoff
        snaps = await _snaps(session, OP_ID)
        assert [s.source for s in snaps] == [
            Source.FOOTBALL_DATA_UK,
            Source.ODDSPORTAL,
            Source.FOOTBALL_DATA_UK,
        ]
        assert snaps[-1].snapshot_time == op_kickoff
        assert stats.kickoff_mismatches == 1

    @pytest.mark.asyncio
    async def test_back_to_back_fixture_is_not_conflated(self, pglite_async_session) -> None:
        session = pglite_async_session
        listed = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        session.add(_event(FDUK_ID, listed))
        session.add(_event("op_live_other", listed + timedelta(hours=20)))
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        assert await _snaps(session, "op_live_other") == []
        assert len(await _snaps(session, FDUK_ID)) == 2
        assert stats.events_matched == 1

    @pytest.mark.asyncio
    async def test_moves_the_kickoff_of_an_fduk_only_event(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add(_event(FDUK_ID, datetime(2024, 8, 17, 12, 30, tzinfo=UTC)))
        await session.flush()

        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()

        event = await session.get(Event, FDUK_ID)
        assert event is not None
        assert event.commence_time == datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        closes = [s for s in await _snaps(session, FDUK_ID) if s.time_basis is TimeBasis.CLOSE]
        assert [s.snapshot_time for s in closes] == [event.commence_time]

    @pytest.mark.asyncio
    async def test_leaves_the_kickoff_of_an_oddsportal_only_event_that_holds_no_id(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        listed = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        session.add(_event(OP_ID, listed))
        session.add(_snapshot(OP_ID, Source.ODDSPORTAL, listed - timedelta(hours=5)))
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        event = await session.get(Event, OP_ID)
        assert event is not None and event.commence_time == listed
        assert stats.kickoff_mismatches == 1

    @pytest.mark.asyncio
    async def test_leaves_the_kickoff_of_an_event_oddsportal_holds_an_id_on(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        listed = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        event = _event(FDUK_ID, listed)
        session.add(event)
        await session.flush()
        session.add(
            EventExternalId(
                source=Source.ODDSPORTAL, external_id="AbCd1234", event_id=event.surrogate_id
            )
        )
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        assert event.commence_time == listed
        assert stats.kickoff_mismatches == 1

    @pytest.mark.asyncio
    async def test_two_candidates_none_within_two_hours_create_a_new_event(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        kickoff = datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        session.add(_event("before", kickoff - timedelta(hours=10)))
        session.add(_event("after", kickoff + timedelta(hours=10)))
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        assert stats.events_created == 1
        assert len(await _snaps(session, FDUK_ID)) == 2


def _unsettled_event(event_id: str, kickoff: datetime, status: EventStatus) -> Event:
    event = _event(event_id, kickoff)
    event.status = status
    event.home_score = None
    event.away_score = None
    event.completed_at = None
    return event


class TestResultsOnExistingEvents:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [EventStatus.FINAL, EventStatus.SCHEDULED, EventStatus.LIVE])
    async def test_fills_result_of_an_event_without_scores(
        self, pglite_async_session, status: EventStatus
    ) -> None:
        session = pglite_async_session
        kickoff = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        event = _unsettled_event(OP_ID, kickoff, status)
        session.add(event)
        await _hold_oddsportal_id(session, event)

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()
        await session.refresh(event)

        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 0)
        assert event.completed_at == kickoff + timedelta(hours=2)
        assert stats.results_filled == 1

    @pytest.mark.asyncio
    async def test_keeps_existing_scores(self, pglite_async_session) -> None:
        session = pglite_async_session
        event = _event(OP_ID, datetime(2024, 8, 17, 12, 30, tzinfo=UTC))
        event.home_score, event.away_score = 3, 3
        session.add(event)
        await _hold_oddsportal_id(session, event)

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()
        await session.refresh(event)

        assert (event.home_score, event.away_score) == (3, 3)
        assert stats.results_filled == 0


class TestOpeningWithheld:
    @pytest.mark.asyncio
    async def test_deadline_after_kickoff_writes_closing_only(self, pglite_async_session) -> None:
        session = pglite_async_session
        row = {**ROW, "Date": "26/12/2023", "Time": "12:30"}
        stats = FdukStats()
        await ingest_row(session, row, "2023-2024", stats)
        await session.flush()

        snaps = await _snaps(session, "fduk_2023-2024_ARS_WOL_2023-12-26")
        assert [s.time_basis for s in snaps] == [TimeBasis.CLOSE]
        assert stats.openings_withheld == 1

    @pytest.mark.asyncio
    async def test_opening_near_kickoff_is_never_the_closing_tier(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        # Tuesday 19:00 GMT: the Tuesday 13:00 deadline is 6 h out
        row = {**ROW, "Date": "14/01/2025", "Time": "19:00"}
        await ingest_row(session, row, "2024-2025", FdukStats())
        await session.flush()

        event = await session.get(Event, "fduk_2024-2025_ARS_WOL_2025-01-14")
        assert event is not None
        snaps = await _snaps(session, event.id)
        assert snapshot_tiers(
            [(s.snapshot_time, s.source, s.time_basis, s.market) for s in snaps],
            event.commence_time,
        ) == [FetchTier.PREGAME, FetchTier.CLOSING]


class TestAtomicityAndIdempotency:
    @pytest.mark.asyncio
    async def test_failure_mid_row_keeps_existing_snapshots(
        self, pglite_async_session, monkeypatch
    ) -> None:
        session = pglite_async_session
        kickoff = datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        session.add(_event(FDUK_ID, kickoff))
        session.add(_snapshot(FDUK_ID, Source.FOOTBALL_DATA_UK, kickoff - timedelta(hours=24)))
        await session.flush()

        async def boom(*_args, **_kwargs) -> None:
            raise RuntimeError("insert failed")

        monkeypatch.setattr("scripts.ingest_football_data_uk.store_shaped_snapshot", boom)
        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)

        assert len(stats.errors) == 1
        assert len(await _snaps(session, FDUK_ID)) == 1

    @pytest.mark.asyncio
    async def test_unchanged_rerun_reports_no_updates(self, pglite_async_session) -> None:
        session = pglite_async_session
        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()
        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        assert stats.snapshots_updated == 0
        assert stats.snapshots_inserted == 0
        assert len(await _snaps(session, FDUK_ID)) == 2
