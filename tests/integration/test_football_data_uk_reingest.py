"""Integration test for replacing football-data.co.uk snapshots on re-ingest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import select

from scripts.ingest_football_data_uk import (
    SOURCE_TAG,
    FdukStats,
    _ingest_one_match,
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


class TestReingest:
    @pytest.mark.asyncio
    async def test_replaces_legacy_timestamps_and_corrects_kickoff(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        legacy_kickoff = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        session.add(
            Event(
                id="fduk_2024-2025_ARS_WOL_2024-08-17",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=legacy_kickoff,
                home_team="Arsenal",
                away_team="Wolves",
                status=EventStatus.FINAL,
                home_score=2,
                away_score=0,
                completed_at=legacy_kickoff + timedelta(hours=2),
            )
        )
        for offset in (24, 1):
            session.add(
                OddsSnapshot(
                    event_id="fduk_2024-2025_ARS_WOL_2024-08-17",
                    snapshot_time=legacy_kickoff - timedelta(hours=offset),
                    raw_data={"bookmakers": [], "source": SOURCE_TAG},
                    bookmaker_count=0,
                    api_request_id=SOURCE_TAG,
                    fetch_tier="early",
                    hours_until_commence=float(offset),
                )
            )
        await session.flush()

        stats = FdukStats()
        await _ingest_one_match(session, ROW, "2024-2025", stats)
        await _ingest_one_match(session, ROW, "2024-2025", stats)
        await session.flush()

        event = await session.get(Event, "fduk_2024-2025_ARS_WOL_2024-08-17")
        assert event is not None
        assert event.commence_time == datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        snaps = (
            (
                await session.execute(
                    select(OddsSnapshot)
                    .where(OddsSnapshot.event_id == event.id)
                    .order_by(OddsSnapshot.snapshot_time)
                )
            )
            .scalars()
            .all()
        )
        assert [s.snapshot_time for s in snaps] == [
            datetime(2024, 8, 16, 16, 0, tzinfo=UTC),
            datetime(2024, 8, 17, 11, 30, tzinfo=UTC),
        ]
        assert [s.fetch_tier for s in snaps] == ["sharp", "closing"]
        assert [s.hours_until_commence for s in snaps] == [19.5, 0.0]
        opening_keys = {bk["key"] for bk in snaps[0].raw_data["bookmakers"]}
        assert opening_keys == {"bet365", "paddypower"}


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


def _snapshot(event_id: str, tag: str, when: datetime) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=when,
        raw_data={"bookmakers": [], "source": tag},
        bookmaker_count=0,
        api_request_id=tag,
        fetch_tier="early",
        hours_until_commence=24.0,
    )


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
        session.add(_event(OP_ID, op_kickoff))
        op_snapshot_time = op_kickoff - timedelta(hours=5)
        session.add(_snapshot(OP_ID, "oddsportal", op_snapshot_time))
        await session.flush()

        stats = FdukStats()
        await ingest_row(session, ROW, "2024-2025", stats)
        await session.flush()

        event = await session.get(Event, OP_ID)
        assert event is not None and event.commence_time == op_kickoff
        snaps = await _snaps(session, OP_ID)
        assert [s.api_request_id for s in snaps] == [
            "football_data_uk",
            "oddsportal",
            "football_data_uk",
        ]
        assert snaps[-1].snapshot_time == op_kickoff
        assert snaps[-1].hours_until_commence == 0.0
        assert stats.kickoff_mismatches == 1

    @pytest.mark.asyncio
    async def test_ambiguous_match_still_corrects_fduk_event(self, pglite_async_session) -> None:
        session = pglite_async_session
        legacy = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)
        session.add(_event(FDUK_ID, legacy))
        session.add(_event("op_live_other", legacy + timedelta(hours=20)))
        await session.flush()

        await ingest_row(session, ROW, "2024-2025", FdukStats())
        await session.flush()

        event = await session.get(Event, FDUK_ID)
        assert event is not None
        assert event.commence_time == datetime(2024, 8, 17, 11, 30, tzinfo=UTC)


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
        await session.flush()

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
        await session.flush()

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
        assert [s.raw_data["timing"]["basis"] for s in snaps] == ["kickoff"]
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

        snaps = await _snaps(session, "fduk_2024-2025_ARS_WOL_2025-01-14")
        assert [s.fetch_tier for s in snaps] == ["pregame", "closing"]


class TestAtomicityAndIdempotency:
    @pytest.mark.asyncio
    async def test_failure_mid_row_keeps_existing_snapshots(
        self, pglite_async_session, monkeypatch
    ) -> None:
        session = pglite_async_session
        kickoff = datetime(2024, 8, 17, 11, 30, tzinfo=UTC)
        session.add(_event(FDUK_ID, kickoff))
        session.add(_snapshot(FDUK_ID, SOURCE_TAG, kickoff - timedelta(hours=24)))
        await session.flush()

        def boom(*_args, **_kwargs) -> None:
            raise RuntimeError("insert failed")

        monkeypatch.setattr(session, "add_all", boom)
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
