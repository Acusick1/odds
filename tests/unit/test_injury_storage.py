"""Tests for injury report storage layer (writer and reader)."""

from datetime import UTC, date, datetime

import pytest
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event, EventStatus
from odds_lambda.injury_fetcher import InjuryRecord
from odds_lambda.storage.injury_reader import InjuryReader
from odds_lambda.storage.injury_writer import InjuryWriter


def _make_record(**overrides) -> InjuryRecord:
    """Build a sample InjuryRecord with sensible defaults."""
    defaults = {
        "report_time": datetime(2026, 1, 15, 22, 0, tzinfo=UTC),
        "game_date": date(2026, 1, 16),
        "game_time_et": "07:00 PM ET",
        "matchup": "BOS@ORL",
        "team": "Boston Celtics",
        "player_name": "Tatum, Jayson",
        "status": InjuryStatus.OUT,
        "reason": "Left Ankle; Sprain",
    }
    defaults.update(overrides)
    return InjuryRecord(**defaults)


def _make_event(
    event_id: str,
    home: str,
    away: str,
    commence: datetime,
) -> Event:
    """Build a sample Event."""
    return Event(
        id=event_id,
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=commence,
        home_team=home,
        away_team=away,
        status=EventStatus.SCHEDULED,
    )


class TestInjuryWriter:
    """Tests for InjuryWriter."""

    @pytest.mark.asyncio
    async def test_upsert_creates_new_records(self, pglite_async_session):
        writer = InjuryWriter(pglite_async_session)
        records = [
            _make_record(),
            _make_record(player_name="Brown, Jaylen", status=InjuryStatus.QUESTIONABLE),
        ]

        count = await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        assert count == 2

        # Verify data in DB
        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        rows = list(result.scalars().all())
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        """Upserting the same records twice produces no duplicates."""
        writer = InjuryWriter(pglite_async_session)
        records = [_make_record()]

        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        rows = list(result.scalars().all())
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        """Re-upserting with changed status updates the existing row."""
        writer = InjuryWriter(pglite_async_session)

        # Insert as OUT
        await writer.upsert_injury_reports([_make_record(status=InjuryStatus.OUT)])
        await pglite_async_session.commit()

        # Update to AVAILABLE
        await writer.upsert_injury_reports([_make_record(status=InjuryStatus.AVAILABLE)])
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].status == InjuryStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_event_matching_finds_match(self, pglite_async_session):
        """Injury records auto-match to sportsbook events at write time."""
        # Create an event for BOS at ORL on Jan 16 (7:10 PM ET = Jan 17 00:10 UTC)
        event = _make_event(
            "evt_bos_orl",
            home="Orlando Magic",
            away="Boston Celtics",
            commence=datetime(2026, 1, 17, 0, 10, tzinfo=UTC),
        )
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        writer = InjuryWriter(pglite_async_session)
        records = [_make_record(team="Boston Celtics", game_date=date(2026, 1, 16))]
        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        row = result.scalar_one()
        assert row.event_id == "evt_bos_orl"

    @pytest.mark.asyncio
    async def test_event_matching_no_match(self, pglite_async_session):
        """event_id is None when no matching event exists."""
        writer = InjuryWriter(pglite_async_session)
        records = [_make_record()]
        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        row = result.scalar_one()
        assert row.event_id is None

    @pytest.mark.asyncio
    async def test_event_matching_ambiguous(self, pglite_async_session):
        """event_id is None when multiple events match the same team+date."""
        # Two events for Boston on the same date (hypothetical)
        for i in range(2):
            event = _make_event(
                f"evt_bos_{i}",
                home="Orlando Magic",
                away="Boston Celtics",
                commence=datetime(2026, 1, 16, 20 + i, 0, tzinfo=UTC),
            )
            pglite_async_session.add(event)
        await pglite_async_session.commit()

        writer = InjuryWriter(pglite_async_session)
        records = [_make_record(team="Boston Celtics", game_date=date(2026, 1, 16))]
        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        row = result.scalar_one()
        assert row.event_id is None

    @pytest.mark.asyncio
    async def test_event_matching_back_to_back(self, pglite_async_session):
        """Back-to-back games on adjacent ET dates match the correct event only."""
        # Jan 15 game: 7 PM ET = Jan 16 00:00 UTC
        evt_jan15 = _make_event(
            "evt_jan15",
            home="Orlando Magic",
            away="Boston Celtics",
            commence=datetime(2026, 1, 16, 0, 0, tzinfo=UTC),
        )
        # Jan 16 game: 7:30 PM ET = Jan 17 00:30 UTC
        evt_jan16 = _make_event(
            "evt_jan16",
            home="Boston Celtics",
            away="Miami Heat",
            commence=datetime(2026, 1, 17, 0, 30, tzinfo=UTC),
        )
        pglite_async_session.add(evt_jan15)
        pglite_async_session.add(evt_jan16)
        await pglite_async_session.commit()

        writer = InjuryWriter(pglite_async_session)
        records = [_make_record(team="Boston Celtics", game_date=date(2026, 1, 16))]
        await writer.upsert_injury_reports(records)
        await pglite_async_session.commit()

        from sqlalchemy import select

        result = await pglite_async_session.execute(select(InjuryReport))
        row = result.scalar_one()
        assert row.event_id == "evt_jan16"

    @pytest.mark.asyncio
    async def test_empty_records(self, pglite_async_session):
        writer = InjuryWriter(pglite_async_session)
        count = await writer.upsert_injury_reports([])
        assert count == 0


class TestInjuryReader:
    """Tests for InjuryReader."""

    async def _seed_reports(self, session, event_id: str | None = "evt_1") -> list[InjuryReport]:
        """Seed the DB with injury reports for testing reads."""
        # Create event if needed
        if event_id:
            event = _make_event(
                event_id,
                home="Orlando Magic",
                away="Boston Celtics",
                commence=datetime(2026, 1, 17, 0, 10, tzinfo=UTC),  # 7:10 PM ET on Jan 16
            )
            session.add(event)
            await session.commit()

        # Insert reports at two different report_times
        writer = InjuryWriter(session)
        early_records = [
            _make_record(
                report_time=datetime(2026, 1, 15, 14, 0, tzinfo=UTC),
                team="Boston Celtics",
                game_date=date(2026, 1, 16),
                player_name="Tatum, Jayson",
                status=InjuryStatus.QUESTIONABLE,
            ),
        ]
        late_records = [
            _make_record(
                report_time=datetime(2026, 1, 15, 22, 0, tzinfo=UTC),
                team="Boston Celtics",
                game_date=date(2026, 1, 16),
                player_name="Tatum, Jayson",
                status=InjuryStatus.OUT,
            ),
            _make_record(
                report_time=datetime(2026, 1, 15, 22, 0, tzinfo=UTC),
                team="Boston Celtics",
                game_date=date(2026, 1, 16),
                player_name="Brown, Jaylen",
                status=InjuryStatus.PROBABLE,
            ),
        ]
        await writer.upsert_injury_reports(early_records)
        await writer.upsert_injury_reports(late_records)
        await session.commit()

        from sqlalchemy import select

        result = await session.execute(select(InjuryReport).order_by(InjuryReport.report_time))
        return list(result.scalars().all())

    @pytest.mark.asyncio
    async def test_get_injuries_for_event(self, pglite_async_session):
        """Returns all injury reports linked to a specific event."""
        await self._seed_reports(pglite_async_session, event_id="evt_1")
        reader = InjuryReader(pglite_async_session)

        reports = await reader.get_injuries_for_event("evt_1")
        assert len(reports) == 3  # 1 early + 2 late

    @pytest.mark.asyncio
    async def test_get_injuries_before_time_prevents_lookahead(self, pglite_async_session):
        """before_time filter prevents look-ahead bias."""
        await self._seed_reports(pglite_async_session, event_id="evt_1")
        reader = InjuryReader(pglite_async_session)

        # Only see the early report
        cutoff = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
        reports = await reader.get_injuries_for_event("evt_1", before_time=cutoff)
        assert len(reports) == 1
        assert reports[0].status == InjuryStatus.QUESTIONABLE

    @pytest.mark.asyncio
    async def test_get_latest_report_for_event(self, pglite_async_session):
        """Returns all rows from the most recent report_time <= as_of."""
        await self._seed_reports(pglite_async_session, event_id="evt_1")
        reader = InjuryReader(pglite_async_session)

        as_of = datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
        reports = await reader.get_latest_report_for_event("evt_1", as_of)

        # Should get the 2 late-report rows (Tatum OUT, Brown PROBABLE)
        assert len(reports) == 2
        names = {r.player_name for r in reports}
        assert names == {"Tatum, Jayson", "Brown, Jaylen"}

    @pytest.mark.asyncio
    async def test_get_latest_report_early_cutoff(self, pglite_async_session):
        """With early cutoff, returns only the early report snapshot."""
        await self._seed_reports(pglite_async_session, event_id="evt_1")
        reader = InjuryReader(pglite_async_session)

        as_of = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
        reports = await reader.get_latest_report_for_event("evt_1", as_of)

        assert len(reports) == 1
        assert reports[0].player_name == "Tatum, Jayson"
        assert reports[0].status == InjuryStatus.QUESTIONABLE

    @pytest.mark.asyncio
    async def test_get_latest_report_no_data(self, pglite_async_session):
        """Returns empty list when no reports exist for the event."""
        reader = InjuryReader(pglite_async_session)
        reports = await reader.get_latest_report_for_event("nonexistent", datetime.now(UTC))
        assert reports == []

    @pytest.mark.asyncio
    async def test_get_pipeline_stats(self, pglite_async_session):
        """Pipeline stats return correct counts."""
        await self._seed_reports(pglite_async_session, event_id="evt_1")
        reader = InjuryReader(pglite_async_session)

        stats = await reader.get_pipeline_stats()
        assert stats["total_reports"] == 3
        assert stats["unique_players"] == 2
        assert stats["events_matched"] == 1
        assert stats["earliest_game_date"] == date(2026, 1, 16)
        assert stats["latest_game_date"] == date(2026, 1, 16)
        assert "OUT" in stats["status_counts"]

    @pytest.mark.asyncio
    async def test_get_pipeline_stats_empty(self, pglite_async_session):
        """Pipeline stats work with no data."""
        reader = InjuryReader(pglite_async_session)
        stats = await reader.get_pipeline_stats()
        assert stats["total_reports"] == 0
        assert stats["unique_players"] == 0
