"""Integration test for replacing football-data.co.uk snapshots on re-ingest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.oddsportal_common import IngestionStats
from sqlalchemy import select

from scripts.ingest_football_data_uk import SOURCE_TAG, _ingest_one_match

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

        stats = IngestionStats()
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
