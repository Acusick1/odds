"""Integration tests for loading football-data.co.uk league CSVs end to end."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot, Price, SnapshotPayload
from odds_core.vocabulary import TimeBasis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scripts.ingest_football_data_uk import FdukLeague, ingest_season
from tests.integration.conftest import FDUK_FIXTURES

LA_LIGA = FdukLeague("SP1", "soccer_spain_la_liga", "La Liga")


async def _dump(session: AsyncSession) -> list[dict[str, Any]]:
    events = await session.execute(select(Event).order_by(Event.commence_time, Event.home_team))
    dumped: list[dict[str, Any]] = []
    for event in events.scalars().all():
        snaps = (
            await session.execute(
                select(OddsSnapshot)
                .where(OddsSnapshot.event_id == event.id)
                .order_by(OddsSnapshot.snapshot_time)
            )
        ).scalars()
        snapshots: list[dict[str, Any]] = []
        payload_ids: set[int | None] = set()
        for snap in snaps.all():
            prices = (
                await session.execute(select(Price).where(Price.snapshot_id == snap.id))
            ).scalars()
            payload_ids.add(snap.payload_id)
            snapshots.append(
                {
                    "source": snap.source,
                    "market": snap.market,
                    "time_basis": snap.time_basis,
                    "snapshot_time": snap.snapshot_time.isoformat(),
                    "prices": sorted(
                        [p.bookmaker, p.outcome, p.side, str(p.price)] for p in prices.all()
                    ),
                }
            )
        (payload_id,) = payload_ids
        payload = await session.get(SnapshotPayload, payload_id)
        assert payload is not None
        dumped.append(
            {
                "sport_key": event.sport_key,
                "sport_title": event.sport_title,
                "home_team": event.home_team,
                "away_team": event.away_team,
                "commence_time": event.commence_time.isoformat(),
                "status": event.status,
                "score": [event.home_score, event.away_score],
                "completed_at": event.completed_at.isoformat() if event.completed_at else None,
                "snapshots": snapshots,
                "payload": payload.payload,
            }
        )
    return json.loads(json.dumps(dumped, default=str))


class TestEplGolden:
    @pytest.mark.asyncio
    async def test_epl_seasons_load_as_recorded(
        self, fduk_session: AsyncSession, fduk_epl: FdukLeague
    ) -> None:
        for season in ("2018-2019", "2025-2026"):
            stats = await ingest_season(fduk_epl, season, no_download=True)
            assert stats.errors == []

        golden = json.loads((FDUK_FIXTURES / "E0_golden.json").read_text())
        assert await _dump(fduk_session) == golden


class TestSecondLeague:
    @pytest.mark.asyncio
    async def test_la_liga_season_loads_final_events_under_its_sport(
        self, fduk_session: AsyncSession
    ) -> None:
        stats = await ingest_season(LA_LIGA, "2024-2025", no_download=True)

        assert stats.errors == []
        assert (stats.games_loaded, stats.events_created) == (14, 14)
        events = (await fduk_session.execute(select(Event))).scalars().all()
        assert {(e.sport_key, e.sport_title) for e in events} == {
            ("soccer_spain_la_liga", "La Liga")
        }
        assert {e.status for e in events} == {EventStatus.FINAL}
        assert all(e.home_score is not None and e.away_score is not None for e in events)

    @pytest.mark.asyncio
    async def test_la_liga_match_is_dated_and_priced_like_epl(
        self, fduk_session: AsyncSession
    ) -> None:
        await ingest_season(LA_LIGA, "2024-2025", no_download=True)

        event = (
            await fduk_session.execute(select(Event).where(Event.home_team == "Ath Bilbao"))
        ).scalar_one()
        assert event.away_team == "Getafe"
        assert event.commence_time == datetime(2024, 8, 15, 17, 0, tzinfo=UTC)
        assert (event.home_score, event.away_score) == (1, 1)
        snaps = (
            await fduk_session.execute(
                select(OddsSnapshot)
                .where(OddsSnapshot.event_id == event.id)
                .order_by(OddsSnapshot.snapshot_time)
            )
        ).scalars()
        assert [(s.time_basis, s.snapshot_time) for s in snaps.all()] == [
            (TimeBasis.UPPER_BOUND, datetime(2024, 8, 13, 12, 0, tzinfo=UTC)),
            (TimeBasis.CLOSE, event.commence_time),
        ]
