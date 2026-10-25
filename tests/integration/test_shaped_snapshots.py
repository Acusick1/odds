"""The shared write path: an envelope stored with its new-shape columns, split markets and prices."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.models import DataQualityLog, Event, Odds, OddsSnapshot, Price, SnapshotPayload
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import make_epl_event

KICKOFF = datetime.now(UTC).replace(microsecond=0) + timedelta(days=1)
TAKEN = KICKOFF - timedelta(hours=5)


def _odds_api_envelope(home_price: int = 120) -> dict[str, Any]:
    return {
        "id": "e9a1",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "williamhill",
                "title": "William Hill",
                "last_update": "2026-10-24T11:00:00Z",
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price": home_price},
                            {"name": "Draw", "price": 250},
                            {"name": "Chelsea", "price": 230},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Arsenal", "price": -105, "point": -0.5},
                            {"name": "Chelsea", "price": -115, "point": 0.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 100, "point": 2.5},
                            {"name": "Under", "price": -120, "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }


@pytest.fixture
async def event(pglite_async_session: AsyncSession) -> Event:
    stored = make_epl_event("e9a1", KICKOFF)
    pglite_async_session.add(stored)
    await pglite_async_session.flush()
    return stored


async def _store(session: AsyncSession, raw_data: dict[str, Any]) -> OddsSnapshot:
    payload_id = await store_payload(session, [raw_data])
    snapshot, _ = await OddsWriter(session).store_odds_snapshot(
        "e9a1",
        raw_data,
        TAKEN,
        shape=SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, payload_id),
        api_request_id="req",
    )
    await session.flush()
    return snapshot


async def _snapshots(session: AsyncSession) -> list[OddsSnapshot]:
    return list((await session.execute(select(OddsSnapshot).order_by(OddsSnapshot.id))).scalars())


class TestMultiMarketEnvelope:
    async def test_each_market_is_a_snapshot_cut_to_its_own_market(
        self, pglite_async_session: AsyncSession, event: Event
    ) -> None:
        session = pglite_async_session
        raw = _odds_api_envelope()

        stored = await _store(session, raw)

        first, spreads, totals = await _snapshots(session)
        assert stored.id == first.id
        assert [
            [m["key"] for m in s.raw_data["bookmakers"][0]["markets"]]
            for s in (first, spreads, totals)
        ] == [["1x2"], ["spreads"], ["totals"]]
        assert first.raw_data["id"] == raw["id"]
        assert [s.market for s in (first, spreads, totals)] == [
            Market.ONE_X_TWO,
            Market.SPREADS,
            Market.TOTALS,
        ]
        assert {s.payload_id for s in (first, spreads, totals)} == {first.payload_id}
        assert {s.api_request_id for s in (first, spreads, totals)} == {"req"}
        assert {s.fetch_tier for s in (first, spreads, totals)} == {first.fetch_tier}

    async def test_odds_rows_and_prices_go_to_the_snapshot_of_their_market(
        self, pglite_async_session: AsyncSession, event: Event
    ) -> None:
        session = pglite_async_session

        await _store(session, _odds_api_envelope())

        first, spreads, totals = await _snapshots(session)
        odds = (await session.execute(select(Odds.snapshot_id, Odds.market_key))).all()
        assert {(snapshot_id, market) for snapshot_id, market in odds} == {
            (first.id, "1x2"),
            (spreads.id, "spreads"),
            (totals.id, "totals"),
        }
        per_snapshot = dict(
            (
                await session.execute(
                    select(Price.snapshot_id, func.count()).group_by(Price.snapshot_id)
                )
            ).all()
        )
        assert per_snapshot == {first.id: 3, spreads.id: 2, totals.id: 2}


class TestKeyConflicts:
    async def test_an_identical_envelope_at_the_same_time_is_not_stored_again(
        self, pglite_async_session: AsyncSession, event: Event
    ) -> None:
        session = pglite_async_session
        first = await _store(session, _odds_api_envelope())

        again = await _store(session, _odds_api_envelope())

        assert again.id == first.id
        assert len(await _snapshots(session)) == 3
        assert (await session.execute(select(func.count()).select_from(Odds))).scalar_one() == 7

    async def test_a_conflicting_envelope_keeps_the_stored_one_and_is_logged(
        self, pglite_async_session: AsyncSession, event: Event
    ) -> None:
        session = pglite_async_session
        first = await _store(session, _odds_api_envelope())

        again = await _store(session, _odds_api_envelope(home_price=140))

        assert again.id == first.id
        assert len(await _snapshots(session)) == 3
        conflicts = (
            await session.execute(
                select(DataQualityLog).where(DataQualityLog.issue_type == "snapshot_key_conflict")
            )
        ).scalars()
        assert [c.raw_data["kept_snapshot_id"] for c in conflicts] == [first.id]
        assert (await session.execute(select(func.count()).select_from(Odds))).scalar_one() == 7
        assert (
            await session.execute(select(func.count()).select_from(SnapshotPayload))
        ).scalar_one() == 2


class TestEmptyEnvelope:
    async def test_an_envelope_quoting_no_market_is_stored_without_one(
        self, pglite_async_session: AsyncSession, event: Event
    ) -> None:
        session = pglite_async_session

        stored = await _store(session, {"id": "e9a1", "bookmakers": []})

        [snapshot] = await _snapshots(session)
        assert snapshot.id == stored.id
        assert (snapshot.source, snapshot.market, snapshot.time_basis) == (
            Source.ODDS_API,
            None,
            TimeBasis.OBSERVED,
        )
