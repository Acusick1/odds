"""A prediction's snapshot must belong to the prediction's event."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_core.models import OddsSnapshot
from odds_core.prediction_models import Prediction
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import API_EVENT_ID, TWIN_EVENT_ID, TWIN_KICKOFF, make_epl_event


async def _snapshot_of(session: AsyncSession, event_id: str) -> OddsSnapshot:
    snapshot = OddsSnapshot(
        event_id=event_id,
        snapshot_time=TWIN_KICKOFF - timedelta(hours=3),
        raw_data={},
        bookmaker_count=1,
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


def _id(snapshot: OddsSnapshot) -> int:
    assert snapshot.id is not None
    return snapshot.id


def _prediction(event_id: str, snapshot_id: int) -> Prediction:
    return Prediction(
        event_id=event_id,
        snapshot_id=snapshot_id,
        model_name="m",
        model_version="1",
        predicted_clv=0.01,
    )


class TestPredictionSnapshotBelongsToEvent:
    @pytest.mark.asyncio
    async def test_matching_snapshot_is_accepted(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.flush()
        snapshot = await _snapshot_of(session, TWIN_EVENT_ID)

        session.add(_prediction(TWIN_EVENT_ID, _id(snapshot)))
        await session.flush()
        await session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))

    @pytest.mark.asyncio
    async def test_snapshot_of_another_event_is_rejected(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.flush()
        snapshot = await _snapshot_of(session, TWIN_EVENT_ID)

        session.add(_prediction(API_EVENT_ID, _id(snapshot)))
        await session.flush()
        with pytest.raises(IntegrityError, match="fk_prediction_snapshot_event"):
            await session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
