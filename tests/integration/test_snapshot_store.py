"""Integration tests for the writer conflict policy on the snapshot key."""

from __future__ import annotations

from decimal import Decimal

import pytest
from odds_core.models import DataQualityLog, OddsSnapshot, Price
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.storage.snapshot_store import SnapshotWriteOutcome, store_snapshot
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import TWIN_KICKOFF


def snapshot_at(event_id: str, source: Source = Source.ODDSPORTAL) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=TWIN_KICKOFF,
        raw_data={"bookmakers": []},
        bookmaker_count=1,
        source=source,
        market=Market.ONE_X_TWO,
        time_basis=TimeBasis.OBSERVED,
    )


def back_prices(home: str = "2.10") -> list[Price]:
    return [
        Price(bookmaker="bet365", outcome=PriceOutcome.HOME, side=Side.BACK, price=Decimal(home)),
        Price(bookmaker="bet365", outcome=PriceOutcome.DRAW, side=Side.BACK, price=Decimal("3.4")),
        Price(bookmaker="bet365", outcome=PriceOutcome.AWAY, side=Side.BACK, price=Decimal("3.6")),
    ]


async def count(session: AsyncSession, model: type) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


class TestStoreSnapshot:
    @pytest.mark.asyncio
    async def test_inserts_a_new_key_with_its_prices(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        write = await store_snapshot(
            pglite_async_session, snapshot_at(stored_epl_event.id), back_prices()
        )

        assert write.outcome is SnapshotWriteOutcome.INSERTED
        assert write.snapshot.id is not None
        assert await count(pglite_async_session, Price) == 3

    @pytest.mark.asyncio
    async def test_skips_an_identical_snapshot(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        first = await store_snapshot(
            pglite_async_session, snapshot_at(stored_epl_event.id), back_prices()
        )

        # Floats equal to the stored decimals are the same prices.
        again = [
            Price(bookmaker=p.bookmaker, outcome=p.outcome, side=p.side, price=float(p.price))
            for p in back_prices()
        ]
        write = await store_snapshot(pglite_async_session, snapshot_at(stored_epl_event.id), again)

        assert write.outcome is SnapshotWriteOutcome.IDENTICAL
        assert write.snapshot.id == first.snapshot.id
        assert await count(pglite_async_session, OddsSnapshot) == 1
        assert await count(pglite_async_session, Price) == 3
        assert await count(pglite_async_session, DataQualityLog) == 0

    @pytest.mark.asyncio
    async def test_keeps_the_first_of_differing_snapshots_and_logs(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        first = await store_snapshot(
            pglite_async_session, snapshot_at(stored_epl_event.id), back_prices("2.10")
        )

        write = await store_snapshot(
            pglite_async_session, snapshot_at(stored_epl_event.id), back_prices("2.20")
        )
        await pglite_async_session.flush()

        assert write.outcome is SnapshotWriteOutcome.CONFLICTING
        assert write.snapshot.id == first.snapshot.id
        stored = await pglite_async_session.execute(
            select(Price.price).where(Price.outcome == PriceOutcome.HOME)
        )
        assert stored.scalars().all() == [Decimal("2.10")]
        [log] = (await pglite_async_session.execute(select(DataQualityLog))).scalars().all()
        assert log.issue_type == "snapshot_key_conflict"
        assert log.raw_data is not None and log.raw_data["kept_snapshot_id"] == first.snapshot.id

    @pytest.mark.asyncio
    async def test_another_source_at_the_same_time_is_another_key(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await store_snapshot(pglite_async_session, snapshot_at(stored_epl_event.id), back_prices())

        write = await store_snapshot(
            pglite_async_session,
            snapshot_at(stored_epl_event.id, Source.ODDS_API),
            back_prices(),
        )

        assert write.outcome is SnapshotWriteOutcome.INSERTED
        assert await count(pglite_async_session, OddsSnapshot) == 2

    @pytest.mark.asyncio
    async def test_a_snapshot_without_its_key_columns_is_refused(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        snapshot = snapshot_at(stored_epl_event.id)
        snapshot.source = None

        with pytest.raises(ValueError, match="source and market"):
            await store_snapshot(pglite_async_session, snapshot, back_prices())
