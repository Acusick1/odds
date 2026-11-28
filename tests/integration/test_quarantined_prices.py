"""A long retail outlier is quarantined when stored, and no reader of prices sees it."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.models import OddsSnapshot, Price
from odds_core.quarantined_price_models import QuarantinedPrice, QuarantineReason
from odds_core.quotes import quotes
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.candidate_rules import SharpInput, load_market_state
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from odds_lambda.storage.snapshot_store import SnapshotWriteOutcome
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
SNAPSHOT_TIME = T - timedelta(minutes=5)
# William Hill's home price is about 6.5 times the next-longest and +540% EV.
BOOKS = [
    three_way_bookmaker("williamhill", (20.0, 3.4, 2.4)),
    three_way_bookmaker("bet365", (3.0, 3.4, 2.4)),
    betfair_exchange_bookmaker((3.1, 3.5, 2.5)),
]
MIGRATION = "d4b7e2a9c1f3_quarantine_long_retail_outliers"


def _migration() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "migrations" / "versions" / f"{MIGRATION}.py"
    spec = importlib.util.spec_from_file_location(MIGRATION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _quarantined(session: AsyncSession) -> list[QuarantinedPrice]:
    return list((await session.execute(select(QuarantinedPrice))).scalars())


async def _williamhill_home(session: AsyncSession) -> list[Price]:
    return list(
        (
            await session.execute(
                select(Price).where(
                    Price.bookmaker == "williamhill",  # type: ignore[arg-type]
                    Price.outcome == PriceOutcome.HOME,  # type: ignore[arg-type]
                )
            )
        ).scalars()
    )


@pytest.fixture
async def event(pglite_async_session: AsyncSession):
    event = make_epl_event(1, KICKOFF)
    pglite_async_session.add(event)
    await store_tagged(
        pglite_async_session,
        make_tagged_snapshot(event.id, SNAPSHOT_TIME, BOOKS, "oddsportal_live"),
    )
    await pglite_async_session.commit()
    return event


class TestQuarantineOnWrite:
    async def test_the_outlier_is_stored_apart_with_its_reason(
        self, pglite_async_session: AsyncSession, event
    ) -> None:
        [row] = await _quarantined(pglite_async_session)

        assert (row.bookmaker, row.outcome, row.side) == ("williamhill", "home", "back")
        assert row.price == Decimal("20.0")
        assert row.reason is QuarantineReason.LONG_RETAIL_OUTLIER
        assert await _williamhill_home(pglite_async_session) == []

    async def test_the_rest_of_the_snapshot_is_stored(
        self, pglite_async_session: AsyncSession, event
    ) -> None:
        stored = (await pglite_async_session.execute(select(Price))).scalars().all()

        # Three books of three outcomes, less the outlier.
        assert len(stored) == 8

    async def test_storing_the_same_snapshot_again_is_identical(
        self, pglite_async_session: AsyncSession, event
    ) -> None:
        again = make_tagged_snapshot(event.id, SNAPSHOT_TIME, BOOKS, "oddsportal_live")
        write = await store_shaped_snapshot(
            pglite_async_session,
            event,
            again.envelope,
            SNAPSHOT_TIME,
            SnapshotShape(Source.ODDSPORTAL, TimeBasis.OBSERVED, None),
        )

        assert write.first is not None
        assert write.first.outcome is SnapshotWriteOutcome.IDENTICAL
        assert len(await _quarantined(pglite_async_session)) == 1


class TestReadersSkipTheOutlier:
    async def test_decision_time_quotes_omit_it(
        self, pglite_async_session: AsyncSession, event
    ) -> None:
        read = await quotes(pglite_async_session, [event.id], Market.ONE_X_TWO, until=T)

        assert read
        assert not any(q.bookmaker == "williamhill" and q.outcome == "home" for q in read)

    async def test_market_state_omits_it(self, pglite_async_session: AsyncSession, event) -> None:
        state = await load_market_state(
            pglite_async_session,
            event,
            Market.ONE_X_TWO,
            T,
            timedelta(minutes=15),
            frozenset({SharpInput.EXCHANGE}),
            UK_RETAIL_BOOKMAKERS,
        )

        assert state.has_input
        assert PriceOutcome.HOME not in state.retail[None]["williamhill"]
        assert state.retail[None]["bet365"][PriceOutcome.HOME] == 3.0


class TestMigrationQuarantinesStoredOutliers:
    async def test_moves_a_stored_outlier_and_keeps_the_rest(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = make_epl_event(1, KICKOFF)
        pglite_async_session.add(event)
        await pglite_async_session.flush()
        snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=SNAPSHOT_TIME,
            source=Source.ODDS_API,
            market=Market.ONE_X_TWO,
            time_basis=TimeBasis.SOURCE_STAMPED,
        )
        pglite_async_session.add(snapshot)
        await pglite_async_session.flush()
        rows = [
            ("williamhill", PriceOutcome.HOME, "20"),
            ("bet365", PriceOutcome.HOME, "3.0"),
            ("betfair_exchange", PriceOutcome.HOME, "3.1"),
            ("betfair_exchange", PriceOutcome.DRAW, "3.5"),
            ("betfair_exchange", PriceOutcome.AWAY, "2.5"),
        ]
        # Stored directly, as rows from before the guard were.
        pglite_async_session.add_all(
            Price(
                snapshot_id=snapshot.id,
                bookmaker=book,
                outcome=outcome,
                side=Side.BACK,
                price=Decimal(price),
            )
            for book, outcome, price in rows
        )
        await pglite_async_session.commit()

        migration = _migration()
        moved = await pglite_async_session.run_sync(
            lambda sync: migration.quarantine_stored_outliers(sync.connection())
        )
        await pglite_async_session.commit()

        assert moved == 1
        assert await _williamhill_home(pglite_async_session) == []
        [row] = await _quarantined(pglite_async_session)
        assert (row.snapshot_id, row.bookmaker, row.price) == (
            snapshot.id,
            "williamhill",
            Decimal("20"),
        )
        remaining = (await pglite_async_session.execute(select(Price))).scalars().all()
        assert len(remaining) == 4
