"""Tests for the a5c8d2e91f37 migration that makes price snapshots unique per market and time."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_lambda.storage.polymarket_writer import PolymarketWriter
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

TABLE = "polymarket_price_snapshots"


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "a5c8d2e91f37_unique_polymarket_price_snapshot_per_market_time.py"
    )
    spec = importlib.util.spec_from_file_location("a5c8d2e91f37_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


class TestUpgrade:
    @pytest.fixture
    async def legacy_session(self, pglite_async_session: AsyncSession) -> AsyncSession:
        session = pglite_async_session
        await session.execute(text(f"ALTER TABLE {TABLE} DROP CONSTRAINT uq_pm_price_market_time"))
        for column in ("best_bid", "best_ask", "spread", "midpoint"):
            await session.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {column} double precision"))
        await session.execute(
            text(
                "CREATE INDEX ix_pm_price_market_time"
                f" ON {TABLE} (polymarket_market_id, snapshot_time)"
            )
        )
        return session

    @staticmethod
    async def _market_ids(session: AsyncSession) -> tuple[int, int]:
        writer = PolymarketWriter(session)
        event = await writer.upsert_event(
            {
                "id": "e1",
                "ticker": "nba-a-b-2026-02-14",
                "slug": "s",
                "title": "t",
                "startDate": "2026-02-14T01:00:00Z",
                "endDate": "2026-02-14T04:00:00Z",
                "markets": [],
            }
        )
        markets = await writer.upsert_markets(
            event.id,
            [
                {
                    "id": f"m{i}",
                    "conditionId": f"c{i}",
                    "question": "q",
                    "clobTokenIds": ["a", "b"],
                    "outcomes": ["A", "B"],
                }
                for i in (1, 2)
            ],
            event.title,
        )
        assert markets[0].id is not None and markets[1].id is not None
        return markets[0].id, markets[1].id

    @pytest.mark.asyncio
    async def test_keeps_lowest_id_of_each_duplicate_and_enforces_uniqueness(
        self, legacy_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = legacy_session
        m1, m2 = await self._market_ids(session)
        for market_id, time in (
            (m1, "2026-02-13 20:00:00+00"),
            (m1, "2026-02-13 20:00:00+00"),
            (m1, "2026-02-13 20:00:00+00"),
            (m1, "2026-02-13 21:00:00+00"),
            (m2, "2026-02-13 20:00:00+00"),
        ):
            await session.execute(
                text(
                    f"INSERT INTO {TABLE} (polymarket_market_id, snapshot_time,"
                    " outcome_0_price, outcome_1_price)"
                    f" VALUES ({market_id}, '{time}', 0.5, 0.5)"
                )
            )
        ids = (await session.execute(text(f"SELECT id FROM {TABLE} ORDER BY id"))).scalars().all()

        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))

        remaining = (
            (await session.execute(text(f"SELECT id FROM {TABLE} ORDER BY id"))).scalars().all()
        )
        assert remaining == [ids[0], ids[3], ids[4]]

        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(
                    text(
                        f"INSERT INTO {TABLE} (polymarket_market_id, snapshot_time,"
                        " outcome_0_price, outcome_1_price)"
                        f" VALUES ({m1}, '2026-02-13 21:00:00+00', 0.5, 0.5)"
                    )
                )

    @pytest.mark.asyncio
    async def test_drops_the_book_columns(
        self, legacy_session: AsyncSession, migration: ModuleType
    ) -> None:
        connection = await legacy_session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))
        columns = (
            (
                await legacy_session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns"
                        f" WHERE table_name = '{TABLE}'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {"best_bid", "best_ask", "spread", "midpoint"}.isdisjoint(columns)

    @pytest.mark.asyncio
    async def test_downgrade_restores_columns_and_index(
        self, legacy_session: AsyncSession, migration: ModuleType
    ) -> None:
        connection = await legacy_session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))
        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))

        columns = (
            (
                await legacy_session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns"
                        f" WHERE table_name = '{TABLE}'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {"best_bid", "best_ask", "spread", "midpoint"} <= set(columns)
        constraints = (
            (
                await legacy_session.execute(
                    text(
                        "SELECT conname FROM pg_constraint WHERE conname = 'uq_pm_price_market_time'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert constraints == []
        indexes = (
            (
                await legacy_session.execute(
                    text(
                        "SELECT indexname FROM pg_indexes WHERE indexname = 'ix_pm_price_market_time'"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert indexes == ["ix_pm_price_market_time"]
