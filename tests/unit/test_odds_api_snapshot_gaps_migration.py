"""Tests for the d2f8a4c6e1b9 migration: a table of the planned Odds API snapshots the provider
does not have."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.backfill_models import OddsApiSnapshotGap
from odds_core.vocabulary import Market
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

PLANNED = datetime(2025, 4, 15, 21, 25, 38, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "d2f8a4c6e1b9_add_odds_api_snapshot_gaps.py"
    )
    spec = importlib.util.spec_from_file_location("md2f8a4c6e1b9", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


async def _migrated(session: AsyncSession, migration: ModuleType) -> None:
    connection = await session.connection()
    for step in ("downgrade", "upgrade"):
        await connection.run_sync(lambda c, step=step: _run(c, migration, step))


class TestOddsApiSnapshotGapsMigration:
    async def test_the_migrated_table_stores_the_model(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = pglite_async_session
        await _migrated(session, migration)
        session.add(
            OddsApiSnapshotGap(
                sport_key="soccer_germany_bundesliga",
                market=Market.ONE_X_TWO,
                planned_snapshot=PLANNED,
                returned_snapshot=datetime(2025, 4, 15, 18, 0, 38, tzinfo=UTC),
            )
        )
        await session.flush()

        stored = (await session.execute(select(OddsApiSnapshotGap))).scalars().one()
        assert (stored.market, stored.next_snapshot) == (Market.ONE_X_TWO, None)

    async def test_an_unknown_market_is_refused(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        await _migrated(pglite_async_session, migration)

        with pytest.raises(IntegrityError, match="ck_odds_api_snapshot_gaps_market"):
            await pglite_async_session.execute(
                text(
                    "INSERT INTO odds_api_snapshot_gaps (sport_key, market, planned_snapshot,"
                    " returned_snapshot, recorded_at) VALUES ('soccer_epl', 'btts', :t, :t, :t)"
                ),
                {"t": PLANNED},
            )
