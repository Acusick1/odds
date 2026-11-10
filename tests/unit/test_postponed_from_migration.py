"""Tests for the e5b7c9d1f3a2 migration: events record the kick-off they were postponed from."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "e5b7c9d1f3a2_add_events_postponed_from.py"
    )
    spec = importlib.util.spec_from_file_location("me5b7c9d1f3a2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


async def _step(session: AsyncSession, migration: ModuleType, step: str) -> None:
    connection = await session.connection()
    await connection.run_sync(lambda c: _run(c, migration, step))


async def _has_column(session: AsyncSession) -> bool:
    found = await session.execute(
        text(
            "SELECT 1 FROM information_schema.columns"
            " WHERE table_name = 'events' AND column_name = 'postponed_from'"
        )
    )
    return found.first() is not None


class TestPostponedFromMigration:
    async def test_downgrade_drops_the_column_and_upgrade_restores_it(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        assert await _has_column(pglite_async_session)

        await _step(pglite_async_session, migration, "downgrade")
        assert not await _has_column(pglite_async_session)

        await _step(pglite_async_session, migration, "upgrade")
        assert await _has_column(pglite_async_session)

    async def test_existing_events_have_no_recorded_postponement(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        await _step(pglite_async_session, migration, "downgrade")
        await pglite_async_session.execute(
            text(
                "INSERT INTO events (sport_key, sport_title, commence_time, home_team, away_team,"
                " status, created_at, updated_at) VALUES ('soccer_epl', 'EPL', now(), 'A', 'B',"
                " 'SCHEDULED', now(), now())"
            )
        )
        await _step(pglite_async_session, migration, "upgrade")

        value = (
            await pglite_async_session.execute(text("SELECT postponed_from FROM events"))
        ).scalar_one()
        assert value is None
