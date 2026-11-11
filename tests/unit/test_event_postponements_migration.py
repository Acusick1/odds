"""Tests for the e5b7c9d1f3a2 migration: a table of the kick-offs events were postponed from."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
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
        / "e5b7c9d1f3a2_add_event_postponements.py"
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


async def _has_table(session: AsyncSession) -> bool:
    found = await session.execute(
        text("SELECT 1 FROM information_schema.tables WHERE table_name = 'event_postponements'")
    )
    return found.first() is not None


async def _event(session: AsyncSession) -> int:
    return int(
        (
            await session.execute(
                text(
                    "INSERT INTO events (sport_key, sport_title, commence_time, home_team,"
                    " away_team, status, created_at, updated_at) VALUES ('soccer_epl', 'EPL',"
                    " now(), 'A', 'B', 'SCHEDULED', now(), now()) RETURNING id"
                )
            )
        ).scalar_one()
    )


class TestEventPostponementsMigration:
    async def test_downgrade_drops_the_table_and_upgrade_restores_it(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        assert await _has_table(pglite_async_session)

        await _step(pglite_async_session, migration, "downgrade")
        assert not await _has_table(pglite_async_session)

        await _step(pglite_async_session, migration, "upgrade")
        assert await _has_table(pglite_async_session)

    async def test_deleting_an_event_deletes_its_postponement(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        await _step(pglite_async_session, migration, "downgrade")
        await _step(pglite_async_session, migration, "upgrade")
        event_id = await _event(pglite_async_session)
        await pglite_async_session.execute(
            text(
                "INSERT INTO event_postponements (event_id, postponed_from, recorded_at)"
                " VALUES (:i, :t, :t)"
            ),
            {"i": event_id, "t": datetime(2026, 11, 7, tzinfo=UTC)},
        )

        await pglite_async_session.execute(
            text("DELETE FROM events WHERE id = :i"), {"i": event_id}
        )

        left = await pglite_async_session.execute(text("SELECT count(*) FROM event_postponements"))
        assert left.scalar_one() == 0
