"""Tests for the d3a7f1c9e2b4 migration renaming match_briefs and espn_fixtures."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.brief_models import Brief
from odds_core.epl_data_models import EspnGame
from odds_core.job_run_models import JobRun
from odds_core.models import AlertHistory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "d3a7f1c9e2b4_rename_match_briefs_and_espn_fixtures.py"
)
NAMES = (
    "SELECT name FROM ("
    " SELECT relname AS name FROM pg_class"
    " UNION SELECT conname FROM pg_constraint) AS names"
)
OLD_NAMES = " WHERE name ILIKE '%match\\_brief%' OR name ILIKE '%espn\\_fixture%'"
STARTED = datetime(2026, 11, 1, 6, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("d3a7f1c9e2b4_migration", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


async def _names(session: AsyncSession, query: str) -> set[str]:
    return set((await session.execute(text(query))).scalars())


async def _job_names(session: AsyncSession) -> tuple[set[str], set[str]]:
    runs = await _names(session, f"SELECT job_name FROM {JobRun.__tablename__}")
    alerts = await _names(session, f"SELECT alert_type FROM {AlertHistory.__tablename__}")
    return runs, alerts


class TestGameAndBriefRename:
    async def test_downgrade_restores_the_old_names_and_upgrade_removes_them(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = pglite_async_session
        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))

        old = await _names(session, NAMES + OLD_NAMES)
        assert {
            "match_briefs",
            "match_briefs_pkey",
            "match_briefs_event_id_fkey",
            "ix_match_briefs_event_id",
            "match_briefs_id_seq",
            "espn_fixtures",
            "espn_fixtures_pkey",
            "espn_fixtures_id_seq",
            "ix_espn_fixtures_date",
            "ix_espn_fixtures_team",
            "ix_espn_fixtures_season",
            "ix_espn_fixture_team_season",
            "uq_espn_fixture_date_team_comp",
        } <= old
        columns = await _names(
            session,
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'espn_fixtures'",
        )
        assert "match_round" in columns

        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))

        assert await _names(session, NAMES + OLD_NAMES) == set()
        new = await _names(session, NAMES)
        assert {Brief.__tablename__, EspnGame.__tablename__} <= new
        assert {"briefs_id_seq", "espn_games_id_seq", "briefs_event_id_fkey"} <= new

    async def test_the_job_runs_and_alerts_follow_the_job_rename(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = pglite_async_session
        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))
        for name in ("fetch-espn-fixtures", "fetch-espn-fixtures-nba", "fetch-odds"):
            await session.execute(
                text(
                    "INSERT INTO job_runs (job_name, started_at, finished_at, outcome)"
                    " VALUES (:name, :at, :at, 'succeeded')"
                ),
                {"name": name, "at": STARTED},
            )
        for alert_type in ("heartbeat:fetch-espn-fixtures-nba", "heartbeat:fetch-odds"):
            await session.execute(
                text(
                    "INSERT INTO alert_history (alert_type, severity, message, sent_at)"
                    " VALUES (:alert_type, 'info', '', :at)"
                ),
                {"alert_type": alert_type, "at": STARTED},
            )

        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))

        assert await _job_names(session) == (
            {"fetch-espn-games", "fetch-espn-games-nba", "fetch-odds"},
            {"heartbeat:fetch-espn-games-nba", "heartbeat:fetch-odds"},
        )

        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))

        assert await _job_names(session) == (
            {"fetch-espn-fixtures", "fetch-espn-fixtures-nba", "fetch-odds"},
            {"heartbeat:fetch-espn-fixtures-nba", "heartbeat:fetch-odds"},
        )
