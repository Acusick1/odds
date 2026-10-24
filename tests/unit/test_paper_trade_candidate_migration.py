"""Tests for the e8b1c4d7a2f6 paper-trade decimal price and candidate migration."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.models import Event, EventStatus
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

MOMENT = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "e8b1c4d7a2f6_paper_trade_decimal_price_and_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("e8b1c4d7a2f6_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


class TestUpgradeAndDowngrade:
    @pytest.fixture
    async def legacy_session(self, pglite_async_session: AsyncSession, migration: ModuleType):
        """The session's database with ``paper_trades`` in its pre-migration shape."""
        session = pglite_async_session
        session.add(
            Event(
                id="e1",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=MOMENT,
                home_team="A",
                away_team="B",
                status=EventStatus.SCHEDULED,
            )
        )
        await session.flush()
        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))
        return session

    @staticmethod
    async def _insert_legacy(session: AsyncSession, odds: int) -> None:
        await session.execute(
            text(
                "INSERT INTO paper_trades (event_id, market, selection, bookmaker, odds, stake,"
                " bankroll_before, placed_at) VALUES ('e1', '1x2', 'home', 'bet365', :odds, 10,"
                " 1000, now())"
            ),
            {"odds": odds},
        )

    async def test_existing_trades_convert_exactly_and_keep_null_candidate(
        self, migration: ModuleType, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        for odds in (-110, 150, -150, 100):
            await self._insert_legacy(session, odds)
        connection = await session.connection()

        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))

        rows = (
            await session.execute(
                text(
                    "SELECT price_decimal, candidate_name, candidate_version, git_sha"
                    " FROM paper_trades ORDER BY id"
                )
            )
        ).all()
        assert [r[0] for r in rows] == pytest.approx([1 + 100 / 110, 2.5, 1 + 100 / 150, 2.0])
        assert all(r[1:] == (None, None, None) for r in rows)

    async def test_constraints_after_upgrade(
        self, migration: ModuleType, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))

        insert = (
            "INSERT INTO paper_trades (event_id, market, selection, bookmaker, price_decimal,"
            " stake, bankroll_before, placed_at, candidate_name, candidate_version)"
            " VALUES ('e1', '1x2', 'home', 'bet365', :price, 10, 1000, now(), :name, :version)"
        )
        for params in (
            {"price": 1.0, "name": None, "version": None},
            {"price": 2.0, "name": "cand", "version": None},
            {"price": 2.0, "name": None, "version": 1},
        ):
            with pytest.raises(IntegrityError):
                async with session.begin_nested():
                    await session.execute(text(insert), params)

        await session.execute(text(insert), {"price": 2.0, "name": "cand", "version": 1})

    async def test_downgrade_rounds_decimal_prices_to_american(
        self, migration: ModuleType, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        connection = await session.connection()
        await connection.run_sync(lambda c: _run(c, migration, "upgrade"))
        for price in (1.91, 2.5, 2.0, 1.6667):
            await session.execute(
                text(
                    "INSERT INTO paper_trades (event_id, market, selection, bookmaker,"
                    " price_decimal, stake, bankroll_before, placed_at)"
                    " VALUES ('e1', '1x2', 'home', 'bet365', :p, 10, 1000, now())"
                ),
                {"p": price},
            )

        await connection.run_sync(lambda c: _run(c, migration, "downgrade"))

        odds = (await session.execute(text("SELECT odds FROM paper_trades ORDER BY id"))).scalars()
        assert list(odds) == [-110, 150, 100, -150]
