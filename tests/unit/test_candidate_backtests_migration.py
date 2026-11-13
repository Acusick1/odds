"""Tests for the a4c8e2f61b93 migration: the stored backtest runs and their bets."""

from __future__ import annotations

import importlib.util
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.candidate_backtest_models import CandidateBacktest, CandidateBacktestBet, WindowKind
from odds_core.vocabulary import Source
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

WHEN = datetime(2026, 11, 13, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "a4c8e2f61b93_add_candidate_backtests.py"
    )
    spec = importlib.util.spec_from_file_location("ma4c8e2f61b93", path)
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


def backtest(**changes: Any) -> CandidateBacktest:
    fields: dict[str, Any] = {
        "candidate_name": "idea",
        "candidate_version": 1,
        "issue": 900,
        "market": "1x2",
        "rule": "exchange_gap",
        "params": {"threshold": 0.02},
        "window_kind": WindowKind.HOLDOUT,
        "start_date": date(2025, 8, 1),
        "end_date": date(2026, 7, 31),
        "leagues": ["soccer_spain_la_liga"],
        "git_sha": "a" * 40,
        "events": 1,
        "bets": 1,
        "bets_with_close": 1,
        "stake": 1.0,
        "pnl": 1.0,
        "mean_ev": 0.03,
        "ci_lower": 0.01,
        "ci_upper": 0.05,
        "ci_level": 0.95,
        **changes,
    }
    return CandidateBacktest(**fields)


class TestCandidateBacktestsMigration:
    async def test_the_migrated_tables_store_the_models(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = pglite_async_session
        await _migrated(session, migration)
        run = backtest()
        session.add(run)
        await session.flush()
        session.add(
            CandidateBacktestBet(
                backtest_id=run.id,
                position=1,
                event_id=7,
                league="soccer_spain_la_liga",
                kickoff=WHEN,
                decision_time=WHEN,
                outcome="home",
                bookmaker="bet365",
                price=2.0,
                stake=1.0,
                close_source=Source.ODDS_API,
            )
        )
        await session.flush()

        stored = (await session.execute(select(CandidateBacktest))).scalars().one()
        bet = (await session.execute(select(CandidateBacktestBet))).scalars().one()
        assert (stored.window_kind, stored.selected, stored.cell_index) == (
            WindowKind.HOLDOUT,
            False,
            None,
        )
        assert stored.leagues == ["soccer_spain_la_liga"] and stored.params == {"threshold": 0.02}
        assert (bet.close_source, bet.pnl, bet.line) == (Source.ODDS_API, None, None)

    async def test_deleting_a_run_deletes_its_bets(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        session = pglite_async_session
        await _migrated(session, migration)
        run = backtest()
        session.add(run)
        await session.flush()
        await session.execute(
            text(
                "INSERT INTO candidate_backtest_bets (backtest_id, position, event_id, league,"
                " kickoff, decision_time, outcome, bookmaker, price, stake)"
                " VALUES (:id, 1, 7, 'l', :t, :t, 'home', 'b', 2.0, 1.0)"
            ),
            {"id": run.id, "t": WHEN},
        )

        await session.execute(text("DELETE FROM candidate_backtests"))

        assert await session.scalar(select(func.count()).select_from(CandidateBacktestBet)) == 0

    @pytest.mark.parametrize(
        ("changes", "constraint"),
        [
            ({"end_date": date(2025, 7, 31)}, "ck_candidate_backtests_range"),
            ({"cell_index": 0}, "ck_candidate_backtests_cell"),
            ({"ci_level": None}, "ck_candidate_backtests_estimate"),
        ],
    )
    async def test_an_inconsistent_run_is_refused(
        self,
        pglite_async_session: AsyncSession,
        migration: ModuleType,
        changes: dict[str, Any],
        constraint: str,
    ) -> None:
        await _migrated(pglite_async_session, migration)
        pglite_async_session.add(backtest(**changes))

        with pytest.raises(IntegrityError, match=constraint):
            await pglite_async_session.flush()

    async def test_a_run_of_the_same_identity_is_refused_even_without_a_grid_cell(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        await _migrated(pglite_async_session, migration)
        pglite_async_session.add_all([backtest(), backtest()])

        with pytest.raises(IntegrityError, match="uq_candidate_backtests_run"):
            await pglite_async_session.flush()

    async def test_the_database_refuses_a_window_kind_the_model_does_not_name(
        self, pglite_async_session: AsyncSession, migration: ModuleType
    ) -> None:
        await _migrated(pglite_async_session, migration)

        with pytest.raises(IntegrityError, match="ck_candidate_backtests_window_kind"):
            await pglite_async_session.execute(
                text(
                    "INSERT INTO candidate_backtests (candidate_name, candidate_version, issue,"
                    " market, rule, params, window_kind, start_date, end_date, leagues, git_sha,"
                    " events, bets, bets_with_close, stake, pnl, created_at) VALUES ('i', 1, 1,"
                    " '1x2', 'r', '{}', 'live', :d, :d, '{}', 'a', 0, 0, 0, 0, 0, :t)"
                ),
                {"d": date(2025, 8, 1), "t": WHEN},
            )
