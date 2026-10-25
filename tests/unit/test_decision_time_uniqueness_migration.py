"""Tests for the 152a63c0fae5 migration: decisions unique per decision lead, bets per outcome."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
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

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "152a63c0fae5_candidate_decisions_per_decision_time.py"
    )
    spec = importlib.util.spec_from_file_location("m152a63c0fae5", path)
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


async def _decide(session: AsyncSession, minutes: int, with_lead: bool = True) -> int:
    lead_column, lead_value = (", minutes_before_kickoff", ", :m") if with_lead else ("", "")
    decision_id = (
        await session.execute(
            text(
                "INSERT INTO candidate_decisions (candidate_name, candidate_version, event_id,"
                f" decision_time, kickoff_at_decision, known_at, bets_placed{lead_column})"
                f" VALUES ('cand', 1, 'e1', :t, :k, :t, 0{lead_value}) RETURNING id"
            ),
            {"t": KICKOFF - timedelta(minutes=minutes), "k": KICKOFF, "m": minutes},
        )
    ).scalar_one()
    return int(decision_id)


async def _bet(session: AsyncSession, decision_id: int, selection: str, bookmaker: str) -> None:
    await session.execute(
        text(
            "INSERT INTO paper_trades (event_id, market, selection, bookmaker, price_decimal,"
            " stake, placed_at, candidate_name, candidate_version, decision_id)"
            " VALUES ('e1', '1x2', :s, :b, 2.0, 1, now(), 'cand', 1, :d)"
        ),
        {"s": selection, "b": bookmaker, "d": decision_id},
    )


class TestUpgradeAndDowngrade:
    @pytest.fixture
    async def session(self, pglite_async_session: AsyncSession) -> AsyncSession:
        pglite_async_session.add(
            Event(
                id="e1",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=KICKOFF,
                home_team="A",
                away_team="B",
                status=EventStatus.SCHEDULED,
            )
        )
        await pglite_async_session.flush()
        return pglite_async_session

    async def test_upgrade_backfills_the_lead_of_existing_decisions(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await _decide(session, 60, with_lead=False)

        await _step(session, migration, "upgrade")

        lead = (
            await session.execute(text("SELECT minutes_before_kickoff FROM candidate_decisions"))
        ).scalar_one()
        assert lead == 60

    async def test_upgrade_allows_several_leads_and_one_bet_per_outcome(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await _step(session, migration, "upgrade")

        early = await _decide(session, 90)
        later = await _decide(session, 60)
        await _bet(session, early, "home", "bet365")
        await _bet(session, later, "away", "bet365")
        for attempt in (
            lambda: _decide(session, 60),
            lambda: _bet(session, later, "home", "betway"),
        ):
            with pytest.raises(IntegrityError):
                async with session.begin_nested():
                    await attempt()

    async def test_downgrade_keeps_each_versions_earliest_decision_on_an_event(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        early = await _decide(session, 90)
        later = await _decide(session, 60)
        await _bet(session, early, "home", "bet365")
        await _bet(session, later, "away", "bet365")

        await _step(session, migration, "downgrade")

        decisions = (await session.execute(text("SELECT id FROM candidate_decisions"))).scalars()
        trades = (await session.execute(text("SELECT decision_id FROM paper_trades"))).scalars()
        assert (list(decisions), list(trades)) == ([early], [early])
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await _decide(session, 30, with_lead=False)
