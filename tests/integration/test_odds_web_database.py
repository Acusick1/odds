"""The web UI against a real database under the ``odds_web`` role and its grants.

PGlite serves one connection, so these tests share the suite's engine and take the role with
``SET LOCAL ROLE`` at the start of each transaction, where the host connects as it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from odds_core import (  # noqa: F401  (tables the role must not read)
    agent_wakeup_models,
    match_brief_models,
)
from odds_core.models import EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_web.app import create_app
from odds_web.db import WebDatabase, restrict_transactions
from odds_web.decisions import load_decision_history
from odds_web.ledger import load_ledger_page
from odds_web.role_check import READ_TABLES, RoleCheckFailed, check_role
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import (
    as_odds_web,
    as_odds_web_login,
    betfair_exchange_bookmaker,
    client_of,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
)

KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)


class TestRoleCheck:
    async def test_passes_for_the_granted_role_that_a_login_makes_read_only(
        self, odds_web_session: AsyncSession
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                report = await check_role(conn)

        assert "INSERT rejected (SQLSTATE 25006)" in report
        assert any(f"can SELECT from {len(READ_TABLES)} tables" in line for line in report)

    async def test_passes_when_only_the_privileges_reject_the_write(
        self, odds_web_session: AsyncSession
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web(engine):
            async with engine.connect() as conn:
                report = await check_role(conn)

        assert "INSERT rejected (SQLSTATE 42501)" in report

    async def test_fails_when_the_role_can_write(self, odds_web_session: AsyncSession) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        await odds_web_session.execute(text("GRANT INSERT ON paper_trades TO odds_web"))
        await odds_web_session.commit()

        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                with pytest.raises(RoleCheckFailed, match="write privileges on: paper_trades"):
                    await check_role(conn)

    @pytest.mark.parametrize("table", READ_TABLES)
    async def test_fails_when_the_role_cannot_read_a_table_the_ui_needs(
        self, odds_web_session: AsyncSession, table: str
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        await odds_web_session.execute(text(f"REVOKE SELECT ON {table} FROM odds_web"))
        await odds_web_session.commit()

        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                with pytest.raises(RoleCheckFailed, match=f"cannot SELECT from {table}"):
                    await check_role(conn)


class TestGrants:
    async def test_the_role_reads_the_listed_tables_and_nothing_else(
        self, odds_web_session: AsyncSession
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web(engine):
            for listed in ("paper_trades", "schedules", "odds_snapshots"):
                async with engine.connect() as conn:
                    await conn.execute(text(f"SELECT count(*) FROM {listed}"))
            for unlisted in ("agent_wakeups", "match_briefs", "snapshot_payloads"):
                async with engine.connect() as conn:
                    with pytest.raises(ProgrammingError, match="permission denied"):
                        await conn.execute(text(f"SELECT count(*) FROM {unlisted}"))

    @pytest.mark.parametrize(
        "write",
        [
            "INSERT INTO paper_trades (event_id) VALUES (1)",
            "UPDATE events SET home_team = 'x'",
            "DELETE FROM candidate_decisions",
            "TRUNCATE paper_trades",
        ],
    )
    async def test_the_role_cannot_write(self, odds_web_session: AsyncSession, write: str) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web(engine):
            async with engine.connect() as conn:
                with pytest.raises(ProgrammingError, match="permission denied"):
                    await conn.execute(text(write))


class TestRestrictedTransactions:
    async def test_every_transaction_carries_the_statement_timeout(
        self, odds_web_session: AsyncSession
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        restrict = restrict_transactions(engine, 200)
        try:
            async with engine.connect() as conn:
                timeout = await conn.scalar(text("SELECT current_setting('statement_timeout')"))
        finally:
            event.remove(engine.sync_engine, "begin", restrict)
        assert timeout == "200ms"

    async def test_a_transaction_is_read_only_even_for_a_superuser(
        self, odds_web_session: AsyncSession
    ) -> None:
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        restrict = restrict_transactions(engine, 20_000)
        try:
            async with engine.connect() as conn:
                with pytest.raises(DBAPIError, match="read-only transaction"):
                    await conn.execute(text("CREATE TABLE should_not_exist (id integer)"))
        finally:
            event.remove(engine.sync_engine, "begin", restrict)


async def seed_one_bet(session: AsyncSession) -> None:
    session.add(make_epl_event(1, KICKOFF, status=EventStatus.FINAL))
    await session.flush()
    await store_tagged(
        session,
        make_tagged_snapshot(
            1,
            KICKOFF - timedelta(minutes=10),
            [betfair_exchange_bookmaker((2.0, 4.0, 4.0))],
            "betfair_api",
        ),
    )
    session.add(
        PaperTrade(
            event_id=1,
            market="1x2",
            selection="home",
            bookmaker="bet365",
            price_decimal=2.2,
            stake=1.0,
            candidate_name="exchange-gap-epl",
            candidate_version=1,
            placed_at=KICKOFF - timedelta(days=1),
            settled_at=KICKOFF + timedelta(hours=3),
            result=TradeResult.WIN,
            pnl=1.2,
        )
    )
    await session.commit()


class TestPortfolioUnderTheRole:
    async def test_renders_from_the_real_tables(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        await seed_one_bet(odds_web_session)
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                body = (await client.get("/")).text

        assert "exchange-gap-epl" in body
        assert "+1.2u" in body
        assert 'class="unavailable"' not in body


class TestCandidatePageUnderTheRole:
    async def test_renders_the_ledger_and_trade_from_the_real_tables(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        await seed_one_bet(odds_web_session)
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                response = await client.get("/c/exchange-gap-epl/1")

        assert response.status_code == 200
        assert "Arsenal v Chelsea" in response.text and "bet365" in response.text
        assert "+1.2u" in response.text and 'class="unavailable"' not in response.text

    async def test_the_readers_need_no_table_beyond_the_grants(
        self, odds_web_session: AsyncSession
    ) -> None:
        await seed_one_bet(odds_web_session)
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web(engine):
            async with database.session_maker() as reader:
                ledger = await load_ledger_page(reader, "exchange-gap-epl", 1)
            async with database.session_maker() as reader:
                decisions = await load_decision_history(reader, "exchange-gap-epl", 1, KICKOFF)

        assert ledger.total == 1 and ledger.rows[0].p_close == pytest.approx(0.5)
        assert decisions.total == 0
