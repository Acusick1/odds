"""The web UI against a real database under the ``odds_web`` role and its grants.

PGlite serves one connection, so these tests share the suite's engine and take the role with
``SET LOCAL ROLE`` at the start of each transaction, where the host connects as it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from odds_core import (  # noqa: F401  (tables the role must not read)
    agent_wakeup_models,
    match_brief_models,
)
from odds_core.models import EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_web.app import create_app
from odds_web.db import WebDatabase, restrict_transactions
from odds_web.role_check import RoleCheckFailed, check_role
from sqlalchemy import Connection, event, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
)

GRANTS = Path(__file__).parents[2] / "scripts" / "odds_web_grants.sql"
KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
# The APScheduler tables exist only once the scheduler has booted; the grants file names them.
SCHEDULER_TABLES = ("schedules", "jobs", "job_results", "tasks")


@pytest.fixture
async def session(pglite_async_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """The suite's empty database with the ``odds_web`` role and its grants applied."""
    for table in SCHEDULER_TABLES:
        await pglite_async_session.execute(text(f"CREATE TABLE IF NOT EXISTS {table} (id integer)"))
    await pglite_async_session.execute(
        text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'odds_web') "
            "THEN CREATE ROLE odds_web NOLOGIN; END IF; END $$"
        )
    )
    for statement in GRANTS.read_text().split(";"):
        if statement.strip():
            await pglite_async_session.execute(text(statement))
    await pglite_async_session.commit()
    yield pglite_async_session


def client_of(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@contextmanager
def as_odds_web(engine: AsyncEngine) -> Iterator[None]:
    """Every transaction on ``engine`` runs as the ``odds_web`` role."""

    def set_role(connection: Connection) -> None:
        connection.exec_driver_sql("SET LOCAL ROLE odds_web")

    event.listen(engine.sync_engine, "begin", set_role)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "begin", set_role)


@contextmanager
def as_odds_web_login(engine: AsyncEngine) -> Iterator[None]:
    """Like ``as_odds_web``, with the role-level ``default_transaction_read_only`` a login applies.

    ``SET ROLE`` does not apply ``ALTER ROLE ... SET``, so the setting is read back from
    ``pg_db_role_setting`` and applied to each transaction as the server does for a new session.
    """

    def set_role(connection: Connection) -> None:
        defaults = connection.exec_driver_sql(
            "SELECT s.setconfig FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole "
            "WHERE r.rolname = 'odds_web'"
        ).scalar()
        connection.exec_driver_sql("SET LOCAL ROLE odds_web")
        if defaults and "default_transaction_read_only=on" in defaults:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")

    event.listen(engine.sync_engine, "begin", set_role)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "begin", set_role)


class TestRoleCheck:
    async def test_passes_for_the_granted_role_that_a_login_makes_read_only(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                report = await check_role(conn)

        assert "INSERT rejected (SQLSTATE 25006)" in report

    async def test_passes_when_only_the_privileges_reject_the_write(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web(engine):
            async with engine.connect() as conn:
                report = await check_role(conn)

        assert "INSERT rejected (SQLSTATE 42501)" in report

    async def test_fails_when_the_role_can_write(self, session: AsyncSession) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        await session.execute(text("GRANT INSERT ON paper_trades TO odds_web"))
        await session.commit()

        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                with pytest.raises(RoleCheckFailed, match="write privileges on: paper_trades"):
                    await check_role(conn)

    async def test_fails_when_the_role_cannot_read_what_the_ui_needs(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        await session.execute(text("REVOKE SELECT ON paper_trades FROM odds_web"))
        await session.commit()

        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                with pytest.raises(RoleCheckFailed, match="cannot SELECT from paper_trades"):
                    await check_role(conn)


class TestGrants:
    async def test_the_role_reads_the_listed_tables_and_nothing_else(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
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
    async def test_the_role_cannot_write(self, session: AsyncSession, write: str) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web(engine):
            async with engine.connect() as conn:
                with pytest.raises(ProgrammingError, match="permission denied"):
                    await conn.execute(text(write))


class TestRestrictedTransactions:
    async def test_every_transaction_carries_the_statement_timeout(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        restrict = restrict_transactions(engine, 200)
        try:
            async with engine.connect() as conn:
                timeout = await conn.scalar(text("SELECT current_setting('statement_timeout')"))
        finally:
            event.remove(engine.sync_engine, "begin", restrict)
        assert timeout == "200ms"

    async def test_a_transaction_is_read_only_even_for_a_superuser(
        self, session: AsyncSession
    ) -> None:
        engine = session.bind
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
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        await seed_one_bet(session)
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                body = (await client.get("/")).text

        assert "exchange-gap-epl" in body
        assert "+1.2u" in body
        assert "Unavailable" not in body
