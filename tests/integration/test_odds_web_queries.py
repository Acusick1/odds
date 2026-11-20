"""The query statistics reader and the /perf page, over the ``odds_web`` role's statistics
functions and a stand-in for ``pg_stat_statements``."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from odds_web.app import create_app
from odds_web.db import WebDatabase, build_web_database
from odds_web.perf import load_query_stats
from odds_web.role_check import RoleCheckFailed, check_role
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import as_odds_web_login, client_of, record_statement

RESET = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)


@pytest.fixture
async def recorded(odds_web_session: AsyncSession) -> AsyncSession:
    """Four statements in this database and one in another, with the statistics' reset time."""
    session = odds_web_session
    await session.execute(
        text("INSERT INTO pg_stat_statements_info VALUES (0, :reset)"), {"reset": RESET}
    )
    # (queryid, query, calls, total ms): "many" has the most total time, "slow" the highest mean.
    for queryid, query, calls, total in (
        (1, "SELECT many", 1000, 9000.0),
        (2, "SELECT slow", 2, 4000.0),
        (3, "SELECT some", 10, 500.0),
        (4, "SELECT few", 1, 1.0),
    ):
        await record_statement(session, queryid, query, calls, total, rows=calls)
    await record_statement(session, 5, "SELECT elsewhere", 1, 99_999.0, dbid=0)
    await session.commit()
    return session


def engine_of(session: AsyncSession) -> AsyncEngine:
    engine = session.bind
    assert isinstance(engine, AsyncEngine)
    return engine


class TestLoadQueryStats:
    async def test_ranks_this_databases_statements_by_total_and_by_mean_time(
        self, recorded: AsyncSession
    ) -> None:
        engine = engine_of(recorded)
        with as_odds_web_login(engine):
            async with async_sessionmaker(engine)() as reader:
                stats = await load_query_stats(reader, limit=3)

        assert [q.query for q in stats.by_total] == ["SELECT many", "SELECT slow", "SELECT some"]
        assert [q.query for q in stats.by_mean] == ["SELECT slow", "SELECT some", "SELECT many"]
        many = stats.by_total[0]
        assert (many.calls, many.total_ms, many.mean_ms, many.rows) == (1000, 9000.0, 9.0, 1000)
        assert many.role == await recorded.scalar(text("SELECT current_user"))
        assert stats.reset_at == RESET

    async def test_the_role_cannot_read_the_statistics_tables_directly(
        self, recorded: AsyncSession
    ) -> None:
        for view in ("pg_stat_statements", "pg_stat_statements_info"):
            assert not await recorded.scalar(
                text("SELECT has_table_privilege('odds_web', :view, 'SELECT')"), {"view": view}
            )


class TestRoleCheck:
    async def test_a_role_that_cannot_execute_the_statistics_functions_fails(
        self, odds_web_session: AsyncSession
    ) -> None:
        await odds_web_session.execute(
            text("REVOKE EXECUTE ON FUNCTION odds_web_query_stats() FROM odds_web")
        )
        await odds_web_session.commit()
        engine = engine_of(odds_web_session)

        with as_odds_web_login(engine), pytest.raises(RoleCheckFailed, match="cannot execute"):
            async with engine.connect() as connection:
                await check_role(connection)

    async def test_a_missing_extension_fails(self, odds_web_session: AsyncSession) -> None:
        await odds_web_session.execute(text("DROP TABLE pg_stat_statements"))
        await odds_web_session.commit()
        engine = engine_of(odds_web_session)

        with (
            as_odds_web_login(engine),
            pytest.raises(RoleCheckFailed, match="cannot read the query statistics"),
        ):
            async with engine.connect() as connection:
                await check_role(connection)

    @pytest.mark.parametrize("predefined", ["pg_read_all_stats", "pg_write_all_data"])
    async def test_membership_of_a_widening_predefined_role_fails(
        self, odds_web_session: AsyncSession, predefined: str
    ) -> None:
        await odds_web_session.execute(text(f"GRANT {predefined} TO odds_web"))
        await odds_web_session.commit()
        engine = engine_of(odds_web_session)

        try:
            with as_odds_web_login(engine), pytest.raises(RoleCheckFailed, match=predefined):
                async with engine.connect() as connection:
                    await check_role(connection)
        finally:
            await odds_web_session.execute(text(f"REVOKE {predefined} FROM odds_web"))
            await odds_web_session.commit()


class TestPerfPage:
    async def test_lists_the_top_statements_with_calls_and_the_reset_time(
        self, recorded: AsyncSession, tmp_path: Path
    ) -> None:
        engine = engine_of(recorded)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))
        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                page = (await client.get("/perf")).text
                fragment = (await client.get("/fragments/queries")).text

        assert re.search(r"as of \d\d:\d\d UTC", page)
        assert "Statistics reset 2026-11-01 06:00 UTC" in page
        totals, means = page.split("By mean time")
        assert totals.index("SELECT many") < totals.index("SELECT slow")
        assert means.index("SELECT slow") < means.index("SELECT many")
        assert '<td class="r num">1,000</td>' in page and "9.0 s" in page
        assert "SELECT elsewhere" not in page
        assert 'href="/perf"' in page
        assert 'id="queries"' in fragment and "<html" not in fragment

    async def test_a_missing_extension_renders_the_section_error(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        await odds_web_session.execute(text("DROP TABLE pg_stat_statements"))
        await odds_web_session.commit()
        engine = engine_of(odds_web_session)
        database = WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))

        with as_odds_web_login(engine):
            async with client_of(create_app(database, tmp_path)) as client:
                response = await client.get("/perf")

        assert response.status_code == 200
        assert "odds-web error" in response.text and "no data loaded yet" in response.text

    async def test_a_database_that_is_down_renders_unavailable(self, tmp_path: Path) -> None:
        database = build_web_database("postgresql+asyncpg://odds_web:x@127.0.0.1:1/odds")
        try:
            async with client_of(create_app(database, tmp_path)) as client:
                response = await client.get("/perf")
        finally:
            await database.engine.dispose()

        assert response.status_code == 200
        assert "Database unavailable · no data loaded yet" in response.text
