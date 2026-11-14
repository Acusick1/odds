"""``odds_web.grants`` applied to a real database: what the role can read after it, and what it loses."""

from __future__ import annotations

import pytest
from odds_core import (  # noqa: F401  (every table of the allowlist, and some outside it, must exist)
    agent_wakeup_models,
    candidate_backtest_models,
    match_brief_models,
    mlb_data_models,
    models,
    paper_trade_models,
)
from odds_web import grants
from odds_web.grants import GrantsFailed, apply_grants
from odds_web.role_check import READ_TABLES, check_role
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from tests.test_helpers import as_odds_web_login


async def readable_tables(session: AsyncSession) -> set[str]:
    rows = await session.scalars(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "AND has_table_privilege('odds_web', format('public.%I', table_name), 'SELECT')"
        )
    )
    return set(rows)


async def other_tables(session: AsyncSession) -> list[str]:
    rows = await session.scalars(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "AND table_name <> ALL(:allowed)"
        ),
        {"allowed": list(READ_TABLES)},
    )
    return sorted(rows)


class TestApplyGrants:
    async def test_the_role_reads_exactly_the_allowlist_and_passes_the_role_check(
        self, odds_web_session: AsyncSession
    ) -> None:
        assert await readable_tables(odds_web_session) == set(READ_TABLES)
        engine = odds_web_session.bind
        assert isinstance(engine, AsyncEngine)
        with as_odds_web_login(engine):
            async with engine.connect() as conn:
                await check_role(conn)

    async def test_is_safe_to_re_run(self, odds_web_session: AsyncSession) -> None:
        await apply_grants(await odds_web_session.connection())
        await odds_web_session.commit()

        assert await readable_tables(odds_web_session) == set(READ_TABLES)

    async def test_revokes_every_privilege_on_a_table_outside_the_allowlist(
        self, odds_web_session: AsyncSession
    ) -> None:
        extras = await other_tables(odds_web_session)
        assert extras
        for table in extras:
            await odds_web_session.execute(text(f"GRANT SELECT, INSERT ON {table} TO odds_web"))
        await odds_web_session.execute(text("GRANT INSERT, UPDATE ON paper_trades TO odds_web"))
        await odds_web_session.commit()

        await apply_grants(await odds_web_session.connection())
        await odds_web_session.commit()

        assert await readable_tables(odds_web_session) == set(READ_TABLES)
        assert not await odds_web_session.scalar(
            text("SELECT has_table_privilege('odds_web', 'paper_trades', 'INSERT, UPDATE')")
        )

    async def test_a_table_dropped_from_the_allowlist_loses_its_grant(
        self, odds_web_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(grants, "READ_TABLES", READ_TABLES[1:])

        await apply_grants(await odds_web_session.connection())
        await odds_web_session.commit()

        assert await readable_tables(odds_web_session) == set(READ_TABLES[1:])

    async def test_makes_the_role_read_only_by_default(
        self, odds_web_session: AsyncSession
    ) -> None:
        settings = await odds_web_session.scalar(
            text(
                "SELECT s.setconfig FROM pg_db_role_setting s "
                "JOIN pg_roles r ON r.oid = s.setrole WHERE r.rolname = 'odds_web'"
            )
        )

        assert settings is not None and "default_transaction_read_only=on" in settings

    async def test_a_missing_role_fails_naming_it(
        self, pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(grants, "ROLE", "odds_web_not_created")

        with pytest.raises(GrantsFailed, match="odds_web_not_created does not exist"):
            await apply_grants(await pglite_async_session.connection())
