"""Tests for the d5a3f7c19e24 migration renaming paper_trades to bets."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.unit.conftest import run_bet_rename


async def _names(session: AsyncSession, query: str) -> set[str]:
    return set((await session.execute(text(query))).scalars())


INDEXES = "SELECT indexname FROM pg_indexes WHERE tablename = :t"


class TestBetRename:
    async def test_downgrade_renames_every_object_to_the_paper_trade_names(
        self, paper_trades_session: AsyncSession
    ) -> None:
        session = paper_trades_session
        indexes = await _names(
            session, "SELECT indexname FROM pg_indexes WHERE tablename = 'paper_trades'"
        )
        constraints = await _names(
            session,
            "SELECT conname FROM pg_constraint WHERE conrelid = 'paper_trades'::regclass",
        )
        sequences = await _names(session, "SELECT relname FROM pg_class WHERE relkind = 'S'")
        enums = await _names(session, "SELECT typname FROM pg_type WHERE typtype = 'e'")

        assert "paper_trades_pkey" in indexes
        assert {"ix_paper_trades_unsettled", "uq_paper_trades_candidate_bet"} <= indexes
        assert {"paper_trades_event_id_fkey", "fk_paper_trades_decision_id"} <= constraints
        assert "ck_paper_trades_candidate_pair" in constraints
        assert "paper_trades_id_seq" in sequences and "bets_id_seq" not in sequences
        assert "traderesult" in enums and "betresult" not in enums

    async def test_upgrade_leaves_no_paper_trade_name_behind(
        self, paper_trades_session: AsyncSession
    ) -> None:
        session = paper_trades_session
        connection = await session.connection()

        await connection.run_sync(lambda c: run_bet_rename(c, "upgrade"))

        leftovers = await _names(
            session,
            "SELECT name FROM ("
            " SELECT relname AS name FROM pg_class"
            " UNION SELECT conname FROM pg_constraint"
            " UNION SELECT typname FROM pg_type) AS names"
            " WHERE name ILIKE '%paper\\_trade%' OR name = 'traderesult'",
        )
        assert leftovers == set()
        assert (await session.execute(text("SELECT count(*) FROM bets"))).scalar_one() == 0
