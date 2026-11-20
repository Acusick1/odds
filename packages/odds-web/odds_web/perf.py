"""The statements that cost the database the most time, from ``pg_stat_statements``.

The ``odds_web`` role reads them only through the functions ``odds_web.grants`` defines, which
run as their owner and return every role's statements in the current database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

TOP = 20


@dataclass(frozen=True)
class QueryStat:
    """One normalised statement as one role ran it, with its times in milliseconds."""

    query: str
    role: str
    calls: int
    total_ms: float
    mean_ms: float
    rows: int


@dataclass(frozen=True)
class QueryStats:
    by_total: list[QueryStat]
    by_mean: list[QueryStat]
    reset_at: datetime | None


def _top(order: str) -> str:
    return (
        "SELECT query, role, calls, total_exec_time, mean_exec_time, rows "
        f"FROM odds_web_query_stats() ORDER BY {order} DESC, queryid LIMIT :limit"
    )


async def load_query_stats(session: AsyncSession, limit: int = TOP) -> QueryStats:
    """The ``limit`` statements with the most total and the highest mean execution time, and when
    the statistics were last reset."""

    async def top(order: str) -> list[QueryStat]:
        rows = await session.execute(text(_top(order)), {"limit": limit})
        return [QueryStat(*row) for row in rows.tuples()]

    reset_at = await session.scalar(text("SELECT odds_web_query_stats_reset()"))
    return QueryStats(await top("total_exec_time"), await top("mean_exec_time"), reset_at)
