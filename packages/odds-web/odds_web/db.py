"""The web app's own database engine, bound to ``DatabaseConfig.web_url`` and nothing else."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from odds_core.config import get_settings
from sqlalchemy import Connection, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

POOL_SIZE = 2
STATEMENT_TIMEOUT_MS = 20_000


class WebDatabaseNotConfigured(RuntimeError):
    """``WEB_DATABASE_URL`` is unset, and the write-capable ``DATABASE_URL`` is not a fallback."""


@dataclass(frozen=True)
class WebDatabase:
    engine: AsyncEngine
    session_maker: async_sessionmaker[AsyncSession]


def restrict_transactions(
    engine: AsyncEngine, statement_timeout_ms: int
) -> Callable[[Connection], None]:
    """Make every transaction on ``engine`` read-only and cancel statements past the timeout.

    Set per transaction, so nothing a session runs can lift it. The ``odds_web`` role's grants are
    what stop a write; this keeps a DSN that is not that role from writing through this app.
    """

    @event.listens_for(engine.sync_engine, "begin")
    def restrict(connection: Connection) -> None:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        connection.exec_driver_sql(f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}")

    return restrict


def build_web_database(url: str, statement_timeout_ms: int = STATEMENT_TIMEOUT_MS) -> WebDatabase:
    engine = create_async_engine(url, pool_size=POOL_SIZE, max_overflow=0, pool_pre_ping=True)
    restrict_transactions(engine, statement_timeout_ms)
    return WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))


def web_database_from_settings() -> WebDatabase:
    url = get_settings().database.web_url
    if not url:
        raise WebDatabaseNotConfigured(
            "WEB_DATABASE_URL is not set; odds-web never falls back to DATABASE_URL"
        )
    return build_web_database(url)
