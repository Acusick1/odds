"""The web app's own database engine, bound to ``DatabaseConfig.web_url`` and nothing else."""

from __future__ import annotations

from dataclasses import dataclass

from odds_core.config import get_settings
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


def build_web_database(url: str, statement_timeout_ms: int = STATEMENT_TIMEOUT_MS) -> WebDatabase:
    """An engine whose sessions are read-only and cancel any statement running past the timeout.

    The ``odds_web`` role's grants are what stop a write; the read-only default keeps a DSN that
    is not that role from writing through this app.
    """
    engine = create_async_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "statement_timeout": str(statement_timeout_ms),
                "default_transaction_read_only": "on",
            }
        },
    )
    return WebDatabase(engine, async_sessionmaker(engine, expire_on_commit=False))


def web_database_from_settings() -> WebDatabase:
    url = get_settings().database.web_url
    if not url:
        raise WebDatabaseNotConfigured(
            "WEB_DATABASE_URL is not set; odds-web never falls back to DATABASE_URL"
        )
    return build_web_database(url)
