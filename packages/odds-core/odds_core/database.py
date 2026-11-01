"""Database connection and session management."""

import os
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from odds_core.config import DatabaseConfig, get_settings


@dataclass(frozen=True)
class _Binding:
    url: str
    pool_size: int
    engine: AsyncEngine
    session_maker: async_sessionmaker[AsyncSession]


_binding: _Binding | None = None


def _create_engine(config: DatabaseConfig) -> AsyncEngine:
    # Lambda runs each invocation in a new event loop (asyncio.run), and asyncpg
    # connections pooled on an earlier loop cannot be reused on it.
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return create_async_engine(config.url, echo=False, future=True, poolclass=NullPool)
    return create_async_engine(
        config.url,
        echo=False,
        future=True,
        pool_size=config.pool_size,
        max_overflow=10,
    )


def _current_binding() -> _Binding:
    """Return the engine for the current settings, building it on first use or after they change."""
    global _binding
    config = get_settings().database
    if _binding is None or (_binding.url, _binding.pool_size) != (config.url, config.pool_size):
        engine = _create_engine(config)
        _binding = _Binding(
            url=config.url,
            pool_size=config.pool_size,
            engine=engine,
            session_maker=async_sessionmaker(engine, expire_on_commit=False),
        )
    return _binding


def get_engine() -> AsyncEngine:
    """Return the engine for the currently configured ``DATABASE_URL``."""
    return _current_binding().engine


def async_session_maker() -> AsyncSession:
    """Open a session on the currently configured ``DATABASE_URL``."""
    return _current_binding().session_maker()


async def init_db() -> None:
    """
    Initialize database schema.

    Creates all tables defined in SQLModel.
    Note: In production, use Alembic migrations instead.
    """
    async with get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await get_engine().dispose()
