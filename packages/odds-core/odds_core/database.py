"""Database connection and session management."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from odds_core.config import get_settings

# Create async engine
# AWS Lambda requires NullPool to avoid "attached to a different loop" errors because:
# 1. Lambda may create an event loop during cold start/initialization
# 2. asyncio.run() in the handler creates a NEW event loop per invocation
# 3. asyncpg connections attached to the old loop can't be used in the new loop
# NullPool disables connection pooling, creating a fresh connection per checkout.
_settings = get_settings()
_is_lambda = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if _is_lambda:
    # Lambda: Use NullPool to prevent connection reuse across event loops
    engine = create_async_engine(
        _settings.database.url,
        echo=False,
        future=True,
        poolclass=NullPool,
    )
else:
    # Local/Railway: Use connection pooling for better performance
    engine = create_async_engine(
        _settings.database.url,
        echo=False,
        future=True,
        pool_size=_settings.database.pool_size,
        max_overflow=10,
    )

# Create async session factory
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session.

    Yields:
        AsyncSession: Database session

    Example:
        async with get_session() as session:
            result = await session.execute(select(Event))
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database schema.

    Creates all tables defined in SQLModel.
    Note: In production, use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
