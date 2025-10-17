"""Setup utilities for development."""

import asyncio

import structlog

from core.database import close_db, init_db

logger = structlog.get_logger()


async def initialize_database():
    """Initialize database schema (for development only)."""
    logger.info("initializing_database")

    try:
        await init_db()
        logger.info("database_initialized")
        print("✓ Database initialized successfully")

    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        print(f"✗ Database initialization failed: {str(e)}")
        raise

    finally:
        await close_db()


if __name__ == "__main__":
    print("Betting Odds Pipeline - Database Setup\n")
    asyncio.run(initialize_database())
