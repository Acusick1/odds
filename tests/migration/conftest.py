import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from typer.testing import CliRunner


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    Recreates the database engine with NullPool to fix "Task got Future attached to
    a different loop" errors that occur when:
    1. The module-level engine in core.database is created with the default event loop
    2. Typer commands use asyncio.run() which creates a NEW event loop
    3. pytest-asyncio tests may run in yet another event loop

    Using NullPool disables connection pooling, which prevents connections from being
    reused across different event loops. This is the recommended approach for testing
    async SQLAlchemy applications.

    This hook runs early enough to replace the engine before any tests import and use it.

    See: https://github.com/sqlalchemy/sqlalchemy/issues/5626
    """
    import odds_core.database
    from odds_core.config import get_settings

    settings = get_settings()

    # Create new engine with NullPool
    test_engine = create_async_engine(
        settings.database.url,
        echo=False,
        future=True,
        poolclass=NullPool,  # Critical: prevents connection reuse across event loops
    )

    # Create new session maker
    test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

    # Replace module-level objects
    odds_core.database.engine = test_engine
    odds_core.database.async_session_maker = test_session_maker


@pytest.fixture(scope="function")
def runner():
    yield CliRunner()
