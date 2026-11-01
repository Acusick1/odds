"""Tests for the CLI database-override helper."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from odds_cli.db_override import override_database_url, swap_dbname
from odds_core.config import get_settings, reset_settings_cache
from sqlalchemy.engine import make_url

BASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/odds"


class TestSwapDbname:
    """Tests for the dbname-component swap used by --db flags."""

    def test_swaps_local_url(self) -> None:
        original = "postgresql+asyncpg://postgres:postgres@localhost:5433/odds"
        assert (
            swap_dbname(original, "odds_test")
            == "postgresql+asyncpg://postgres:postgres@localhost:5433/odds_test"
        )

    def test_preserves_query_string(self) -> None:
        original = "postgresql+asyncpg://u:p@ep-foo.neon.tech/neondb?sslmode=require"
        assert (
            swap_dbname(original, "odds_test")
            == "postgresql+asyncpg://u:p@ep-foo.neon.tech/odds_test?sslmode=require"
        )

    def test_preserves_custom_port(self) -> None:
        original = "postgresql://user:pw@db.internal:6543/production"
        assert swap_dbname(original, "sandbox") == "postgresql://user:pw@db.internal:6543/sandbox"

    def test_handles_url_without_port(self) -> None:
        original = "postgresql+asyncpg://user:pw@host/olddb"
        assert swap_dbname(original, "newdb") == "postgresql+asyncpg://user:pw@host/newdb"

    def test_preserves_scheme_including_driver_suffix(self) -> None:
        # The +asyncpg suffix must survive — pydantic settings / SQLAlchemy rely on it.
        original = "postgresql+asyncpg://u:p@h/a"
        assert swap_dbname(original, "b").startswith("postgresql+asyncpg://")

    def test_overwrites_existing_path_segments(self) -> None:
        # Defensive: URLs shouldn't carry extra path segments, but if one does,
        # the swap collapses the path to just /newdb rather than appending.
        original = "postgresql://u:p@h:5432/olddb/extra"
        assert swap_dbname(original, "newdb") == "postgresql://u:p@h:5432/newdb"

    def test_rejects_dbname_with_slash(self) -> None:
        with pytest.raises(ValueError, match="invalid dbname"):
            swap_dbname("postgresql://u:p@h/a", "bad/name")

    def test_rejects_dbname_with_query_marker(self) -> None:
        with pytest.raises(ValueError, match="invalid dbname"):
            swap_dbname("postgresql://u:p@h/a", "bad?x=1")


@pytest.fixture
def base_database_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", BASE_URL)
    monkeypatch.delenv("AGENT_DATABASE_URL", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.mark.usefixtures("base_database_url")
class TestOverrideDatabaseUrl:
    """Tests that the override re-points settings, the engine and the agent DSN."""

    def test_mutates_env_and_clears_cache(self) -> None:
        import os

        assert get_settings().database.url.endswith("/odds")

        new_url = override_database_url("odds_test")

        assert new_url.endswith("/odds_test")
        assert os.environ["DATABASE_URL"].endswith("/odds_test")
        assert get_settings().database.url.endswith("/odds_test")

    async def test_repoints_session_built_before_override(self) -> None:
        from odds_core.database import async_session_maker

        async with async_session_maker() as session:
            assert make_url(str(session.bind.url)).database == "odds"

        override_database_url("odds_test")

        async with async_session_maker() as session:
            assert make_url(str(session.bind.url)).database == "odds_test"

    def test_agent_dsn_keeps_role_and_takes_override_dbname(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from odds_lambda.jobs.agent_run import _build_agent_subprocess_env

        monkeypatch.setenv(
            "AGENT_DATABASE_URL", "postgresql+asyncpg://odds_agent:pw@localhost:5433/odds"
        )
        reset_settings_cache()

        override_database_url("odds_test")

        agent_url = make_url(_build_agent_subprocess_env()["DATABASE_URL"])
        assert agent_url.username == "odds_agent"
        assert agent_url.database == "odds_test"
