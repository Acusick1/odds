"""Tests for the CLI database-override helper."""

from __future__ import annotations

import pytest
from odds_cli.db_override import swap_dbname


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


class TestOverrideDatabaseUrl:
    """Tests that the env override reads current settings and clears the cache."""

    def test_mutates_env_and_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        from odds_cli.db_override import override_database_url
        from odds_core.config import get_settings, reset_settings_cache

        reset_settings_cache()
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/odds"
        )
        # Prime the cache with the initial URL
        assert get_settings().database.url.endswith("/odds")

        new_url = override_database_url("odds_test")

        assert new_url.endswith("/odds_test")
        assert os.environ["DATABASE_URL"].endswith("/odds_test")
        # Cache should have been cleared so the next get_settings() reflects the override.
        assert get_settings().database.url.endswith("/odds_test")
