from __future__ import annotations

from collections.abc import Iterator

import pytest
from odds_cli.main import app
from odds_core.config import reset_settings_cache
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, object]]]:
    """The arguments of each uvicorn.run call, which does not start a server."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda _app, **kwargs: calls.append(kwargs))
    yield calls
    reset_settings_cache()


class TestWebServe:
    def test_refuses_to_start_without_web_database_url(
        self, monkeypatch: pytest.MonkeyPatch, served: list[dict[str, object]]
    ) -> None:
        monkeypatch.delenv("WEB_DATABASE_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:pw@localhost:5433/odds")
        monkeypatch.chdir("/")
        reset_settings_cache()

        result = runner.invoke(app, ["web", "serve"])

        assert result.exit_code == 1
        assert "WEB_DATABASE_URL" in result.output
        assert served == []

    def test_listens_on_loopback_on_the_port(
        self, monkeypatch: pytest.MonkeyPatch, served: list[dict[str, object]]
    ) -> None:
        monkeypatch.setenv(
            "WEB_DATABASE_URL", "postgresql+asyncpg://odds_web:pw@localhost:5433/odds"
        )
        reset_settings_cache()

        result = runner.invoke(app, ["web", "serve", "--port", "9001"])

        assert result.exit_code == 0, result.output
        assert served == [{"host": "127.0.0.1", "port": 9001}]

    def test_default_port_is_8765(
        self, monkeypatch: pytest.MonkeyPatch, served: list[dict[str, object]]
    ) -> None:
        monkeypatch.setenv(
            "WEB_DATABASE_URL", "postgresql+asyncpg://odds_web:pw@localhost:5433/odds"
        )
        reset_settings_cache()

        runner.invoke(app, ["web", "serve"])

        assert served == [{"host": "127.0.0.1", "port": 8765}]

    def test_has_no_host_option(self) -> None:
        result = runner.invoke(app, ["web", "serve", "--host", "0.0.0.0"])

        assert result.exit_code != 0
