"""Unit tests for configuration management."""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

# Set up minimal environment before importing Settings
if "ODDS_API_KEY" not in os.environ:
    os.environ["ODDS_API_KEY"] = "test_key"
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from odds_core.config import APIConfig, DatabaseConfig, Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_defaults(self):
        """Test default configuration values."""
        # Create fresh settings with minimal env (not loading .env file)
        with patch.dict(
            os.environ,
            {"ODDS_API_KEY": "test_key", "DATABASE_URL": "postgresql://test:test@localhost/test"},
            clear=True,
        ):
            settings = Settings(_env_file=None)  # Don't load .env file

            assert settings.api.base_url == "https://api.the-odds-api.com/v4"
            assert settings.api.keys is None
            assert settings.database.pool_size == 5
            assert settings.data_collection.sports == [
                "soccer_epl",
                "baseball_mlb",
                "basketball_nba",
            ]
            assert settings.data_collection.bookmakers_for("soccer_epl") == [
                "pinnacle",
                "betfair_ex_uk",
                "williamhill",
                "paddypower",
                "skybet",
                "ladbrokes_uk",
                "coral",
                "betfred_uk",
                "betvictor",
                "sport888",
            ]
            assert settings.data_collection.markets == ["h2h", "spreads", "totals"]
            assert settings.data_collection.regions == ["us"]
            assert settings.scheduler.backend == "local"
            assert settings.scheduler.dry_run is False
            assert settings.scheduler.lookahead_days == 7
            assert settings.scheduler.bootstrap_jobs == [
                # "agent-run",
                "fetch-odds",
                "fetch-scores",
                "update-status",
                "check-health",
                "fetch-oddsportal",
                "fetch-oddsportal-results",
                "fetch-mlb-results",
                "fetch-espn-fixtures",
                "fetch-betfair-exchange",
                "fetch-polymarket-odds",
                "fetch-mlb-probables",
                "fetch-injuries",
                "run-candidates",
            ]
            assert settings.data_quality.enable_validation is True
            assert settings.alerts.alert_enabled is False
            assert settings.logging.level == "INFO"

    def test_settings_custom_values(self):
        """Test custom configuration values."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "custom_key",
                "DATABASE_URL": "postgresql://custom",
                "SCHEDULER_BACKEND": "aws",
                "SCHEDULER_DRY_RUN": "true",
                "SCHEDULER_LOOKAHEAD_DAYS": "14",
                "ENABLE_VALIDATION": "false",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            settings = Settings()

            assert settings.api.key == "custom_key"
            assert settings.database.url == "postgresql://custom"
            assert settings.scheduler.backend == "aws"
            assert settings.scheduler.dry_run is True
            assert settings.scheduler.lookahead_days == 14
            assert settings.data_quality.enable_validation is False
            assert settings.logging.level == "DEBUG"

    def test_settings_bookmakers(self):
        """Test bookmaker configuration."""
        with patch.dict(
            os.environ,
            {"ODDS_API_KEY": "test_key", "DATABASE_URL": "postgresql://test:test@localhost/test"},
        ):
            settings = Settings()

            expected_bookmakers = [
                "pinnacle",
                "circasports",
                "draftkings",
                "fanduel",
                "betmgm",
                "williamhill_us",
                "betrivers",
                "bovada",
            ]

            assert settings.data_collection.bookmakers_for("baseball_mlb") == expected_bookmakers
            assert settings.data_collection.bookmakers_for("basketball_nba") == [
                "pinnacle",
                "williamhill",
                "ladbrokes_uk",
                "coral",
                "sport888",
                "betvictor",
                "paddypower",
                "skybet",
                "unibet_uk",
                "betway",
            ]

    def test_settings_bookmakers_fit_one_region(self):
        """Each sport names at most 10 bookmakers, which the Odds API bills as one region."""
        with patch.dict(
            os.environ,
            {"ODDS_API_KEY": "test_key", "DATABASE_URL": "postgresql://test:test@localhost/test"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

            for bookmakers in settings.data_collection.bookmakers.values():
                assert len(bookmakers) <= 10

    def test_settings_bookmakers_unknown_sport_raises(self):
        """A sport with no configured bookmakers fails instead of requesting none."""
        with patch.dict(
            os.environ,
            {"ODDS_API_KEY": "test_key", "DATABASE_URL": "postgresql://test:test@localhost/test"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

            with pytest.raises(ValueError, match="icehockey_nhl"):
                settings.data_collection.bookmakers_for("icehockey_nhl")

    def test_settings_aws_configuration(self):
        """Test AWS-specific configuration fields."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "test_key",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
                "SCHEDULER_BACKEND": "aws",
                "AWS_REGION": "us-east-1",
                "AWS_LAMBDA_ARN": "arn:aws:lambda:us-east-1:123456789:function:fetch-odds",
            },
        ):
            settings = Settings()

            assert settings.scheduler.backend == "aws"
            assert settings.aws.region == "us-east-1"
            assert (
                settings.aws.lambda_arn == "arn:aws:lambda:us-east-1:123456789:function:fetch-odds"
            )

    def test_polymarket_defaults(self):
        """Test Polymarket configuration defaults."""
        with patch.dict(
            os.environ,
            {"ODDS_API_KEY": "test_key", "DATABASE_URL": "postgresql://test:test@localhost/test"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

            assert settings.polymarket.gamma_base_url == "https://gamma-api.polymarket.com"
            assert settings.polymarket.clob_base_url == "https://clob.polymarket.com"
            assert settings.polymarket.enabled is True

    def test_polymarket_env_overrides(self):
        """Test Polymarket configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "test_key",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
                "POLYMARKET_ENABLED": "false",
            },
        ):
            settings = Settings()

            assert settings.polymarket.enabled is False

    def test_betfair_defaults(self):
        """Defaults: disabled, no creds."""
        from odds_core.config import BetfairConfig

        with patch.dict(os.environ, {}, clear=True):
            cfg = BetfairConfig(_env_file=None)
            assert cfg.enabled is False
            assert cfg.username is None
            assert cfg.password is None
            assert cfg.app_key is None
            assert cfg.cert_file is None
            assert cfg.cert_key is None
            assert cfg.lookahead_hours == 168

    def test_betfair_env_overrides(self):
        """Env-driven overrides for credentials and toggles."""
        from odds_core.config import BetfairConfig

        with patch.dict(
            os.environ,
            {
                "BETFAIR_ENABLED": "true",
                "BETFAIR_USERNAME": "alice",
                "BETFAIR_PASSWORD": "secret",
                "BETFAIR_APP_KEY": "appkey123",
                "BETFAIR_CERT_FILE": "/tmp/bf.crt",
                "BETFAIR_CERT_KEY": "/tmp/bf.key",
                "BETFAIR_LOOKAHEAD_HOURS": "72",
            },
            clear=True,
        ):
            cfg = BetfairConfig(_env_file=None)
            assert cfg.enabled is True
            assert cfg.username == "alice"
            assert cfg.password == "secret"
            assert cfg.app_key == "appkey123"
            assert cfg.cert_file == "/tmp/bf.crt"
            assert cfg.cert_key == "/tmp/bf.key"
            assert cfg.lookahead_hours == 72

    def test_betfair_enabled_requires_credentials(self):
        """enabled=True with missing creds must fail at config load, not at runtime."""
        import pytest
        from odds_core.config import BetfairConfig
        from pydantic import ValidationError

        with patch.dict(
            os.environ,
            {"BETFAIR_ENABLED": "true", "BETFAIR_APP_KEY": "appkey123"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                BetfairConfig(_env_file=None)
            msg = str(exc_info.value)
            assert "BETFAIR_USERNAME" in msg
            assert "BETFAIR_PASSWORD" in msg

    def test_betfair_disabled_skips_credential_check(self):
        """enabled=False with no creds is valid (default state)."""
        from odds_core.config import BetfairConfig

        with patch.dict(os.environ, {"BETFAIR_ENABLED": "false"}, clear=True):
            cfg = BetfairConfig(_env_file=None)
            assert cfg.enabled is False


class TestAPIConfigKeys:
    def test_keys_alone_satisfy_config(self):
        with patch.dict(
            os.environ,
            {"ODDS_API_KEYS": "a,b", "DATABASE_URL": "postgresql://test:test@localhost/test"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

        assert settings.api.key is None
        assert settings.api.keys == "a,b"

    def test_each_sport_spends_the_key_at_its_index(self):
        api = APIConfig(keys="epl-key, nba-key", _env_file=None)

        assert api.key_for("soccer_epl") == "epl-key"
        assert api.key_for(None) == "epl-key"
        assert api.key_for("basketball_nba") == "nba-key"

    def test_single_key_serves_the_sports_at_index_zero(self):
        api = APIConfig(key="only", _env_file=None)

        assert api.key_for("soccer_epl") == "only"

    def test_a_sport_beyond_the_configured_keys_raises(self):
        api = APIConfig(key="only", _env_file=None)

        with pytest.raises(ValueError, match="basketball_nba spends Odds API key index 1"):
            api.key_for("basketball_nba")

    def test_key_index_comes_from_the_environment(self):
        with patch.dict(
            os.environ, {"ODDS_API_KEY_INDEX_BY_SPORT": '{"baseball_mlb": 1}'}, clear=True
        ):
            api = APIConfig(keys="a,b", _env_file=None)

        assert api.key_for("baseball_mlb") == "b"
        assert api.key_for("basketball_nba") == "a"


class TestEnvExampleTemplate:
    """`.env.example` copied to `.env` loads, with its commented settings enabled or not."""

    TEMPLATE = Path(__file__).resolve().parents[2] / ".env.example"

    @staticmethod
    def _load(env_text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
        (tmp_path / ".env").write_text(env_text)
        monkeypatch.chdir(tmp_path)
        with patch.dict(os.environ, {}, clear=True):
            return Settings()

    def test_template_loads(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._load(self.TEMPLATE.read_text(), tmp_path, monkeypatch)

        assert (
            settings.database.local_url
            == "postgresql+asyncpg://postgres:postgres@localhost:5433/odds"
        )

    def test_template_loads_with_commented_settings_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enabled = re.sub(r"(?m)^# ([A-Z][A-Z0-9_]*=)", r"\1", self.TEMPLATE.read_text())

        settings = self._load(enabled, tmp_path, monkeypatch)

        assert settings.data_collection.sports == ["soccer_epl", "baseball_mlb", "basketball_nba"]
        assert settings.data_collection.markets == ["h2h", "spreads", "totals"]
        assert settings.data_collection.regions == ["us"]
        assert "fetch-oddsportal" in settings.scheduler.bootstrap_jobs
        assert settings.database.agent_url is not None


class TestDatabaseURLsFromEnvFile:
    def test_auxiliary_urls_are_read_from_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".env").write_text(
            "DATABASE_URL=postgresql://main/db\n"
            "AGENT_DATABASE_URL=postgresql://agent/db\n"
            "PROD_DATABASE_URL=postgresql://prod/db\n"
            "LOCAL_DATABASE_URL=postgresql://local/db\n"
        )
        monkeypatch.chdir(tmp_path)

        with patch.dict(os.environ, {}, clear=True):
            database = Settings().database

        assert database.url == "postgresql://main/db"
        assert database.agent_url == "postgresql://agent/db"
        assert database.prod_url == "postgresql://prod/db"
        assert database.local_url == "postgresql://local/db"

    def test_auxiliary_urls_default_to_none(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://main/db"}, clear=True):
            database = DatabaseConfig(_env_file=None)

        assert database.agent_url is None
        assert database.prod_url is None
        assert database.local_url is None
