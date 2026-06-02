"""Unit tests for configuration management."""

import os
from unittest.mock import patch

# Set up minimal environment before importing Settings
if "ODDS_API_KEY" not in os.environ:
    os.environ["ODDS_API_KEY"] = "test_key"
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from odds_core.config import Settings

_REQUIRED_ENV = {
    "ODDS_API_KEY": "test_key",
    "DATABASE_URL": "postgresql://test:test@localhost/test",
}


class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_defaults(self):
        """Default values apply when env vars are absent; _env_file=None is respected."""
        with patch.dict(os.environ, _REQUIRED_ENV, clear=True):
            settings = Settings(_env_file=None)

            assert settings.api.base_url == "https://api.the-odds-api.com/v4"
            assert settings.api.quota == 500
            assert settings.api.keys is None
            assert settings.database.pool_size == 5
            assert settings.data_collection.sports == ["soccer_epl", "baseball_mlb"]
            assert len(settings.data_collection.bookmakers) == 8
            assert settings.data_collection.markets == ["h2h", "spreads", "totals"]
            assert settings.data_collection.regions == ["us"]
            assert settings.scheduler.backend == "local"
            assert settings.scheduler.dry_run is False
            assert settings.scheduler.lookahead_days == 7
            assert settings.scheduler.bootstrap_jobs == [
                # "agent-run",
                "fetch-oddsportal",
                "fetch-oddsportal-results",
                # "daily-digest",
                "fetch-espn-fixtures",
                "fetch-betfair-exchange",
                "fetch-mlb-probables",
            ]
            assert settings.data_quality.enable_validation is True
            assert settings.data_quality.reject_invalid_odds is False
            assert settings.alerts.alert_enabled is False
            assert settings.logging.level == "INFO"
            assert settings.model.name is None
            assert settings.model.bucket is None

    def test_settings_defaults_with_discord_webhook_in_env(self):
        """DISCORD_WEBHOOK_URL in env does not flip alert_enabled when _env_file=None."""
        with patch.dict(
            os.environ,
            {**_REQUIRED_ENV, "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/x"},
            clear=True,
        ):
            settings = Settings(_env_file=None)
            assert settings.alerts.discord_webhook_url == "https://discord.com/api/webhooks/x"
            assert settings.alerts.alert_enabled is False

    def test_settings_custom_values(self):
        """Custom configuration values from environment variables."""
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
        """Bookmaker configuration defaults."""
        with patch.dict(os.environ, _REQUIRED_ENV):
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

            assert settings.data_collection.bookmakers == expected_bookmakers

    def test_settings_aws_configuration(self):
        """AWS-specific configuration fields."""
        with patch.dict(
            os.environ,
            {
                **_REQUIRED_ENV,
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
        """Polymarket configuration defaults."""
        with patch.dict(os.environ, _REQUIRED_ENV, clear=True):
            settings = Settings(_env_file=None)

            assert settings.polymarket.gamma_base_url == "https://gamma-api.polymarket.com"
            assert settings.polymarket.clob_base_url == "https://clob.polymarket.com"
            assert settings.polymarket.nba_series_id == "10345"
            assert settings.polymarket.game_tag_id == "100639"
            assert settings.polymarket.enabled is True
            assert settings.polymarket.price_poll_interval == 300
            assert settings.polymarket.orderbook_poll_interval == 1800
            assert settings.polymarket.collect_moneyline is True
            assert settings.polymarket.collect_spreads is True
            assert settings.polymarket.collect_totals is True
            assert settings.polymarket.collect_player_props is False
            assert settings.polymarket.orderbook_tiers == ["closing", "pregame"]

    def test_polymarket_env_overrides(self):
        """Polymarket configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                **_REQUIRED_ENV,
                "POLYMARKET_ENABLED": "false",
                "POLYMARKET_PRICE_POLL_INTERVAL": "60",
                "POLYMARKET_ORDERBOOK_POLL_INTERVAL": "900",
                "POLYMARKET_COLLECT_PLAYER_PROPS": "true",
                "POLYMARKET_NBA_SERIES_ID": "99999",
            },
        ):
            settings = Settings()

            assert settings.polymarket.enabled is False
            assert settings.polymarket.price_poll_interval == 60
            assert settings.polymarket.orderbook_poll_interval == 900
            assert settings.polymarket.collect_player_props is True
            assert settings.polymarket.nba_series_id == "99999"

    def test_betfair_defaults(self):
        """BetfairConfig defaults when no env vars are set."""
        from odds_core.config import BetfairConfig

        cfg = BetfairConfig()
        assert cfg.enabled is False
        assert cfg.username is None
        assert cfg.password is None
        assert cfg.app_key is None
        assert cfg.cert_file is None
        assert cfg.cert_key is None
        assert cfg.sports is None
        assert cfg.lookahead_hours == 168

    def test_betfair_env_overrides(self):
        """BETFAIR_* env vars populate settings.betfair via Settings."""
        with patch.dict(
            os.environ,
            {
                **_REQUIRED_ENV,
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
            settings = Settings(_env_file=None)
            cfg = settings.betfair
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

        with pytest.raises(ValidationError) as exc_info:
            BetfairConfig(enabled=True, app_key="appkey123")
        msg = str(exc_info.value)
        assert "BETFAIR_USERNAME" in msg
        assert "BETFAIR_PASSWORD" in msg

    def test_betfair_disabled_skips_credential_check(self):
        """enabled=False with no creds is valid (default state)."""
        from odds_core.config import BetfairConfig

        cfg = BetfairConfig(enabled=False)
        assert cfg.enabled is False

    def test_env_var_mapping(self):
        """Key env vars map to the correct nested fields on Settings."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "myapikey",
                "ODDS_API_KEYS": "key1,key2",
                "ODDS_API_QUOTA": "1000",
                "DATABASE_URL": "postgresql://db",
                "DATABASE_POOL_SIZE": "10",
                "SCHEDULER_BACKEND": "railway",
                "SCHEDULER_DRY_RUN": "true",
                "SCHEDULER_LOOKAHEAD_DAYS": "3",
                "AWS_REGION": "eu-west-1",
                "AWS_LAMBDA_ARN": "arn:aws:lambda:eu-west-1:999:function:f",
                "AWS_RULE_PREFIX": "odds",
                "MODEL_NAME": "epl-clv-home",
                "MODEL_BUCKET": "my-bucket",
                "ENABLE_VALIDATION": "false",
                "REJECT_INVALID_ODDS": "true",
                "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test",
                "ALERT_ENABLED": "true",
                "LOG_LEVEL": "DEBUG",
                "LOG_FILE": "/var/log/odds.log",
                "POLYMARKET_ENABLED": "false",
                "BETFAIR_ENABLED": "false",
            },
            clear=True,
        ):
            s = Settings(_env_file=None)

            assert s.api.key == "myapikey"
            assert s.api.keys == "key1,key2"
            assert s.api.quota == 1000
            assert s.database.url == "postgresql://db"
            assert s.database.pool_size == 10
            assert s.scheduler.backend == "railway"
            assert s.scheduler.dry_run is True
            assert s.scheduler.lookahead_days == 3
            assert s.aws.region == "eu-west-1"
            assert s.aws.lambda_arn == "arn:aws:lambda:eu-west-1:999:function:f"
            assert s.aws.rule_prefix == "odds"
            assert s.model.name == "epl-clv-home"
            assert s.model.bucket == "my-bucket"
            assert s.data_quality.enable_validation is False
            assert s.data_quality.reject_invalid_odds is True
            assert s.alerts.discord_webhook_url == "https://discord.com/api/webhooks/test"
            assert s.alerts.alert_enabled is True
            assert s.logging.level == "DEBUG"
            assert s.logging.file == "/var/log/odds.log"
            assert s.polymarket.enabled is False
            assert s.betfair.enabled is False

    def test_env_file_suppressed_by_env_file_none(self):
        """Settings(_env_file=None) does not load any sub-config from .env."""
        # Even if DISCORD_WEBHOOK_URL or ALERT_ENABLED appear in .env, they must not
        # affect a Settings instance constructed with _env_file=None.
        with patch.dict(os.environ, _REQUIRED_ENV, clear=True):
            settings = Settings(_env_file=None)
            # Defaults apply — no bleed from any .env file on the developer's machine
            assert settings.alerts.alert_enabled is False
            assert settings.alerts.discord_webhook_url is None
            assert settings.scheduler.backend == "local"
