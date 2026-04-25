"""Unit tests for configuration management."""

import os
from unittest.mock import patch

# Set up minimal environment before importing Settings
if "ODDS_API_KEY" not in os.environ:
    os.environ["ODDS_API_KEY"] = "test_key"
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"

from odds_core.config import Settings


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
                "agent-run",
                "fetch-oddsportal",
                "fetch-oddsportal-results",
                "daily-digest",
                "fetch-espn-fixtures",
                "fetch-betfair-exchange",
            ]
            assert settings.data_quality.enable_validation is True
            assert settings.data_quality.reject_invalid_odds is False
            assert settings.alerts.alert_enabled is False
            assert settings.logging.level == "INFO"
            assert settings.model.name is None
            assert settings.model.bucket is None

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

            assert settings.data_collection.bookmakers == expected_bookmakers

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
        """Test Polymarket configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "test_key",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
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
        """Defaults: disabled, no creds, both EPL+MLB in scope."""
        from odds_core.config import BetfairConfig

        with patch.dict(os.environ, {}, clear=True):
            cfg = BetfairConfig(_env_file=None)
            assert cfg.enabled is False
            assert cfg.username is None
            assert cfg.password is None
            assert cfg.app_key is None
            assert cfg.cert_file is None
            assert cfg.cert_key is None
            assert cfg.sports == ["soccer_epl", "baseball_mlb"]
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
