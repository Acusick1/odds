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
            assert settings.api.quota == 20_000
            assert settings.database.pool_size == 5
            assert settings.data_collection.sports == ["basketball_nba"]
            assert len(settings.data_collection.bookmakers) == 8
            assert settings.data_collection.markets == ["h2h", "spreads", "totals"]
            assert settings.data_collection.regions == ["us"]
            assert settings.scheduler.backend == "local"
            assert settings.scheduler.dry_run is False
            assert settings.scheduler.lookahead_days == 7
            assert settings.data_quality.enable_validation is True
            assert settings.data_quality.reject_invalid_odds is False
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
