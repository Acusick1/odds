"""Unit tests for configuration management."""

import os
from unittest.mock import patch

# Set up minimal environment before importing Settings
if "ODDS_API_KEY" not in os.environ:
    os.environ["ODDS_API_KEY"] = "test_key"
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "test_url"

from core.config import Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_defaults(self):
        """Test default configuration values."""
        # Create fresh settings with minimal env (not loading .env file)
        with patch.dict(
            os.environ, {"ODDS_API_KEY": "test_key", "DATABASE_URL": "test_url"}, clear=True
        ):
            settings = Settings(_env_file=None)  # Don't load .env file

            assert settings.odds_api_base_url == "https://api.the-odds-api.com/v4"
            assert settings.odds_api_quota == 20_000
            assert settings.database_pool_size == 5
            assert settings.sports == ["basketball_nba"]
            assert len(settings.bookmakers) == 8
            assert settings.markets == ["h2h", "spreads", "totals"]
            assert settings.regions == ["us"]
            assert settings.sampling_mode == "adaptive"
            assert settings.fixed_interval_minutes == 30
            assert settings.enable_validation is True
            assert settings.reject_invalid_odds is False
            assert settings.alert_enabled is False
            assert settings.log_level == "INFO"

    def test_settings_custom_values(self):
        """Test custom configuration values."""
        with patch.dict(
            os.environ,
            {
                "ODDS_API_KEY": "custom_key",
                "DATABASE_URL": "postgresql://custom",
                "SAMPLING_MODE": "adaptive",
                "FIXED_INTERVAL_MINUTES": "15",
                "ENABLE_VALIDATION": "false",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            settings = Settings()

            assert settings.odds_api_key == "custom_key"
            assert settings.database_url == "postgresql://custom"
            assert settings.sampling_mode == "adaptive"
            assert settings.fixed_interval_minutes == 15
            assert settings.enable_validation is False
            assert settings.log_level == "DEBUG"

    def test_settings_bookmakers(self):
        """Test bookmaker configuration."""
        with patch.dict(os.environ, {"ODDS_API_KEY": "test_key", "DATABASE_URL": "test_url"}):
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

            assert settings.bookmakers == expected_bookmakers

    def test_settings_adaptive_intervals(self):
        """Test adaptive sampling interval configuration."""
        with patch.dict(os.environ, {"ODDS_API_KEY": "test_key", "DATABASE_URL": "test_url"}):
            settings = Settings()

            assert settings.adaptive_intervals == {
                "opening": 72.0,
                "early": 24.0,
                "sharp": 12.0,
                "pregame": 3.0,
                "closing": 0.5,
            }
