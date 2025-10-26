"""Configuration management using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Configuration
    odds_api_key: str = Field(..., description="The Odds API key")
    odds_api_base_url: str = Field(
        default="https://api.the-odds-api.com/v4", description="Base URL for The Odds API"
    )
    odds_api_quota: int = Field(default=20_000, description="Monthly API request quota")

    # Database
    database_url: str = Field(..., description="PostgreSQL connection URL")
    database_pool_size: int = Field(default=5, description="Database connection pool size")

    # Data Collection
    sports: list[str] = Field(default=["basketball_nba"], description="Sports to track")
    bookmakers: list[str] = Field(
        default=[
            "pinnacle",
            "circasports",
            "draftkings",
            "fanduel",
            "betmgm",
            "williamhill_us",  # Caesars
            "betrivers",
            "bovada",
        ],
        description="Bookmakers to track",
    )
    markets: list[str] = Field(
        default=["h2h", "spreads", "totals"], description="Markets to collect"
    )
    regions: list[str] = Field(default=["us"], description="Regions for odds data")

    # Scheduler Configuration
    scheduler_backend: str = Field(
        default="local",
        description="Scheduler backend: 'aws', 'railway', or 'local'",
    )
    scheduler_dry_run: bool = Field(
        default=False,
        description="Enable dry-run mode for scheduler (log operations without executing)",
    )
    scheduling_lookahead_days: int = Field(
        default=7,
        description="How many days ahead to check for games when scheduling",
    )

    # AWS Configuration (only needed when scheduler_backend='aws')
    aws_region: str | None = Field(default=None, description="AWS region")
    lambda_arn: str | None = Field(
        default=None, description="Lambda function ARN (for self-scheduling)"
    )

    # Data Quality
    enable_validation: bool = Field(default=True, description="Enable data quality validation")
    reject_invalid_odds: bool = Field(
        default=False, description="Reject invalid odds (vs log only)"
    )

    # Alerts (infrastructure for future use)
    discord_webhook_url: str | None = Field(default=None, description="Discord webhook URL")
    alert_enabled: bool = Field(default=False, description="Enable alerts")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/odds_pipeline.log", description="Log file path")


@lru_cache
def get_settings() -> "Settings":
    """Return cached application settings instance."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings; primarily for testing overrides."""
    get_settings.cache_clear()
