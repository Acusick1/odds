"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """The Odds API configuration."""

    model_config = SettingsConfigDict(env_prefix="ODDS_API_", extra="ignore")

    key: str = Field(..., description="The Odds API key")
    base_url: str = Field(
        default="https://api.the-odds-api.com/v4", description="Base URL for The Odds API"
    )
    quota: int = Field(default=20_000, description="Monthly API request quota")


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", extra="ignore")

    url: str = Field(..., description="PostgreSQL connection URL")
    pool_size: int = Field(default=5, description="Database connection pool size")


class DataCollectionConfig(BaseSettings):
    """Data collection parameters."""

    model_config = SettingsConfigDict(extra="ignore")

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


class SchedulerConfig(BaseSettings):
    """Scheduler backend configuration."""

    model_config = SettingsConfigDict(env_prefix="SCHEDULER_", extra="ignore")

    backend: str = Field(
        default="local",
        description="Scheduler backend: 'aws', 'railway', or 'local'",
    )
    dry_run: bool = Field(
        default=False,
        description="Enable dry-run mode for scheduler (log operations without executing)",
    )
    lookahead_days: int = Field(
        default=7,
        description="How many days ahead to check for games when scheduling",
    )


class AWSConfig(BaseSettings):
    """AWS-specific configuration (only needed when scheduler_backend='aws')."""

    model_config = SettingsConfigDict(env_prefix="AWS_", extra="ignore")

    region: str | None = Field(default=None, description="AWS region")
    lambda_arn: str | None = Field(
        default=None,
        description="Lambda function ARN (for self-scheduling)",
    )


class DataQualityConfig(BaseSettings):
    """Data quality and validation settings."""

    model_config = SettingsConfigDict(extra="ignore")

    enable_validation: bool = Field(default=True, description="Enable data quality validation")
    reject_invalid_odds: bool = Field(
        default=False, description="Reject invalid odds (vs log only)"
    )


class AlertConfig(BaseSettings):
    """Alert configuration (infrastructure for future use)."""

    model_config = SettingsConfigDict(extra="ignore")

    discord_webhook_url: str | None = Field(default=None, description="Discord webhook URL")
    alert_enabled: bool = Field(default=False, description="Enable alerts")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: str = Field(default="INFO", description="Logging level")
    file: str = Field(default="logs/odds_pipeline.log", description="Log file path")


class Settings(BaseSettings):
    """
    Composed application settings loaded from environment variables.

    Configuration is organized into logical sections for better maintainability.
    Each section can be independently configured and tested.

    Example usage:
        settings = get_settings()
        api_key = settings.api.key
        db_url = settings.database.url
        bookmakers = settings.data_collection.bookmakers
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Composed configuration sections
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings; primarily for testing overrides."""
    get_settings.cache_clear()
