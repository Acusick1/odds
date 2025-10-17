"""Configuration management using Pydantic Settings."""

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

    # Sampling Configuration
    sampling_mode: str = Field(
        default="adaptive", description="Sampling mode: 'fixed' or 'adaptive'"
    )
    fixed_interval_minutes: int = Field(
        default=30, description="Fixed sampling interval in minutes"
    )
    adaptive_intervals: dict[str, float] = Field(
        default={
            "opening": 72.0,  # 3 days before game
            "early": 24.0,  # 24 hours before
            "sharp": 12.0,  # 12 hours before
            "pregame": 3.0,  # 3 hours before
            "closing": 0.5,  # 30 minutes before
        },
        description="Adaptive sampling intervals in hours before game",
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


# Global settings instance
settings = Settings()
