"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from odds_core.sports import SportKey


class APIConfig(BaseSettings):
    """The Odds API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ODDS_API_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    key: str | None = Field(default=None, description="A single Odds API key")
    keys: str | None = Field(
        default=None, description="Comma-separated API keys, in index order (overrides key)"
    )
    key_index_by_sport: dict[str, int] = Field(
        default={"basketball_nba": 1},
        description="Index into the keys of the key each sport spends; absent sports use index 0",
    )
    base_url: str = Field(
        default="https://api.the-odds-api.com/v4", description="Base URL for The Odds API"
    )

    @property
    def key_list(self) -> list[str]:
        """The configured keys in index order: ``keys`` when set, else ``key``."""
        if self.keys:
            return [k.strip() for k in self.keys.split(",") if k.strip()]
        return [self.key] if self.key else []

    def key_index_for(self, sport: str | None) -> int:
        """Index of the key ``sport`` spends: its ``key_index_by_sport`` entry, 0 by default."""
        return self.key_index_by_sport.get(sport, 0) if sport else 0

    def key_for(self, sport: str | None) -> str:
        """The key ``sport`` spends."""
        keys = self.key_list
        if not keys:
            raise ValueError("No Odds API key configured: set ODDS_API_KEYS or ODDS_API_KEY")
        index = self.key_index_for(sport)
        if index >= len(keys):
            raise ValueError(
                f"{sport} spends Odds API key index {index} but {len(keys)} key(s) are "
                "configured: set ODDS_API_KEYS to one key per index"
            )
        return keys[index]


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    url: str = Field(..., description="PostgreSQL connection URL")
    pool_size: int = Field(default=5, description="Database connection pool size")
    agent_url: str | None = Field(
        default=None,
        validation_alias="AGENT_DATABASE_URL",
        description="DSN of the restricted odds_agent role the agent subprocess runs as",
    )
    prod_url: str | None = Field(
        default=None,
        validation_alias="PROD_DATABASE_URL",
        description="Source database of `odds copy from-prod`",
    )
    local_url: str | None = Field(
        default=None,
        validation_alias="LOCAL_DATABASE_URL",
        description="Target database of `odds copy from-prod`",
    )


_US_BOOKMAKERS: tuple[str, ...] = (
    "pinnacle",
    "circasports",
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",  # Caesars
    "betrivers",
    "bovada",
)

_UK_BOOKMAKERS: tuple[str, ...] = (
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
)


# Pinnacle and the UK retail books the Odds API prices for NBA (probed 2026-10-31, region uk).
_NBA_BOOKMAKERS: tuple[str, ...] = (
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
)


class DataCollectionConfig(BaseSettings):
    """Data collection parameters."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sports: list[SportKey] = Field(
        default=["soccer_epl", "baseball_mlb", "basketball_nba"], description="Sports to track"
    )
    bookmakers: dict[SportKey, list[str]] = Field(
        default_factory=lambda: {
            "soccer_epl": list(_UK_BOOKMAKERS),
            "baseball_mlb": list(_US_BOOKMAKERS),
            "basketball_nba": list(_NBA_BOOKMAKERS),
        },
        description="Odds API bookmaker keys requested per sport; at most 10 cost one region",
    )
    markets: list[str] = Field(
        default=["h2h", "spreads", "totals"], description="Markets to collect"
    )
    regions: list[str] = Field(default=["us"], description="Regions for odds data")

    def bookmakers_for(self, sport: str) -> list[str]:
        bookmakers = self.bookmakers.get(sport)
        if bookmakers is None:
            raise ValueError(f"No Odds API bookmakers configured for sport {sport!r}")
        return bookmakers


class SchedulerConfig(BaseSettings):
    """Scheduler backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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
    bootstrap_jobs: list[str] = Field(
        default=[
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
        ],
        description="Jobs to bootstrap when starting the local scheduler",
    )


class AWSConfig(BaseSettings):
    """AWS-specific configuration (only needed when scheduler_backend='aws')."""

    model_config = SettingsConfigDict(
        env_prefix="AWS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    region: str | None = Field(default=None, description="AWS region")
    lambda_arn: str | None = Field(
        default=None,
        description="Lambda function ARN (for self-scheduling)",
    )
    rule_prefix: str | None = Field(
        default=None,
        description="EventBridge rule name prefix (e.g. 'odds') for querying deployed schedules",
    )


class DataQualityConfig(BaseSettings):
    """Data quality and validation settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    enable_validation: bool = Field(default=True, description="Enable data quality validation")


class AlertConfig(BaseSettings):
    """Alert configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    discord_webhook_url: str | None = Field(default=None, description="Discord webhook URL")
    alert_enabled: bool = Field(default=False, description="Enable alerts")
    dead_mans_switch_url: str | None = Field(
        default=None,
        description="Ping URL of an external monitor that alerts when check-health stops pinging",
    )

    # Quota monitoring thresholds
    quota_warning_threshold: float = Field(
        default=0.2,
        description="Trigger warning when API quota falls below this fraction (0-1)",
    )
    quota_critical_threshold: float = Field(
        default=0.1,
        description="Trigger critical alert when API quota falls below this fraction (0-1)",
    )

    # Data staleness monitoring
    stale_data_hours: int = Field(
        default=2,
        description="Minimum active hours without a source's rows before a stale data alert",
    )
    stale_cadence_multiplier: float = Field(
        default=3.0,
        description="Polling intervals of a source's job that may pass without rows before alert",
    )

    stale_last_interval_multiplier: float = Field(
        default=1.5,
        description="Intervals the job chose at its last write that may pass before alert",
    )
    stale_reminder_minutes: int = Field(
        default=360,
        description="Minutes between repeats of an alert while its condition holds (stale source, failing job, health-check warning)",
    )

    betfair_close_lookback_hours: int = Field(
        default=24,
        description="Hours after kick-off an EPL event is still checked for a live Betfair close",
    )

    # Data quality monitoring
    data_quality_error_threshold: int = Field(
        default=10,
        description="Number of error/critical data quality issues in 24h before triggering alert",
    )

    # Job heartbeat monitoring: max hours between completions before alerting.
    # Keys are compound job names matching EventBridge rules (e.g. "fetch-oddsportal-epl").
    heartbeat_expectations: dict[str, float] = Field(
        default={
            # Overnight windows (EPL 22-06 UTC, MLB 05-14 UTC, NBA 07-15 UTC) pause these jobs, so the
            # allowed gap is the window plus the polling interval.
            # The longest gaps between runs: the 48 h opening tier for fetch-odds-epl, the 24 h
            # early, opening and no-game interval for fetch-odds-nba plus the 00:00-10:00 UTC
            # quiet window that can push a run to 10:00 (gaps up to about 33 h), and the 12 h idle
            # interval for fetch-scores, each plus slack. Only fetch-odds-nba pauses overnight.
            "fetch-odds-epl": 49,
            "fetch-odds-nba": 35,
            "fetch-scores-epl": 13,
            "fetch-oddsportal-epl": 11,
            "fetch-oddsportal-results-epl": 26,
            "fetch-oddsportal-mlb": 12,
            "fetch-oddsportal-nba": 11,
            "fetch-mlb-results": 26,
            "fetch-mlb-probables": 30,
            # Daily 06:00 UTC cron, plus slack.
            "fetch-espn-fixtures": 26,
            "fetch-espn-fixtures-nba": 26,
            "fetch-betfair-exchange-epl": 10,
            "fetch-betfair-exchange-mlb": 11,
            "fetch-betfair-exchange-nba": 11,
            "fetch-polymarket-odds-epl": 11,
            "fetch-polymarket-odds-nba": 11,
            # Lead runs restart the 6 h cadence, so the last run of a night can be as late as
            # 01:00 UTC or later and the next is 15:00 UTC after the overnight window: up to
            # 14 h, plus 1 h slack.
            "fetch-injuries-nba": 15,
            "check-health": 2,
        },
        description="Max hours between job completions before alerting (job_name -> hours)",
    )

    # Heartbeat retention
    heartbeat_retention_days: int = Field(
        default=7,
        description="Days to retain heartbeat rows in AlertHistory before purging",
    )


class PolymarketConfig(BaseSettings):
    """Polymarket prediction market configuration."""

    model_config = SettingsConfigDict(
        env_prefix="POLYMARKET_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com", description="Gamma API base URL"
    )
    clob_base_url: str = Field(
        default="https://clob.polymarket.com", description="CLOB API base URL"
    )
    enabled: bool = Field(default=True, description="Enable Polymarket data collection")


class BetfairConfig(BaseSettings):
    """Betfair Exchange API configuration.

    Read-only ingestion via the delayed application key. The free delayed key
    returns prices with a 1-180s variable lag — sufficient as a sharp benchmark
    for pre-match decisions made hours before kickoff.

    Authentication: cert-based non-interactive login is the production path.
    For dev/local probes, leaving cert paths unset falls back to interactive
    login (which prompts 2FA if enabled — not viable unattended).
    """

    model_config = SettingsConfigDict(
        env_prefix="BETFAIR_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable Betfair Exchange ingestion")
    username: str | None = Field(default=None, description="Betfair account username")
    password: str | None = Field(default=None, description="Betfair account password")
    app_key: str | None = Field(default=None, description="Betfair application key (delayed)")
    cert_file: str | None = Field(default=None, description="Path to client SSL cert (.crt)")
    cert_key: str | None = Field(default=None, description="Path to client SSL key (.key)")

    # SSM SecureString parameter names holding the cert/key PEM contents.
    # When set, the Lambda materializes them into ``/tmp`` at cold start and
    # uses those paths (taking precedence over ``cert_file``/``cert_key``).
    # Lets the cert rotate without redeploying the container image.
    cert_pem_ssm_param: str | None = Field(
        default=None,
        description="SSM SecureString parameter holding the cert PEM (Lambda-only)",
    )
    key_pem_ssm_param: str | None = Field(
        default=None,
        description="SSM SecureString parameter holding the key PEM (Lambda-only)",
    )

    # Look-ahead window for event discovery (covers an EPL gameweek + MLB week)
    lookahead_hours: int = Field(
        default=168, description="Hours ahead to look when discovering events"
    )

    @model_validator(mode="after")
    def _require_creds_when_enabled(self) -> BetfairConfig:
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("BETFAIR_USERNAME", self.username),
                    ("BETFAIR_PASSWORD", self.password),
                    ("BETFAIR_APP_KEY", self.app_key),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"BETFAIR_ENABLED=true but credentials missing: {', '.join(missing)}"
                )
        return self


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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
        bookmakers = settings.data_collection.bookmakers_for("soccer_epl")
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Repository root (walk up from packages/odds-core/odds_core/config.py)
    project_root: Path = Field(default=Path(__file__).resolve().parents[3])

    # Composed configuration sections
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    betfair: BetfairConfig = Field(default_factory=BetfairConfig)

    @property
    def alerts_deliverable(self) -> bool:
        return self.alerts.alert_enabled and not self.scheduler.dry_run


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings; primarily for testing overrides."""
    get_settings.cache_clear()
