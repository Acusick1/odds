"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from odds_core.sports import SportKey


class APIConfig(BaseModel):
    """The Odds API configuration."""

    key: str = Field(..., description="The Odds API key")
    keys: str | None = Field(
        default=None, description="Comma-separated API keys for rotation (overrides key)"
    )
    base_url: str = Field(
        default="https://api.the-odds-api.com/v4", description="Base URL for The Odds API"
    )
    quota: int = Field(default=500, description="Monthly API request quota per key")


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = Field(..., description="PostgreSQL connection URL")
    pool_size: int = Field(default=5, description="Database connection pool size")


class DataCollectionConfig(BaseModel):
    """Data collection parameters."""

    sports: list[SportKey] = Field(
        default=["soccer_epl", "baseball_mlb"], description="Sports to track"
    )
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


class SchedulerConfig(BaseModel):
    """Scheduler backend configuration."""

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
            "fetch-oddsportal",
            "fetch-oddsportal-results",
            # "daily-digest",
            "fetch-espn-fixtures",
            "fetch-betfair-exchange",
            "fetch-mlb-probables",
        ],
        description="Jobs to bootstrap when starting the local scheduler",
    )


class AWSConfig(BaseModel):
    """AWS-specific configuration (only needed when scheduler_backend='aws')."""

    region: str | None = Field(default=None, description="AWS region")
    lambda_arn: str | None = Field(
        default=None,
        description="Lambda function ARN (for self-scheduling)",
    )
    rule_prefix: str | None = Field(
        default=None,
        description="EventBridge rule name prefix (e.g. 'odds') for querying deployed schedules",
    )


class ModelConfig(BaseModel):
    """CLV model artifact location (S3) for local scoring and Lambda inference."""

    name: str | None = Field(
        default=None,
        description="S3 key prefix for the model artifact (e.g. 'epl-clv-home')",
    )
    bucket: str | None = Field(
        default=None,
        description="S3 bucket containing '<name>/latest/model.pkl' and 'config.yaml'",
    )


class DataQualityConfig(BaseModel):
    """Data quality and validation settings."""

    enable_validation: bool = Field(default=True, description="Enable data quality validation")
    reject_invalid_odds: bool = Field(
        default=False, description="Reject invalid odds (vs log only)"
    )


class AlertConfig(BaseModel):
    """Alert configuration."""

    discord_webhook_url: str | None = Field(default=None, description="Discord webhook URL")
    alert_enabled: bool = Field(default=False, description="Enable alerts")

    # Quota monitoring thresholds
    quota_warning_threshold: float = Field(
        default=0.2,
        description="Trigger warning when API quota falls below this fraction (0-1)",
    )
    quota_critical_threshold: float = Field(
        default=0.1,
        description="Trigger critical alert when API quota falls below this fraction (0-1)",
    )

    # Failure tracking thresholds
    consecutive_failures_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before triggering alert",
    )

    # Data staleness monitoring
    stale_data_hours: int = Field(
        default=2,
        description="Hours without new data before triggering stale data alert",
    )

    # Alert rate limiting
    alert_rate_limit_minutes: int = Field(
        default=30,
        description="Minimum minutes between identical alerts to prevent spam",
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
            "fetch-oddsportal-epl": 2,
            "fetch-oddsportal-results-epl": 26,
            "daily-digest-epl": 26,
            "fetch-oddsportal-mlb": 2,
            "fetch-oddsportal-results-mlb": 26,
            "check-health": 2,
        },
        description="Max hours between job completions before alerting (job_name -> hours)",
    )

    # Heartbeat retention
    heartbeat_retention_days: int = Field(
        default=7,
        description="Days to retain heartbeat rows in AlertHistory before purging",
    )


class PolymarketConfig(BaseModel):
    """Polymarket prediction market configuration."""

    gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com", description="Gamma API base URL"
    )
    clob_base_url: str = Field(
        default="https://clob.polymarket.com", description="CLOB API base URL"
    )
    nba_series_id: str = Field(default="10345", description="NBA series tag ID")
    game_tag_id: str = Field(default="100639", description="NBA game tag ID")
    enabled: bool = Field(default=True, description="Enable Polymarket data collection")

    # Polling intervals (seconds)
    price_poll_interval: int = Field(
        default=300, description="Price snapshot poll interval (seconds)"
    )
    orderbook_poll_interval: int = Field(
        default=1800, description="Order book poll interval (seconds)"
    )

    # Market type collection toggles
    collect_moneyline: bool = Field(default=True, description="Collect moneyline markets")
    collect_spreads: bool = Field(default=True, description="Collect spread markets")
    collect_totals: bool = Field(default=True, description="Collect totals markets")
    collect_player_props: bool = Field(default=False, description="Collect player prop markets")

    # Order book collection tiers
    orderbook_tiers: list[str] = Field(
        default=["closing", "pregame"], description="Tiers for order book collection"
    )


class BetfairConfig(BaseModel):
    """Betfair Exchange API configuration.

    Read-only ingestion via the delayed application key. The free delayed key
    returns prices with a 1-180s variable lag — sufficient as a sharp benchmark
    for pre-match decisions made hours before kickoff.

    Authentication: cert-based non-interactive login is the production path.
    For dev/local probes, leaving cert paths unset falls back to interactive
    login (which prompts 2FA if enabled — not viable unattended).
    """

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

    # Sport scoping override. ``None`` means "all sports the BFE adapter supports"
    # (resolved at job-execution time from ``odds_lambda.betfair.SPORT_CONFIG``).
    sports: list[SportKey] | None = Field(
        default=None,
        description="Override sport scope; defaults to all BFE-supported sports",
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


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    file: str = Field(default="logs/odds_pipeline.log", description="Log file path")


class _NestedEnvSource(PydanticBaseSettingsSource):
    """
    Maps prefixed/unprefixed flat env vars to Settings' nested sub-config fields.

    Reads from os.environ (higher priority) and the dotenv file (lower priority),
    respecting Settings(_env_file=None) by delegating file reads to the dotenv source.
    """

    # Maps Settings field name -> uppercase env var prefix
    _PREFIXED: dict[str, str] = {
        "api": "ODDS_API_",
        "database": "DATABASE_",
        "scheduler": "SCHEDULER_",
        "aws": "AWS_",
        "model": "MODEL_",
        "logging": "LOG_",
        "polymarket": "POLYMARKET_",
        "betfair": "BETFAIR_",
    }

    # Maps Settings field name -> explicit uppercase env var names (no shared prefix)
    _UNPREFIXED: dict[str, list[str]] = {
        "data_collection": ["SPORTS", "BOOKMAKERS", "MARKETS", "REGIONS"],
        "data_quality": ["ENABLE_VALIDATION", "REJECT_INVALID_ODDS"],
        "alerts": [
            "DISCORD_WEBHOOK_URL",
            "ALERT_ENABLED",
            "QUOTA_WARNING_THRESHOLD",
            "QUOTA_CRITICAL_THRESHOLD",
            "CONSECUTIVE_FAILURES_THRESHOLD",
            "STALE_DATA_HOURS",
            "ALERT_RATE_LIMIT_MINUTES",
            "DATA_QUALITY_ERROR_THRESHOLD",
            "HEARTBEAT_EXPECTATIONS",
            "HEARTBEAT_RETENTION_DAYS",
        ],
    }

    def __init__(
        self, settings_cls: type[BaseSettings], dotenv_settings: PydanticBaseSettingsSource
    ) -> None:
        super().__init__(settings_cls)
        self._dotenv = dotenv_settings

    def _all_vars(self) -> dict[str, str]:
        """Merge dotenv (lower priority) and os.environ (higher priority). All keys lowercase."""
        merged: dict[str, str] = {}
        for k, v in self._dotenv._read_env_files().items():  # type: ignore[attr-defined]
            if v is not None:
                merged[k.lower()] = v
        for k, v in os.environ.items():
            merged[k.lower()] = v
        return merged

    @staticmethod
    def _maybe_parse_json(v: str) -> Any:
        stripped = v.strip()
        if stripped and stripped[0] in ("[", "{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return v

    def __call__(self) -> dict[str, Any]:
        env = self._all_vars()
        result: dict[str, Any] = {}

        for field_name, prefix in self._PREFIXED.items():
            lprefix = prefix.lower()
            group = {
                k[len(lprefix) :]: self._maybe_parse_json(v)
                for k, v in env.items()
                if k.startswith(lprefix)
            }
            if group:
                result[field_name] = group

        for field_name, env_keys in self._UNPREFIXED.items():
            group: dict[str, Any] = {}
            for key in env_keys:
                lkey = key.lower()
                if lkey in env:
                    group[lkey] = self._maybe_parse_json(env[lkey])
            if group:
                result[field_name] = group

        return result

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        return None, field_name, False


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # Repository root (walk up from packages/odds-core/odds_core/config.py)
    project_root: Path = Field(default=Path(__file__).resolve().parents[3])

    # Sub-configs with required fields — no default; must be provided via env vars
    api: APIConfig
    database: DatabaseConfig

    # Sub-configs with all-optional fields — defaults apply when env vars are absent
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    betfair: BetfairConfig = Field(default_factory=BetfairConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _NestedEnvSource(settings_cls, dotenv_settings),
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings; primarily for testing overrides."""
    get_settings.cache_clear()
