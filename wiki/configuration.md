---
type: component
title: Configuration (Settings)
summary: "The Pydantic Settings classes, env-var naming (prefixed vs bare), what is frozen at import, which variables bypass Settings, and the rules for changing config."
sources:
  - packages/odds-core/odds_core/config.py::Settings
  - packages/odds-core/odds_core/config.py::get_settings
  - packages/odds-core/odds_core/config.py::reset_settings_cache
  - packages/odds-core/odds_core/database.py
  - packages/odds-core/odds_core/alerts.py::alert_manager
  - packages/odds-cli/odds_cli/db_override.py::override_database_url
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py
  - docs/DEVELOPMENT.md#configuration-management
  - docs/DEVELOPMENT.md#modifying-configuration
verified_at: 7887039
---

# Configuration (Settings)

`packages/odds-core/odds_core/config.py` is authoritative for every setting and default ([DEVELOPMENT.md](../docs/DEVELOPMENT.md#configuration-management) keeps a summary table). `Settings` composes one `BaseSettings` class per concern; `get_settings()` returns a cached instance and `reset_settings_cache()` clears it. `.env.example` is the variable template.

## Sections

| Attribute | Class | Env names | Owning page |
|---|---|---|---|
| `api` | `APIConfig` | `ODDS_API_KEY`, `ODDS_API_KEYS`, `ODDS_API_BASE_URL`, `ODDS_API_QUOTA` | [Odds API client](odds-api-client.md) |
| `database` | `DatabaseConfig` | `DATABASE_URL`, `DATABASE_POOL_SIZE` | [Database connections](database-connections.md) |
| `data_collection` | `DataCollectionConfig` | `SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS` (bare) | [Data sources](data-sources.md) |
| `scheduler` | `SchedulerConfig` | `SCHEDULER_BACKEND`, `_DRY_RUN`, `_LOOKAHEAD_DAYS`, `_BOOTSTRAP_JOBS` | [Local scheduler](local-scheduler.md) |
| `aws` | `AWSConfig` | `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` | [AWS EventBridge backend](aws-eventbridge-backend.md) |
| `model` | `ModelConfig` | `MODEL_NAME`, `MODEL_BUCKET` | [S3 model store](s3-model-store.md) |
| `data_quality` | `DataQualityConfig` | `ENABLE_VALIDATION`, `REJECT_INVALID_ODDS` (bare) | [Odds validation](data-quality-validation.md) |
| `alerts` | `AlertConfig` | `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED`, thresholds, `HEARTBEAT_EXPECTATIONS` (bare) | [Alerts module](alerts-module.md) |
| `logging` | `LoggingConfig` | `LOG_LEVEL`, `LOG_FILE` | [Code conventions](code-conventions.md) |
| `polymarket`, `betfair` | `PolymarketConfig`, `BetfairConfig` | `POLYMARKET_*`, `BETFAIR_*` | [Polymarket](polymarket-pipeline.md), [Betfair](betfair-exchange-client.md) |

`Settings.project_root` is the repo root derived from the file location.

## Changing config

1. Add the field to the right class with a type and a default.
2. Add it to `.env.example`.
3. Keep a sensible default so existing environments keep working ([DEVELOPMENT.md](../docs/DEVELOPMENT.md#modifying-configuration)).

## Variables that bypass Settings

Read with `os.getenv` and so visible only in the process environment: `AGENT_DATABASE_URL` ([agent run](job-agent-run.md)), `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL` ([copy from prod](copy-from-production.md)), `LAMBDA_ARN` and `RULE_PREFIX` (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`), `AWS_LAMBDA_FUNCTION_NAME` (`packages/odds-core/odds_core/database.py`), `SSM_API_KEY_INDEX`.

## Hazards

- Copying `.env.example` to `.env` unchanged makes `get_settings()` raise `SettingsError` on `sports`: list fields (`SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS`, `SCHEDULER_BOOTSTRAP_JOBS`) are parsed as JSON arrays, and the template writes comma-separated values (reproduced). It also sets `SPORTS=basketball_nba` while the code default is EPL and MLB.
- The `.env.example` comment for `SCHEDULER_BOOTSTRAP_JOBS` lists `agent-run` and `daily-digest` as defaults; the code default (`SchedulerConfig`) has both commented out.
- `.env` is read by pydantic-settings relative to the working directory and never copied into `os.environ`, so the bypass variables above are not picked up from `.env` (reproduced). `python-dotenv` is declared but not imported.
- Import-time freezing: `odds_core.database` builds the engine from settings on import, and `odds_core.alerts.alert_manager` snapshots `alert_enabled` and the webhook at import. `override_database_url` updates `os.environ` and clears the settings cache, but `odds_core.database.engine` keeps the original URL (reproduced), so in-process `async_session_maker()` calls after `odds scheduler start --db X` still hit the original database; only subprocesses and later `get_settings()` calls see `X`.
- Eight sections are prefixed and three are bare (`SPORTS`, `ALERT_ENABLED`, `HEARTBEAT_EXPECTATIONS` as raw JSON), so names collide easily across sections.
- `AlertConfig`'s docstring says "infrastructure for future use" but it drives deployed alerting; its `heartbeat_expectations` keys hardcode compound job names that `make_compound_job_name` generates elsewhere.

## Related

- [Project overview](project-overview.md) — where config sits in the stack
- [Sport and market keys](sport-and-market-keys.md) — what `SPORTS` accepts
- [Timezone handling](timezone-handling.md) — related convention
