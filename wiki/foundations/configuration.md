---
type: reference
title: Configuration (Settings)
description: The Settings sections and env names, what is read at import, and the variables read with os.getenv outside Settings.
sources:
  - odds_core.config.Settings
  - odds_core.config.get_settings
  - odds_core.config.reset_settings_cache
  - odds_core.config.SchedulerConfig
  - odds_core.config.AlertConfig
  - odds_core.config.PolymarketConfig
  - odds_core.config.DataQualityConfig
  - odds_core.database
  - odds_core.alerts.alert_manager
  - odds_cli.db_override.override_database_url
  - odds_lambda.scheduling.backends.aws
  - .env.example
status: stable
---

# Configuration (Settings)

`odds_core.config` is authoritative for every setting and default. `odds_core.config.Settings` composes one `BaseSettings` class per concern; `odds_core.config.get_settings()` returns a cached instance and `odds_core.config.reset_settings_cache()` clears it. `.env.example` is the variable template. The `project_root` field of `odds_core.config.Settings` is the repo root derived from the file location. To add a field, see [Set up and run locally](howto-develop-locally.md#add-a-setting).

Model and feature settings (`target_type`, `sharp_bookmakers`) are not here: they live in the training YAML ([Training config](../modelling/training-config.md)).

## Sections

| Attribute | Class | Env names | Owning page |
|---|---|---|---|
| `api` | `odds_core.config.APIConfig` | `ODDS_API_KEY`, `ODDS_API_KEYS`, `ODDS_API_BASE_URL`, `ODDS_API_QUOTA` | [Odds API client](../collection/odds-api-client.md) |
| `database` | `odds_core.config.DatabaseConfig` | `DATABASE_URL`, `DATABASE_POOL_SIZE` | [Databases and environments](databases-and-environments.md) |
| `data_collection` | `odds_core.config.DataCollectionConfig` | `SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS` (bare) | [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md) |
| `scheduler` | `odds_core.config.SchedulerConfig` | `SCHEDULER_BACKEND`, `_DRY_RUN`, `_LOOKAHEAD_DAYS`, `_BOOTSTRAP_JOBS` | [Local scheduler](../scheduling/local-scheduler.md) |
| `aws` | `odds_core.config.AWSConfig` | `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` | [Scheduler backends](../scheduling/scheduler-backends.md) |
| `model` | `odds_core.config.ModelConfig` | `MODEL_NAME`, `MODEL_BUCKET` | [Model store](../modelling/model-store.md) |
| `data_quality` | `odds_core.config.DataQualityConfig` | `ENABLE_VALIDATION`, `REJECT_INVALID_ODDS` (bare); `reject_invalid_odds` is read by nothing (#448) | [Odds validation](../storage/odds-validator.md) |
| `alerts` | `odds_core.config.AlertConfig` | `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED`, thresholds, `HEARTBEAT_EXPECTATIONS` (bare) | [Alerts](../operations/alerts.md) |
| `logging` | `odds_core.config.LoggingConfig` | `LOG_LEVEL`, `LOG_FILE` | [Code conventions](code-conventions.md#logging) |
| `polymarket`, `betfair` | `odds_core.config.PolymarketConfig`, `odds_core.config.BetfairConfig` | `POLYMARKET_*`, `BETFAIR_*` | [Polymarket](../collection/polymarket.md), [Betfair Exchange](../collection/betfair-exchange.md) |

Eight sections carry an env prefix and three are bare (`SPORTS`, `ALERT_ENABLED`, `HEARTBEAT_EXPECTATIONS` as raw JSON).

## List fields and the template

List fields (`SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS`, `SCHEDULER_BOOTSTRAP_JOBS`) are parsed as JSON arrays. `.env.example` writes them comma-separated and sets `SPORTS=basketball_nba`, so copying it unchanged makes `odds_core.config.get_settings()` raise `SettingsError` on `sports` (#444). The code default for `SCHEDULER_BOOTSTRAP_JOBS` is `odds_core.config.SchedulerConfig`'s, which has `agent-run` and `daily-digest` commented out.

## Read at import

- `odds_core.database` builds the engine from settings on import of the module, with `NullPool` keyed on `AWS_LAMBDA_FUNCTION_NAME` (#445). `odds_cli.db_override.override_database_url` (the CLI `--db` flag) therefore does not reach it: see [Engine creation](databases-and-environments.md#engine-creation).
- `odds_core.alerts.alert_manager` snapshots `ALERT_ENABLED` and the webhook at import (#457).

## Variables read outside Settings

These are read with `os.getenv`. pydantic-settings reads `.env` relative to the working directory but never exports it to the process environment, so they must be exported in the shell (#444):

- `AGENT_DATABASE_URL` ([Database role](../agent/agent-run-job.md#database-role)), `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL` ([copy production data](howto-copy-production-data.md)).
- `LAMBDA_ARN` and `RULE_PREFIX` on the deployed Lambdas (`odds_lambda.scheduling.backends.aws`); a local `odds scheduler list-jobs --backend aws` uses the prefixed `AWS_LAMBDA_ARN` and `AWS_RULE_PREFIX` through `odds_core.config.AWSConfig`.
- `AWS_LAMBDA_FUNCTION_NAME` (`odds_core.database`), which switches the engine to `NullPool` ([Engine creation](databases-and-environments.md#engine-creation)).
- `SSM_API_KEY_INDEX` ([Odds API client](../collection/odds-api-client.md)).

## Related

- [About the system and its pivots](project-overview.md) — where config sits in the stack
- [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md) — what `SPORTS` accepts
- [Code conventions](code-conventions.md) — logging rules
