---
type: component
title: Configuration (Settings)
description: The Pydantic Settings sections and env names, what is frozen at import, which variables bypass Settings, how .env.example breaks if copied, and how to add a setting.
sources:
  - packages/odds-core/odds_core/config.py::Settings
  - packages/odds-core/odds_core/config.py::get_settings
  - packages/odds-core/odds_core/config.py::reset_settings_cache
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-core/odds_core/config.py::AlertConfig
  - packages/odds-core/odds_core/config.py::PolymarketConfig
  - packages/odds-core/odds_core/database.py
  - packages/odds-core/odds_core/alerts.py::alert_manager
  - packages/odds-cli/odds_cli/db_override.py::override_database_url
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py
  - deployment/aws/terraform/scraper.tf
status: stable
---

# Configuration (Settings)

`packages/odds-core/odds_core/config.py` is authoritative for every setting and default. `packages/odds-core/odds_core/config.py::Settings` composes one `BaseSettings` class per concern; `packages/odds-core/odds_core/config.py::get_settings()` returns a cached instance and `packages/odds-core/odds_core/config.py::reset_settings_cache()` clears it. `.env.example` is the variable template. the `project_root` field of `packages/odds-core/odds_core/config.py::Settings` is the repo root derived from the file location.

Model and feature settings (`target_type`, `sharp_bookmakers`) are not here: they live in the training YAML ([Training pipeline](../modelling/training-pipeline.md)).

## Sections

| Attribute | Class | Env names | Owning page |
|---|---|---|---|
| `api` | `packages/odds-core/odds_core/config.py::APIConfig` | `ODDS_API_KEY`, `ODDS_API_KEYS`, `ODDS_API_BASE_URL`, `ODDS_API_QUOTA` | [Odds API client and quota](../collection/odds-api-client-and-quota.md) |
| `database` | `packages/odds-core/odds_core/config.py::DatabaseConfig` | `DATABASE_URL`, `DATABASE_POOL_SIZE` | [Databases](databases-and-environments.md) |
| `data_collection` | `packages/odds-core/odds_core/config.py::DataCollectionConfig` | `SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS` (bare) | [Sources and keys](../collection/sources-and-keys.md) |
| `scheduler` | `packages/odds-core/odds_core/config.py::SchedulerConfig` | `SCHEDULER_BACKEND`, `_DRY_RUN`, `_LOOKAHEAD_DAYS`, `_BOOTSTRAP_JOBS` | [Local scheduler](../scheduling/local-scheduler.md) |
| `aws` | `packages/odds-core/odds_core/config.py::AWSConfig` | `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` | [Scheduler backends](../scheduling/scheduler-backends.md#aws-eventbridge-backend) |
| `model` | `packages/odds-core/odds_core/config.py::ModelConfig` | `MODEL_NAME`, `MODEL_BUCKET` | [Model publishing and scoring](../modelling/model-publishing-and-scoring.md) |
| `data_quality` | `packages/odds-core/odds_core/config.py::DataQualityConfig` | `ENABLE_VALIDATION`, `REJECT_INVALID_ODDS` (bare) | [Odds validation](../storage/data-validation.md) |
| `alerts` | `packages/odds-core/odds_core/config.py::AlertConfig` | `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED`, thresholds, `HEARTBEAT_EXPECTATIONS` (bare) | [Alerts and health](../operations/alerts-and-health.md) |
| `logging` | `packages/odds-core/odds_core/config.py::LoggingConfig` | `LOG_LEVEL`, `LOG_FILE` | [Conventions](conventions.md#logging) |
| `polymarket`, `betfair` | `packages/odds-core/odds_core/config.py::PolymarketConfig`, `packages/odds-core/odds_core/config.py::BetfairConfig` | `POLYMARKET_*`, `BETFAIR_*` | [Polymarket](../collection/polymarket.md), [Betfair Exchange](../collection/betfair-exchange.md) |

## Changing config

1. Add the field to the right class with a type and a default.
2. Add it to `.env.example`.
3. Keep a sensible default so existing environments keep working (inferred).
4. Never remove an existing field; deprecate it instead (inferred).


## Variables that bypass Settings

Read with `os.getenv` and so visible only in the process environment:

- `AGENT_DATABASE_URL` ([Betting agent](../agent/betting-agent.md#database-role)), `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL` ([copy from production](databases-and-environments.md#copy-production-data-locally)).
- `LAMBDA_ARN` and `RULE_PREFIX` on the deployed Lambdas (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`); a local `odds scheduler list-jobs --backend aws` uses the prefixed `AWS_LAMBDA_ARN` and `AWS_RULE_PREFIX` through `packages/odds-core/odds_core/config.py::AWSConfig`.
- `AWS_LAMBDA_FUNCTION_NAME` (`packages/odds-core/odds_core/database.py`), which switches the engine to `NullPool` ([Databases](databases-and-environments.md#engine-creation)).
- `SSM_API_KEY_INDEX` ([Odds API client and quota](../collection/odds-api-client-and-quota.md)).

## Hazards

- **`.env.example` breaks if copied.** Copying it to `.env` unchanged makes `packages/odds-core/odds_core/config.py::get_settings()` raise `SettingsError` on `sports`: list fields (`SPORTS`, `BOOKMAKERS`, `MARKETS`, `REGIONS`, `SCHEDULER_BOOTSTRAP_JOBS`) are parsed as JSON arrays, and the template writes comma-separated values (reproduced). It also sets `SPORTS=basketball_nba` while the code default is EPL and MLB.
- The `.env.example` comment for `SCHEDULER_BOOTSTRAP_JOBS` lists `agent-run` and `daily-digest` as defaults; the code default (`packages/odds-core/odds_core/config.py::SchedulerConfig`) has both commented out.
- `.env` is read by pydantic-settings relative to the working directory and never copied into `os.environ`, so the bypass variables above are not picked up from `.env` (reproduced). `python-dotenv` is declared but not imported.
- **Import-time freezing.** `odds_core.database` builds the engine from settings on import (NullPool keyed on `AWS_LAMBDA_FUNCTION_NAME`), and `odds_core.alerts.alert_manager` snapshots `ALERT_ENABLED` and the webhook at import. `packages/odds-cli/odds_cli/db_override.py::override_database_url` (the CLI `--db` flag) updates `os.environ` and clears the settings cache, but `odds_core.database.engine` keeps the original URL (reproduced), so in-process `async_session_maker()` calls after `odds scheduler start --db X` still hit the original database; only subprocesses and later `packages/odds-core/odds_core/config.py::get_settings()` calls see `X`.
- Eight sections are prefixed and three are bare (`SPORTS`, `ALERT_ENABLED`, `HEARTBEAT_EXPECTATIONS` as raw JSON), so names collide easily across sections.
- `packages/odds-core/odds_core/config.py::AlertConfig`'s docstring says "infrastructure for future use" but it drives deployed alerting; its `heartbeat_expectations` keys hardcode compound job names that `packages/odds-lambda/odds_lambda/scheduling/jobs.py::make_compound_job_name` generates elsewhere. `packages/odds-core/odds_core/config.py::AlertConfig` also has `ALERT_RATE_LIMIT_MINUTES`, which `packages/odds-core/odds_core/alerts.py::check_rate_limit` ignores (hard-coded 30) ([Alerts and health](../operations/alerts-and-health.md)).
- `REJECT_INVALID_ODDS` is read by nothing; validation never rejects a write ([Odds validation](../storage/data-validation.md)).
- `ODDS_API_QUOTA` (default 500) must be set for paid keys, otherwise quota alerts and `odds status quota` are wrong ([Odds API client and quota](../collection/odds-api-client-and-quota.md)).
- The `enabled` field of `packages/odds-core/odds_core/config.py::PolymarketConfig` defaults true while Terraform `enable_polymarket` defaults false; `BETFAIR_ENABLED` defaults false, and enabling it without credentials fails validation.
- `deployment/aws/terraform/scraper.tf` sets no `ALERT_ENABLED`, `DISCORD_WEBHOOK_URL`, `MODEL_NAME` or `MODEL_BUCKET`, so the scraper Lambda runs with the code defaults for those ([AWS infrastructure](../operations/aws-infrastructure.md)).

## Related

- [Project overview and pivots](project-overview.md) — where config sits in the stack
- [Sources and keys](../collection/sources-and-keys.md) — what `SPORTS` accepts
- [Conventions](conventions.md) — timezone and logging rules
