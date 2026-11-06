---
type: reference
title: Configuration (Settings)
description: The Settings sections and env names, the JSON list fields, what is read at import, and the variables read with os.getenv outside Settings.
sources:
  - odds_core.config.Settings
  - odds_core.config.get_settings
  - odds_core.config.reset_settings_cache
  - odds_core.config.SchedulerConfig
  - odds_core.config.AlertConfig
  - odds_core.config.PolymarketConfig
  - odds_core.config.BackupConfig
  - odds_core.config.DataQualityConfig
  - odds_core.database
  - odds_core.alerts.alert_manager
  - odds_cli.db_override.override_database_url
  - odds_lambda.scheduling.backends.aws
status: stable
---

# Configuration (Settings)

`odds_core.config` defines operator settings and environment names. `odds_core.job_specs.JOB_SPECS` owns scheduling metadata and supplies default bootstrap membership, heartbeat expectations and forward polling cadences. `odds_core.config.SchedulerConfig.bootstrap_jobs` and `odds_core.config.AlertConfig.heartbeat_expectations` remain environment overrides; other defaults, including alert reminder intervals, belong to their settings classes. No setting names a season or its months: collectors go dormant from the fixture calendar ([dormancy](../scheduling/decision-engine.md#dormancy)). Per-sport Polymarket identifiers live in `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`. `odds_core.config.Settings` composes one `BaseSettings` class per concern; `odds_core.config.get_settings()` returns a cached instance and `odds_core.config.reset_settings_cache()` clears it. `odds_core.config.Settings.alerts_deliverable` is the one switch the alert paths read: `alert_enabled` and not `SCHEDULER_DRY_RUN`. The variable template is described in [Databases and environments](databases-and-environments.md#environments). The `project_root` field of `odds_core.config.Settings` is the repo root derived from the file location. To add a field, see [Set up and run locally](howto-add-setting.md).

`ODDS_API_KEYS` and `ODDS_API_KEY` are both optional; settings import without either of them, and `odds_lambda.data_fetcher.TheOddsAPIClient` raises when neither is set, or when a sport's key index (`ODDS_API_KEY_INDEX_BY_SPORT`, a JSON object; NBA is 1, every other sport 0) has no key ([keys](../collection/odds-api-client.md#retry-and-keys)). No setting holds a key's plan size; the client reads it from each response's quota headers ([cost per call](../collection/odds-api-client.md#cost-per-call)).

Model and feature settings (`target_type`, `sharp_bookmakers`) are not here: they live in the training YAML ([Training config](../modelling/training-config.md)).

## Sections

| Attribute | Class | Env names | Owning page |
|---|---|---|---|
| `api` | `odds_core.config.APIConfig` | `ODDS_API_KEY`, `ODDS_API_KEYS`, `ODDS_API_KEY_INDEX_BY_SPORT`, `ODDS_API_ACQUISITION_KEY_INDEX` (unset by default; `odds backfill execute` needs it), `ODDS_API_BASE_URL` | [Odds API client](../collection/odds-api-client.md) |
| `database` | `odds_core.config.DatabaseConfig` | `DATABASE_URL`, `DATABASE_POOL_SIZE`; `AGENT_DATABASE_URL` as `agent_url` ([Database role](../agent/agent-run-job.md#database-role)), `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL` as `prod_url` and `local_url` ([copy production data](howto-copy-production-data.md)) | [Databases and environments](databases-and-environments.md) |
| `data_collection` | `odds_core.config.DataCollectionConfig` | `SPORTS` (default: EPL, MLB and NBA), `BOOKMAKERS`, `MARKETS`, `REGIONS` (bare) | [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md) |
| `scheduler` | `odds_core.config.SchedulerConfig` | `SCHEDULER_BACKEND` (`local` or `aws`), `_DRY_RUN`, `_LOOKAHEAD_DAYS`, `_BOOTSTRAP_JOBS` | [Local scheduler](../scheduling/local-scheduler.md) |
| `aws` | `odds_core.config.AWSConfig` | `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` | [Scheduler backends](../scheduling/scheduler-backends.md) |
| `data_quality` | `odds_core.config.DataQualityConfig` | `ENABLE_VALIDATION` (bare), read only in `odds_lambda.ingestion.OddsIngestionService.ingest_sport` (Odds API ingestion, which resolves each listed game, an `odds_core.api_models.ApiEvent`, through the event resolver), whose `validate` argument overrides it | [Odds validation](../storage/odds-validator.md) |
| `alerts` | `odds_core.config.AlertConfig` | `DISCORD_WEBHOOK_URL`, `ALERT_ENABLED`, `DEAD_MANS_SWITCH_URL` (see [dead-man's switch](../operations/health-monitor.md#dead-mans-switch)), thresholds (stale data uses `STALE_DATA_HOURS` as a floor, `STALE_CADENCE_MULTIPLIER` and `STALE_LAST_INTERVAL_MULTIPLIER` times each job's cadence, `STALE_REMINDER_MINUTES` between repeats, `BETFAIR_CLOSE_LOOKBACK_HOURS` for the missing-Betfair-close check, see [health monitor](../operations/health-monitor.md#stale-data-check) and [Betfair close check](../operations/health-monitor.md#betfair-close-check)), `HEARTBEAT_EXPECTATIONS` (bare, hours per compound job name such as `fetch-espn-fixtures`, 2 to 49 by default) | [Alerts](../operations/alerts.md) |
| `backup` | `odds_core.config.BackupConfig` | `BACKUP_REMOTE`, `BACKUP_KEEP_DAILY`, `BACKUP_KEEP_WEEKLY`; `ODDS_BACKUP_DIR` as `dir`, shared with `scripts/backup-prod-db.sh` | [Nightly database backup job](database-backup-job.md) |
| `logging` | `odds_core.config.LoggingConfig` | `LOG_LEVEL`, `LOG_FILE` | [Code conventions](code-conventions.md#logging) |
| `polymarket`, `betfair` | `odds_core.config.PolymarketConfig`, `odds_core.config.BetfairConfig` | `POLYMARKET_GAMMA_BASE_URL`, `POLYMARKET_CLOB_BASE_URL`, `POLYMARKET_ENABLED`, `BETFAIR_*` (neither has a sport list; a job's sport comes from its compound name) | [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md) (series and tag IDs and team naming are per sport in `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`, not settings), [Betfair Exchange](../collection/betfair-exchange.md) |

Eight sections carry an env prefix and three are bare (`SPORTS`, `ALERT_ENABLED`, `HEARTBEAT_EXPECTATIONS` as raw JSON).

## List fields and the template

List fields (`SPORTS`, `MARKETS`, `REGIONS`, `SCHEDULER_BOOTSTRAP_JOBS`) are parsed as JSON arrays, and `BOOKMAKERS` as a JSON object of sport key to bookmaker list that replaces the whole default map; a comma-separated value makes `odds_core.config.get_settings()` raise `SettingsError`. The template leaves these fields commented out ([Databases and environments](databases-and-environments.md#environments)). `odds_core.config.SchedulerConfig` generates the code default for `SCHEDULER_BOOTSTRAP_JOBS` from `odds_core.job_specs.default_bootstrap_jobs`: the production job set, which includes `run-candidates`, `fetch-injuries`, `fetch-mlb-results` and `backup-database`, with `agent-run` excluded by its record ([Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)).

## Read at import

- `odds_core.alerts.alert_manager` snapshots `ALERT_ENABLED` and the webhook at import (#457).

## Variables read outside Settings

These are read with `os.getenv`. pydantic-settings reads `.env` relative to the working directory but never exports it to the process environment, so they must be exported in the shell:

- `LAMBDA_ARN` and `RULE_PREFIX` on the deployed Lambdas (`odds_lambda.scheduling.backends.aws`, whose schedule expressions are UTC); a local `odds scheduler list-jobs --backend aws` uses the prefixed `AWS_LAMBDA_ARN` and `AWS_RULE_PREFIX` through `odds_core.config.AWSConfig`.
- `AWS_LAMBDA_FUNCTION_NAME` (`odds_core.database`), which switches an engine built while it is set to `NullPool` ([Engine creation](databases-and-environments.md#engine-creation)).

## Related

- [About the system and its pivots](project-overview.md) — where config sits in the stack
- [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md) — what `SPORTS` accepts
- [Code conventions](code-conventions.md) — logging rules
