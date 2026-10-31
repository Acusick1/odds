---
type: howto
title: Add a setting
description: Steps to add a configuration setting so it loads from .env and reaches the production host.
sources:
  - odds_core.config
  - .env.example
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/variables.tf
status: stable
---

# Add a setting

How settings are structured and which variables bypass them is in [Configuration](configuration.md).

1. **Add the field** to the right settings class in `odds_core.config` (a default list such as `odds_core.config.SchedulerConfig.bootstrap_jobs` is edited in place, as for a new sport's heartbeat keys) (a derived switch such as `odds_core.config.Settings.alerts_deliverable` is a property on `odds_core.config.Settings`, not a field) with a type and a default (a credential that deployments may supply one of two ways, such as `odds_core.config.APIConfig.key` and `odds_core.config.APIConfig.keys`, defaults to `None`; [credential handling](../collection/odds-api-client.md#retry-and-keys)), plus a description (a dict default such as `odds_core.config.AlertConfig.heartbeat_expectations` maps compound job names (such as `fetch-polymarket-odds-epl`, `fetch-odds-epl`, `fetch-oddsportal-nba`, `fetch-injuries-nba`, or a global job's bare name such as `fetch-mlb-results`) to allowed hours rather than holding a scalar). The Betfair-close lookback hours is a setting on `odds_core.config.AlertConfig`; the repeat interval of a job-failure alert and of every health-monitor warning is `stale_reminder_minutes` there; per-source stale-data thresholds are not settings either: they are the job's cadence times the multipliers in `odds_core.config.AlertConfig` ([Stale-data check](../operations/health-monitor.md#stale-data-check)). Per-sport identifiers such as Polymarket series IDs are not settings; they live in a per-sport map beside their client ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#sport-config-and-discovery)).
   - A section with an `env_prefix` reads `<PREFIX><FIELD>`; a bare section reads the upper-cased field name; `odds_core.config.PolymarketConfig.enabled` is `POLYMARKET_ENABLED` ([Sections](configuration.md#sections)).
2. **Add it to `.env.example`**, which documents settings and is not their source of defaults. Remove a setting from it in the same change that removes the field.
   - Write list fields as JSON arrays and dict fields (such as `BOOKMAKERS` or `ODDS_API_KEY_INDEX_BY_SPORT`) as JSON objects; the template's comma-separated lists make `odds_core.config.get_settings` raise ([List fields and the template](configuration.md#list-fields-and-the-template)).
3. **Set it on the production host** if it needs a non-default value: add it to the `.env` in the production host's repo checkout, which `odds-scheduler.service` reads, then redeploy ([Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)).
4. **Read it through Settings.**
   - A variable read with `os.getenv` bypasses `.env` ([Configuration](configuration.md#variables-read-outside-settings)).

## Related

- [Configuration](configuration.md)
- [Set up and run locally](howto-develop-locally.md)
