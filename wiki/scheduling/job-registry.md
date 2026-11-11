---
type: reference
title: Job registry
description: Which jobs exist in the registry, bootstrap and smoke policies, which jobs are per sport and how compound names are built, and what JobContext carries.
sources:
  - odds_core.job_specs
  - odds_core.job_specs.JOB_SPECS
  - odds_lambda.scheduling.jobs.get_job_sports
  - odds_lambda.scheduling.jobs.resolve_target_sports
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_lambda.scheduling.jobs.JobContext
  - odds_lambda.scheduling.jobs.get_job_function
  - odds_lambda.scheduling.jobs.get_bootstrap_function
  - odds_lambda.scheduling.jobs.make_compound_job_name
  - odds_lambda.scheduling.jobs.resolve_job_name
  - odds_lambda.jobs.update_status
status: stable
---

# Job registry

`odds_core.job_specs.JOB_SPECS` contains immutable `odds_core.job_specs.JobSpec` records. Each record owns the entry point, sport routing (`fetch-espn-fixtures` lists seven sports; `fetch-oddsportal-results` lists the six football leagues, and `fetch-oddsportal` and `fetch-betfair-exchange` carry a heartbeat for each league they collect), bootstrap default and override, smoke side effects, forward cadence, static cron and heartbeat identifiers. Core stores module names without importing runtime code; `odds_lambda.scheduling.jobs` resolves names and imports entry points lazily. [Where each job runs](job-deployment-matrix.md#where-each-job-runs) covers deployment.

## Registry

- `module` and `function` identify the normal entry point. `odds_lambda.scheduling.jobs.get_job_function` imports it on first use and caches the function.
- `bootstrap_function` optionally names a different function in the same module. `odds_lambda.scheduling.jobs.get_bootstrap_function` adapts it to `odds_lambda.scheduling.jobs.JobContext`; `agent-run` books its next wake-up through `odds_lambda.jobs.agent_run.schedule_next`.
- `bootstrap` controls membership in `odds_core.job_specs.default_bootstrap_jobs`. `odds_core.config.SchedulerConfig.bootstrap_jobs` remains the operator override, including an explicitly empty list.
- `agent_trades`, `candidate_trades` and `paid_api` drive the separate smoke opt-ins `--include-agent`, `--include-candidates` and `--include-paid-api` ([CLI](local-scheduler.md#cli)).
- `cron` gives the fixed local schedule. A self-scheduling job leaves it unset; bootstrap otherwise replaces its pending run with that cron.
- `per_sport` controls compound naming. `sports` optionally restricts configured sports through an explicit tuple or a lazy `odds_core.job_specs.SportMap` reference to the adapter's authoritative map; `odds_lambda.scheduling.jobs.resolve_target_sports` intersects the allowlist with the operator's configured sports. An unrestricted job runs for every configured sport.
- `cadence` holds the default forward polling intervals; `sport_cadences` supplies sport overrides. `odds_core.job_specs.job_cadence` selects them without runtime imports. Runtime `odds_lambda.scheduling.decision.CadenceConfig` adds tier lookup to the core value record.
- `heartbeats` carries exact persisted identifiers, allowed hours and sports. `odds_core.job_specs.default_heartbeat_expectations` supplies defaults to `odds_core.config.AlertConfig.heartbeat_expectations`; that setting remains an independent operator override.

## Compound job names

`odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` maps sport keys to `epl`, `mlb`, `nba`, `championship`, `laliga`, `seriea`, `bundesliga` and `ligue1`. `odds_lambda.scheduling.jobs.make_compound_job_name("fetch-odds", "soccer_epl")` returns `fetch-odds-epl`, the EventBridge rule suffix and APScheduler schedule id; `odds_lambda.scheduling.jobs.resolve_job_name` reverses it. An unknown sport suffix keeps the bare name; a per-sport job given no sport raises `ValueError`.

Heartbeat identifiers are separate from schedule ids. `odds_core.job_specs.heartbeat_name` preserves the bare `fetch-mlb-probables` and EPL `fetch-espn-fixtures` identifiers; their schedules remain `fetch-mlb-probables-mlb` and `fetch-espn-fixtures-epl`. NBA ESPN uses `fetch-espn-fixtures-nba` for both.

## JobContext

`odds_lambda.scheduling.jobs.JobContext` is the dataclass every `main(ctx)` receives; it carries only `sport`. `odds_lambda.scheduling.jobs.JobContext.from_payload` keeps known keys and drops unknown ones with a debug log, so a typo in an EventBridge payload is silent. `odds_lambda.scheduling.jobs.require_sport` returns `ctx.sport` or raises `ValueError`; every per-sport `main` calls it first, so a run without a sport fails before it books anything. Jobs read dry-run from `settings.scheduler.dry_run`.

## Related

- [Where each job runs](job-deployment-matrix.md)
- [Self-scheduling](self-scheduling.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
