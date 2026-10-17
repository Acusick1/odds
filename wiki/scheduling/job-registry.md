---
type: reference
title: Job registry
description: Which jobs exist in the registry, the bootstrap and smoke maps, which jobs are per sport and how compound names are built, and what JobContext carries.
sources:
  - odds_lambda.scheduling.jobs._JOB_MODULE_MAP
  - odds_lambda.scheduling.jobs._JOB_BOOTSTRAP_MAP
  - odds_lambda.scheduling.jobs._PER_SPORT_JOBS
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._SMOKE_UNSAFE_JOBS
  - odds_lambda.scheduling.jobs._OUTWARD_POSTING_JOBS
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

`odds_lambda.scheduling.jobs` is the source of truth for which jobs exist. It is a different thing from the `odds_lambda/jobs/` package that holds the job modules, so "jobs" imports are ambiguous (`odds_lambda.jobs.update_status`'s docstring cites a path, scheduler/jobs, that does not exist). Where each registered job runs is on [Where each job runs](job-deployment-matrix.md#where-each-job-runs).

## Registry

- `_JOB_MODULE_MAP`: job name to `(module path, "main")` for 14 jobs: `fetch-odds`, `fetch-scores`, `update-status`, `check-health`, `fetch-polymarket`, `backfill-polymarket`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`.
- `odds_lambda.scheduling.jobs.get_job_function` imports the module on first use and caches it, so the scraper Lambda never imports xgboost-dependent modules unless asked.
- `_JOB_BOOTSTRAP_MAP`: jobs whose local bootstrap uses a different function; only `agent-run` (`odds_lambda.jobs.agent_run.schedule_next`). `odds_lambda.scheduling.jobs.get_bootstrap_function` wraps it to the `(JobContext)` signature. As a result `odds scheduler start` never launches the agent run itself; it only books its next wake-up.
- `_SMOKE_UNSAFE_JOBS` (`agent-run`) and `_OUTWARD_POSTING_JOBS` (`daily-digest`) drive [`odds scheduler smoke`](local-scheduler.md#cli); smoke skips `agent-run` unless `--include-agent`.
- `_JOB_CRON_MAP` holds the local cron jobs ([cron table](local-scheduler.md#bootstrap-jobs-and-cron-table)). At import the registry asserts every `_JOB_CRON_MAP` key is a registered job.

## Compound job names

`_PER_SPORT_JOBS` lists the jobs that take a sport: `fetch-odds`, `fetch-scores`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`. `update-status`, `check-health` and both Polymarket jobs are global.

`_SPORT_SUFFIX_MAP` knows only `epl` (`soccer_epl`) and `mlb` (`baseball_mlb`). `odds_lambda.scheduling.jobs.make_compound_job_name("fetch-odds", "soccer_epl")` returns `fetch-odds-epl`, which is the EventBridge rule suffix and the APScheduler schedule id; `odds_lambda.scheduling.jobs.resolve_job_name` reverses it. A sport with no suffix, such as NBA, yields the bare job name. The compound name must equal the rule name, or a reschedule goes nowhere ([Add a scheduler job](howto-add-scheduler-job.md)). Adding a sport touches several tables, listed in [Adding a sport](../collection/howto-add-sport.md).

## JobContext

`odds_lambda.scheduling.jobs.JobContext` is the dataclass every `main(ctx)` receives: `sport`, `include_spreads`, `include_totals`, `odds_lambda.scheduling.jobs.JobContext.dry_run`, `lookback_hours` (24), `lookahead_hours` (48). `odds_lambda.scheduling.jobs.JobContext.from_payload` keeps known keys and drops unknown ones with a debug log, so a typo in an EventBridge payload is silent. Jobs read dry-run from `settings.scheduler.dry_run`; only `backfill-polymarket` reads `ctx.dry_run`.

## Related

- [Where each job runs](job-deployment-matrix.md)
- [Self-scheduling](self-scheduling.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
