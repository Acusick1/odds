---
type: reference
title: Local scheduler
description: The APScheduler-on-Postgres backend, its scheduling primitives, the odds scheduler commands, bootstrap jobs and the cron table.
sources:
  - odds_lambda.scheduling.backends.local.LocalSchedulerBackend
  - odds_cli.commands.scheduler.start_local
  - odds_cli.commands.scheduler.info
  - odds_cli.commands.scheduler.smoke
  - odds_cli.commands.scheduler.health
  - odds_core.config.SchedulerConfig
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._SMOKE_UNSAFE_JOBS
  - odds_lambda.jobs.agent_run.schedule_next
  - odds_mcp.server.refresh_scrape
  - odds_mcp.server.get_scheduled_jobs
  - odds_lambda.jobs.fetch_oddsportal.ingest_league
  - packages/odds-lambda/pyproject.toml
status: stable
---

# Local scheduler

`odds_lambda.scheduling.backends.local.LocalSchedulerBackend` runs APScheduler 4 (pinned `4.0.0a6` in `packages/odds-lambda/pyproject.toml`) with `SQLAlchemyDataStore` and `AsyncpgEventBroker` on the project Postgres, so any process sharing the database can add a schedule and the running scheduler sees it via LISTEN/NOTIFY. It is the default backend (`SCHEDULER_BACKEND=local`) and the only home of jobs that have no EventBridge rule ([Where each job runs](job-deployment-matrix.md#where-each-job-runs)). The MCP tools `odds_mcp.server.refresh_scrape` and `odds_mcp.server.get_scheduled_jobs` use this store; `odds_mcp.server.refresh_scrape` queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` jobs that wait if no scheduler process is running ([MCP scheduling tools](../agent/mcp-server.md#scrape-and-scheduling-tools)). To run it as a service, see [Run the scheduler as a systemd service](howto-run-scheduler-service.md).

`odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_scheduled_jobs` returns `[]` only for a backend built with `dry_run=True`; `odds scheduler list-jobs` builds it without, so `SCHEDULER_DRY_RUN` does not empty the listing.

## Scheduling primitives

- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.schedule_next_execution` writes a one-shot `DateTrigger` schedule with `id=job_name` and `ConflictPolicy.replace`.
- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.schedule_cron` writes a `CronTrigger` (UTC) with the same id and policy.
- Both reuse a live scheduler (the backend's own, or `current_async_scheduler` while a job runs); otherwise `build_scheduler(role=SchedulerRole.scheduler)` opens a short-lived one that only writes to the store.
- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.cancel_scheduled_execution` needs a started scheduler.

## CLI

In `odds_cli.commands.scheduler`:

- `odds_cli.commands.scheduler.start_local`: exits 1 unless `SCHEDULER_BACKEND=local`. With bootstrap on, it first removes stored schedules not in `expected_compound_job_names(bootstrap_jobs, sports)`, then for each job in `odds_core.config.SchedulerConfig.bootstrap_jobs` either installs its `_JOB_CRON_MAP` cron (re-applied on every start) or runs its bootstrap function once, skipping jobs that already have a `next_run_time`. `--db NAME` swaps the database name in `DATABASE_URL` and defaults bootstrap off; `--bootstrap` forces it on. With `--db` set and bootstrap off, stale-schedule pruning is also skipped. Sidecar mode keeps a scheduler on a sandbox database from starting an agent run alongside one you are driving by hand. Invocations: `odds scheduler start` (prod-like, bootstrap on), `odds scheduler start --db odds_test` (dev sidecar, bootstrap off), `odds scheduler start --db odds_test --bootstrap` (dev full-loop smoke test); it runs until Ctrl+C. In-process sessions keep the original engine ([Databases and environments](../foundations/databases-and-environments.md#engine-creation)).
- `run-once JOB`: runs one registered job, compound names allowed (`fetch-odds-epl`), on any backend.
- `odds_cli.commands.scheduler.smoke`: runs every bootstrapped job once and prints PASS/FAIL/SKIP; exit code is the failed count (max 255). `agent-run` is skipped unless `--include-agent` (`_SMOKE_UNSAFE_JOBS`); `--no-side-effects` drops `daily-digest`, which posts to Discord. `--only` and `--exclude` take base or compound names and may repeat. Gated jobs run their body only when due, so smoke otherwise checks imports, config, decision and schema (#395).
- `list-jobs [--backend aws]`, `odds_cli.commands.scheduler.info`, `odds_cli.commands.scheduler.health`, `test-backend`: inspection. A fresh `odds_cli.commands.scheduler.health` process has no running scheduler and reports "initialized (not started)".

## Bootstrap jobs and cron table

`odds_core.config.SchedulerConfig.bootstrap_jobs` defaults to `fetch-oddsportal`, `fetch-oddsportal-results`, `fetch-espn-fixtures`, `fetch-betfair-exchange` and `fetch-mlb-probables`; `agent-run` and `daily-digest` are commented out there, so they are opt-in via `SCHEDULER_BOOTSTRAP_JOBS`. For `agent-run` the bootstrap function is `odds_lambda.jobs.agent_run.schedule_next`, not `odds_lambda.jobs.agent_run.main`.

`odds_lambda.scheduling.jobs._JOB_CRON_MAP`: `daily-digest` `0 8 * * *` (EPL), `fetch-espn-fixtures` `0 6 * * *` (EPL), `fetch-mlb-probables` `0 6 * * *` (MLB), `fetch-betfair-exchange` `*/30 * * * *` (every Betfair-supported sport).

`fetch-betfair-exchange-<sport>` uses one schedule id for its cron and its self-schedule, both with `ConflictPolicy.replace`, so the first self-schedule replaces the `*/30` cron until the next `odds scheduler start` (#469).

Inline CLV scoring inside `fetch-oddsportal` works only here: the scraper Lambda image lacks the analytics stack ([Prediction scoring](../modelling/prediction-scoring.md)). Running the same job on AWS and locally double-scrapes ([OddsPortal job](../collection/fetch-oddsportal-job.md)); check `odds scheduler list-jobs --backend aws` first ([Deployed state](job-deployment-matrix.md#deployed-state)).

## Related

- [Run the scheduler as a systemd service](howto-run-scheduler-service.md)
- [Scheduler backends](scheduler-backends.md)
- [Where each job runs](job-deployment-matrix.md)
- [Decision engine](decision-engine.md)
- [The odds CLI](../foundations/odds-cli.md)
