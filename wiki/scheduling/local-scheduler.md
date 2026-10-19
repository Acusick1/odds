---
type: reference
title: Local scheduler
description: The APScheduler-on-Postgres backend, its scheduling primitives, the odds scheduler commands, bootstrap jobs and the cron table.
sources:
  - odds_lambda.scheduling.backends.local.LocalSchedulerBackend
  - odds_lambda.scheduling.backends.local.utc_cron_trigger
  - odds_lambda.scheduling.bootstrap.bootstrap_job
  - odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_one_off_run
  - odds_cli.commands.scheduler.start_local
  - odds_cli.commands.scheduler.info
  - odds_cli.commands.scheduler.smoke
  - odds_cli.commands.scheduler.health
  - odds_core.config.SchedulerConfig
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_lambda.scheduling.jobs.resolve_target_sports
  - odds_lambda.scheduling.jobs._SMOKE_UNSAFE_JOBS
  - odds_lambda.jobs.agent_run.schedule_next
  - odds_mcp.server.refresh_scrape
  - odds_mcp.server.get_scheduled_jobs
  - odds_lambda.jobs.fetch_oddsportal.ingest_league
  - packages/odds-lambda/pyproject.toml
status: stable
---

# Local scheduler

`odds_lambda.scheduling.backends.local.LocalSchedulerBackend` runs APScheduler 4 (pinned `4.0.0a6` in `packages/odds-lambda/pyproject.toml`) with `SQLAlchemyDataStore` and `AsyncpgEventBroker` on the project Postgres, so any process sharing the database can add a schedule and the running scheduler sees it via LISTEN/NOTIFY. It is the default backend (`SCHEDULER_BACKEND=local`), and the instance on the mini-PC is production: every scheduled job runs there ([Where each job runs](job-deployment-matrix.md#where-each-job-runs), [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). The MCP tools `odds_mcp.server.refresh_scrape` and `odds_mcp.server.get_scheduled_jobs` use this store; `odds_mcp.server.refresh_scrape` queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` jobs (each scrapes OddsPortal itself over HTTP) that wait if no scheduler process is running and, unlike the scheduled job, neither alert on an empty run nor write a heartbeat ([MCP scheduling tools](../agent/mcp-server.md#scrape-and-scheduling-tools)). To run it as a service, see [Run the scheduler as a systemd service](howto-run-scheduler-service.md).

`odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_scheduled_jobs` returns `[]` only for a backend built with `dry_run=True`; `odds scheduler list-jobs` builds it without, so `SCHEDULER_DRY_RUN` does not empty the listing.

## Scheduling primitives

- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.schedule_next_execution` writes a one-shot `DateTrigger` schedule with `id=job_name` and `ConflictPolicy.replace`. It raises `odds_lambda.scheduling.exceptions.SchedulingFailedError` for a job in `odds_lambda.scheduling.jobs._JOB_CRON_MAP`, because the one-off run would replace that job's cron.
- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.schedule_cron` writes a `CronTrigger` with the same id and policy, built by `odds_lambda.scheduling.backends.local.utc_cron_trigger`: UTC fields, matched from the current UTC time. APScheduler's default start time is the naive local time, which on the mini-PC in BST put a `*/30` cron's first fire an hour late (#562).
- Both reuse a live scheduler (the backend's own, or `current_async_scheduler` while a job runs); otherwise `build_scheduler(role=SchedulerRole.scheduler)` opens a short-lived one that only writes to the store.
- `odds_lambda.scheduling.backends.local.LocalSchedulerBackend.cancel_scheduled_execution` needs a started scheduler.

## CLI

In `odds_cli.commands.scheduler`:

- `odds_cli.commands.scheduler.start_local` exits 1 unless `SCHEDULER_BACKEND=local`. With bootstrap on, it first removes stored schedules not in `expected_compound_job_names(bootstrap_jobs, sports)`, then calls `odds_lambda.scheduling.bootstrap.bootstrap_job` for each job in `odds_core.config.SchedulerConfig.bootstrap_jobs` ([Bootstrap jobs and cron table](#bootstrap-jobs-and-cron-table)). A per-sport job is expanded only for the sports `odds_lambda.scheduling.jobs.resolve_target_sports` returns, and each configured sport outside its allowlist prints a skip line. `--db NAME` swaps the database name in `DATABASE_URL` and defaults bootstrap off; `--bootstrap` forces it on. With `--db` set and bootstrap off, stale-schedule pruning is also skipped. Sidecar mode keeps a scheduler on a sandbox database from starting an agent run alongside one you are driving by hand. Invocations: `odds scheduler start` (prod-like, bootstrap on), `odds scheduler start --db odds_test` (dev sidecar, bootstrap off), `odds scheduler start --db odds_test --bootstrap` (dev full-loop smoke test); it runs until Ctrl+C. In-process sessions keep the original engine ([Databases and environments](../foundations/databases-and-environments.md#engine-creation)).
- `run-once JOB`: runs one registered job, compound names allowed (`fetch-odds-epl`), on any backend.
- `odds_cli.commands.scheduler.smoke`: runs every bootstrapped job once and prints PASS/FAIL/SKIP; exit code is the failed count (max 255). `agent-run` is skipped unless `--include-agent` (`_SMOKE_UNSAFE_JOBS`); `--no-side-effects` drops `daily-digest`, which posts to Discord. `--only` and `--exclude` take base or compound names and may repeat. Gated jobs run their body only when due, so smoke otherwise checks imports, config, decision and schema (#395). It calls each job's real `main()`: `SCHEDULER_DRY_RUN` suppresses only schedule writes, so a due `fetch-odds` or `fetch-scores` spends Odds API quota, and `fetch-odds` always calls the free `/events` endpoint.
- `list-jobs [--backend aws]`, `odds_cli.commands.scheduler.info` (settings, including each sport's Odds API bookmakers), `odds_cli.commands.scheduler.health`, `test-backend`: inspection. A fresh `odds_cli.commands.scheduler.health` process has no running scheduler and reports "initialized (not started)".

## Bootstrap jobs and cron table

`odds_core.config.SchedulerConfig.bootstrap_jobs` is the production job set; `agent-run` and `daily-digest` are commented out there, so they are opt-in via `SCHEDULER_BOOTSTRAP_JOBS`, as is the registered `fetch-polymarket-odds` until [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md) is accepted. On a development machine the default set includes `fetch-odds` and `fetch-scores`, so `odds scheduler start` there spends Odds API quota; set `SCHEDULER_BOOTSTRAP_JOBS` to the jobs you need. For `agent-run` the bootstrap function is `odds_lambda.jobs.agent_run.schedule_next`, not `odds_lambda.jobs.agent_run.main`.

`odds_lambda.scheduling.bootstrap.bootstrap_job` installs one job, per target sport for a per-sport job:

- A job in `odds_lambda.scheduling.jobs._JOB_CRON_MAP` has its cron re-applied on every start, so an edit to the expression takes effect.
- A self-scheduling job that has a pending one-off run (`odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_one_off_run`) keeps it. A restart, such as a deploy, therefore never moves the run the job booked for itself.
- A self-scheduling job without one has any other stored schedule removed (such as a cron stored for it under an older `_JOB_CRON_MAP`), then runs its bootstrap function inline. For the data jobs that is a full run that books the next one.

`odds_lambda.scheduling.jobs._JOB_CRON_MAP` holds the fixed cron expressions: `daily-digest` `0 8 * * *`, `fetch-espn-fixtures` `0 6 * * *`, `fetch-mlb-probables` `0 6 * * *`. A cron and a self-schedule share one schedule id, so a self-scheduling job never has a cron entry.

`odds_lambda.scheduling.jobs._JOB_SPORTS` holds the sport allowlists for cron and self-scheduling jobs alike; [Where each job runs](job-deployment-matrix.md#where-each-job-runs) lists each job's sports. An allowlist keeps a job whose `main()` raises on other sports from alerting on every fire, or keeps a job from spending quota on every configured sport. A sport in an allowlist but absent from `data_collection.sports` is not scheduled.

Inline CLV scoring inside `fetch-oddsportal` works only here: the scraper Lambda image lacks the analytics stack ([Prediction scoring](../modelling/prediction-scoring.md)). A re-enabled AWS rule would run the same job on both hosts and double-fetch; `odds scheduler list-jobs --backend aws` shows the rule states ([Deployed state](job-deployment-matrix.md#deployed-state)).

## Related

- [Run the scheduler as a systemd service](howto-run-scheduler-service.md)
- [Scheduler backends](scheduler-backends.md)
- [Where each job runs](job-deployment-matrix.md)
- [Decision engine](decision-engine.md)
- [The odds CLI](../foundations/odds-cli.md)
