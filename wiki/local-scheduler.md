---
type: component
title: Local scheduler (APScheduler)
summary: "The `odds scheduler` start/run-once/smoke/list-jobs commands on APScheduler 4 with a Postgres store: bootstrap_jobs, cron floors, dry-run and `--db` sandboxing."
sources:
  - docs/ARCHITECTURE.md#local-backend
  - docs/DEPLOYMENT.md#local-scheduler
  - docs/DEPLOYMENT.md#smoke-check
  - docs/CLI.md#scheduler-commands
  - docs/DEVELOPMENT.md#scheduler-settings-schedulerconfig
  - docs/REPORTING.md#5-cli-status-surfaces
  - packages/odds-lambda/odds_lambda/scheduling/backends/local.py::LocalSchedulerBackend
  - packages/odds-cli/odds_cli/commands/scheduler.py::start_local
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
verified_at: 7887039
---

# Local scheduler (APScheduler)

`packages/odds-lambda/odds_lambda/scheduling/backends/local.py::LocalSchedulerBackend` runs APScheduler 4 (pinned `4.0.0a6` in `packages/odds-lambda/pyproject.toml`) with `SQLAlchemyDataStore` and `AsyncpgEventBroker` on the project Postgres, so any process sharing the database can add a schedule and the running scheduler sees it via LISTEN/NOTIFY. It is the default backend (`SCHEDULER_BACKEND=local`) and the only home of jobs that have no EventBridge rule (see [job deployment matrix](job-deployment-matrix.md)).

## Scheduling primitives

- `schedule_next_execution` writes a one-shot `DateTrigger` schedule with `id=job_name` and `ConflictPolicy.replace`.
- `schedule_cron` writes a `CronTrigger` (UTC) with the same id and policy.
- Both reuse a live scheduler (the backend's own, or `current_async_scheduler` while a job runs); otherwise `build_scheduler(role=SchedulerRole.scheduler)` opens a short-lived one that only writes to the store.
- `cancel_scheduled_execution` needs a started scheduler.

## CLI (`packages/odds-cli/odds_cli/commands/scheduler.py`)

- `start`: exits 1 unless `SCHEDULER_BACKEND=local`. With bootstrap on, it first removes stored schedules not in `expected_compound_job_names(bootstrap_jobs, sports)`, then for each job in `SchedulerConfig.bootstrap_jobs` either installs its `_JOB_CRON_MAP` cron (re-applied on every start) or runs its bootstrap function once, skipping jobs that already have a `next_run_time`. `--db NAME` swaps the database name in `DATABASE_URL` and defaults bootstrap off; `--bootstrap` forces it on.
- `run-once JOB`: runs one registered job, compound names allowed (`fetch-odds-epl`), on any backend. Its help text lists three jobs but any registered name works.
- `smoke`: runs every bootstrapped job once and prints PASS/FAIL/SKIP; exit code is the failed count (max 255). `agent-run` is skipped unless `--include-agent` (`_SMOKE_UNSAFE_JOBS`); `--no-side-effects` drops `daily-digest`. Gated jobs only run their body when `decision.should_execute` is true. `deploy.sh` runs it under `SCHEDULER_DRY_RUN=true` ([host deploy how-to](howto-deploy-local-host.md)).
- `list-jobs [--backend aws]`, `info`, `health`, `test-backend`: inspection. A fresh `health` process has no running scheduler and reports "initialized (not started)".

## Cron table

`packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP`: `daily-digest` `0 8 * * *` (EPL), `fetch-espn-fixtures` `0 6 * * *` (EPL), `fetch-mlb-probables` `0 6 * * *` (MLB), `fetch-betfair-exchange` `*/30 * * * *` (every Betfair-supported sport). Defaults for `bootstrap_jobs` are in `SchedulerConfig`; `agent-run` and `daily-digest` are commented out there.

## Hazards

- Cron and self-scheduling share one schedule id for `fetch-betfair-exchange-<sport>`, and both writes use `ConflictPolicy.replace`, so the first `self_schedule` call replaces the cron trigger with a one-shot trigger until the next `odds scheduler start`. Whether the "safety floor" comment in `_JOB_CRON_MAP` still holds after that is not verified.
- `scheduler info` has no help text: its docstring sits after a statement (`packages/odds-cli/odds_cli/commands/scheduler.py::info`).
- With `--db` set and bootstrap off, stale-schedule pruning is also skipped.
- Under `SCHEDULER_DRY_RUN`, `get_scheduled_jobs` returns `[]`, so `list-jobs` looks empty.

## Related

- [Scheduler backends](scheduler-backends.md), [job registry](job-registry.md), [self-scheduling](self-scheduling.md), [CLI overview](cli-overview.md).
