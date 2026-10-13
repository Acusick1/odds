---
type: component
title: Local scheduler and always-on host
description: The APScheduler-on-Postgres backend, odds scheduler start/run-once/smoke/list-jobs, bootstrap jobs and cron floors, and running it as a systemd service redeployed by deploy.sh.
sources:
  - packages/odds-lambda/odds_lambda/scheduling/backends/local.py::LocalSchedulerBackend
  - packages/odds-cli/odds_cli/commands/scheduler.py::start_local
  - packages/odds-cli/odds_cli/commands/scheduler.py::info
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - deploy.sh
  - odds-scheduler.service
  - packages/odds-cli/odds_cli/commands/scheduler.py::smoke
status: stable
---

# Local scheduler and always-on host

`packages/odds-lambda/odds_lambda/scheduling/backends/local.py::LocalSchedulerBackend` runs APScheduler 4 (pinned `4.0.0a6` in `packages/odds-lambda/pyproject.toml`) with `SQLAlchemyDataStore` and `AsyncpgEventBroker` on the project Postgres, so any process sharing the database can add a schedule and the running scheduler sees it via LISTEN/NOTIFY. It is the default backend (`SCHEDULER_BACKEND=local`) and the only home of jobs that have no EventBridge rule ([Jobs](jobs-and-deployment-matrix.md#where-each-job-runs)). The MCP tools `packages/odds-mcp/odds_mcp/server.py::refresh_scrape` and `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` use this store; `packages/odds-mcp/odds_mcp/server.py::refresh_scrape` queues `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::ingest_league` jobs that wait if no scheduler process is running ([MCP server](../agent/mcp-server.md)).

## Scheduling primitives

- `schedule_next_execution` writes a one-shot `DateTrigger` schedule with `id=job_name` and `ConflictPolicy.replace`.
- `schedule_cron` writes a `CronTrigger` (UTC) with the same id and policy.
- Both reuse a live scheduler (the backend's own, or `current_async_scheduler` while a job runs); otherwise `build_scheduler(role=SchedulerRole.scheduler)` opens a short-lived one that only writes to the store.
- `cancel_scheduled_execution` needs a started scheduler.

## CLI

In `packages/odds-cli/odds_cli/commands/scheduler.py`:

- `start`: exits 1 unless `SCHEDULER_BACKEND=local`. With bootstrap on, it first removes stored schedules not in `expected_compound_job_names(bootstrap_jobs, sports)`, then for each job in `packages/odds-core/odds_core/config.py::SchedulerConfig.bootstrap_jobs` either installs its `_JOB_CRON_MAP` cron (re-applied on every start) or runs its bootstrap function once, skipping jobs that already have a `next_run_time`. `--db NAME` swaps the database name in `DATABASE_URL` and defaults bootstrap off; `--bootstrap` forces it on. Sidecar mode keeps a scheduler on a sandbox database from starting an agent run alongside one you are driving by hand. Invocations: `odds scheduler start` (prod-like, bootstrap on), `odds scheduler start --db odds_test` (dev sidecar, bootstrap off), `odds scheduler start --db odds_test --bootstrap` (dev full-loop smoke test); it runs until Ctrl+C. In-process sessions keep the original engine ([Configuration](../foundations/configuration.md#hazards)).
- `run-once JOB`: runs one registered job, compound names allowed (`fetch-odds-epl`), on any backend. Its help text lists three jobs but any registered name works.
- `packages/odds-cli/odds_cli/commands/scheduler.py::smoke`: runs every bootstrapped job once and prints PASS/FAIL/SKIP; exit code is the failed count (max 255). `agent-run` is skipped unless `--include-agent` (`_SMOKE_UNSAFE_JOBS`); `--no-side-effects` drops `daily-digest`, which posts to Discord. `--only` and `--exclude` take base or compound names and may repeat. Gated jobs only run their body when `decision.should_execute` is true, so smoke otherwise exercises import, config, decision and schema paths.
- `list-jobs [--backend aws]`, `packages/odds-cli/odds_cli/commands/scheduler.py::info`, `packages/odds-cli/odds_cli/commands/scheduler.py::health`, `test-backend`: inspection. A fresh `packages/odds-cli/odds_cli/commands/scheduler.py::health` process has no running scheduler and reports "initialized (not started)".

## Bootstrap jobs and cron table

`packages/odds-core/odds_core/config.py::SchedulerConfig.bootstrap_jobs` defaults to `fetch-oddsportal`, `fetch-oddsportal-results`, `fetch-espn-fixtures`, `fetch-betfair-exchange` and `fetch-mlb-probables`; `agent-run` and `daily-digest` are commented out there, so they are opt-in via `SCHEDULER_BOOTSTRAP_JOBS`. For `agent-run` the bootstrap function is `packages/odds-lambda/odds_lambda/jobs/agent_run.py::schedule_next`, not `packages/odds-lambda/odds_lambda/jobs/agent_run.py::main`.

`packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP`: `daily-digest` `0 8 * * *` (EPL), `fetch-espn-fixtures` `0 6 * * *` (EPL), `fetch-mlb-probables` `0 6 * * *` (MLB), `fetch-betfair-exchange` `*/30 * * * *` (every Betfair-supported sport).

Inline CLV scoring inside `fetch-oddsportal` works only here: the scraper Lambda image lacks the analytics stack ([Publishing and scoring](../modelling/model-publishing-and-scoring.md)). Running the same job on AWS and locally double-scrapes ([OddsPortal](../collection/oddsportal.md)); check `odds scheduler list-jobs --backend aws` first.

## Run as a systemd service

The scheduler can run on a mini-PC under systemd and be redeployed by a script that proves changed jobs run before restarting.

Install the unit:

1. Edit `odds-scheduler.service` (repo root): set `User`, `WorkingDirectory`, `EnvironmentFile` (the repo `.env`) and the absolute `uv` path in `ExecStart` (`uv run odds scheduler start`). systemd needs an absolute executable path; `which uv` shows it.
2. Install and start:
   ```bash
   sudo cp odds-scheduler.service /etc/systemd/system/odds-scheduler.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now odds-scheduler
   ```
   `Restart=always` with `RestartSec=5` restarts it after a crash or reboot.
3. Check: `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

Redeploy by running `./deploy.sh` from the repo root on the host. In order it runs `git pull --ff-only`, `uv sync`, `uv run alembic upgrade head`, `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`, `sudo systemctl restart odds-scheduler`, then status and `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration or smoke fails, so the old code keeps serving.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched.
- `alembic upgrade head` runs before the restart, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, `sudo systemctl stop odds-scheduler` first, migrate, then redeploy ([Create and apply a database migration](../foundations/howto-database-migration.md)).

## Hazards

- Cron and self-scheduling share one schedule id for `fetch-betfair-exchange-<sport>`, and both writes use `ConflictPolicy.replace`, so the first `packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule` call replaces the cron trigger with a one-shot trigger until the next `odds scheduler start`. Whether the "safety floor" comment in `_JOB_CRON_MAP` still holds after that is not verified.
- `scheduler info` has no help text: its docstring sits after a statement (`packages/odds-cli/odds_cli/commands/scheduler.py::info`).
- With `--db` set and bootstrap off, stale-schedule pruning is also skipped.
- `packages/odds-lambda/odds_lambda/scheduling/backends/local.py::LocalSchedulerBackend.get_scheduled_jobs` returns `[]` only for a backend built with `dry_run=True`; `odds scheduler list-jobs` builds it without, so `SCHEDULER_DRY_RUN` does not empty the listing.
- Do not run this host's scheduler and an AWS scraper for the same jobs unless intended.
- `deploy.sh` is not idempotent with local edits: `--ff-only` fails on a diverged checkout.

## Related

- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md)
- [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md)
- [Self-scheduling and the decision engine](self-scheduling.md)
- [Packages and the odds CLI](../foundations/packages-and-cli.md)
