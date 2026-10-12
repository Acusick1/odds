---
type: component
title: Job registry and JobContext
summary: "scheduling/jobs.py: the job map, lazy import, per-sport jobs, compound job names, bootstrap overrides and the JobContext passed to every job."
sources:
  - docs/ARCHITECTURE.md#job-registry
  - docs/DEVELOPMENT.md#adding-scheduler-jobs
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_PER_SPORT_JOBS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::JobContext
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::make_compound_job_name
verified_at: 7887039
---

# Job registry and JobContext

`packages/odds-lambda/odds_lambda/scheduling/jobs.py` is the source of truth for which jobs exist. [Job deployment matrix](job-deployment-matrix.md) says where each can run. Do not confuse it with the `odds_lambda/jobs/` package that holds the job modules.

## Registry

- `_JOB_MODULE_MAP`: job name to `(module path, "main")` for 14 jobs: `fetch-odds`, `fetch-scores`, `update-status`, `check-health`, `fetch-polymarket`, `backfill-polymarket`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`.
- `get_job_function` imports the module on first use and caches it, so the scraper Lambda never imports xgboost-dependent modules unless asked.
- `_JOB_BOOTSTRAP_MAP`: jobs whose local bootstrap uses a different function; only `agent-run` (`schedule_next`). `get_bootstrap_function` wraps it to the `(JobContext)` signature.
- `_SMOKE_UNSAFE_JOBS` (`agent-run`) and `_OUTWARD_POSTING_JOBS` (`daily-digest`) drive [`odds scheduler smoke`](local-scheduler.md).

## Per-sport jobs and compound names

`_PER_SPORT_JOBS` lists jobs that take a sport: `fetch-odds`, `fetch-scores`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`. `update-status`, `check-health` and both Polymarket jobs are global.

`_SPORT_SUFFIX_MAP` knows only `epl` (`soccer_epl`) and `mlb` (`baseball_mlb`). `make_compound_job_name("fetch-odds", "soccer_epl")` returns `fetch-odds-epl`, the EventBridge rule suffix and APScheduler schedule id; `resolve_job_name` reverses it. A sport with no suffix, such as NBA, yields the bare job name.

## JobContext

`JobContext` is the dataclass every `main(ctx)` receives: `sport`, `include_spreads`, `include_totals`, `dry_run`, `lookback_hours` (24), `lookahead_hours` (48). `JobContext.from_payload` keeps known keys and drops unknown ones with a debug log, so a typo in an EventBridge payload is silent. Jobs read dry-run from `settings.scheduler.dry_run`; only `backfill-polymarket` reads `ctx.dry_run`.

## Adding a job

Follow [add a scheduler job](howto-add-scheduler-job.md). The registry also asserts at import that every `_JOB_CRON_MAP` key is a registered job.

## Hazards

- `docs/DEVELOPMENT.md#adding-scheduler-jobs` says to wrap every job in `job_alert_context`; `fetch-odds`, `fetch-scores`, `fetch-polymarket` and `backfill-polymarket` do not, so they write no heartbeat ([alerts module](alerts-module.md)).
- The name collision between the `scheduling` registry module and the `jobs/` package makes "jobs" imports ambiguous; `update_status.py`'s docstring still cites a nonexistent old path, scheduler/jobs.

## Related

- [Self-scheduling](self-scheduling.md), [Lambda handler](lambda-handler.md), [code conventions](code-conventions.md).
