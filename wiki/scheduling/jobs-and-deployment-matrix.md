---
type: component
title: "Jobs: registry, context and where each runs"
description: The 14 registered jobs, per-sport compound names and JobContext, and for each job where it can run (scheduler Lambda, scraper Lambda, local default or opt-in) and on what schedule, plus how to check the deployed state.
sources:
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_BOOTSTRAP_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_PER_SPORT_JOBS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::JobContext
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::make_compound_job_name
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::resolve_job_name
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-core/odds_core/config.py::AlertConfig
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Jobs: registry, context and where each runs

`packages/odds-lambda/odds_lambda/scheduling/jobs.py` is the source of truth for which jobs exist. Do not confuse it with the `odds_lambda/jobs/` package that holds the job modules, which makes "jobs" imports ambiguous (`packages/odds-lambda/odds_lambda/jobs/update_status.py`'s docstring still cites a nonexistent old path, scheduler/jobs). Jobs run on two targets: two AWS Lambdas driven by EventBridge rules, and the [local scheduler](local-scheduler.md). Where a job *can* run is fixed by the gates below; where it *does* run is only knowable from the AWS account.

## Registry

- `_JOB_MODULE_MAP`: job name to `(module path, "main")` for 14 jobs: `fetch-odds`, `fetch-scores`, `update-status`, `check-health`, `fetch-polymarket`, `backfill-polymarket`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`.
- `packages/odds-lambda/odds_lambda/scheduling/jobs.py::get_job_function` imports the module on first use and caches it, so the scraper Lambda never imports xgboost-dependent modules unless asked.
- `_JOB_BOOTSTRAP_MAP`: jobs whose local bootstrap uses a different function; only `agent-run` (`packages/odds-lambda/odds_lambda/jobs/agent_run.py::schedule_next`). `packages/odds-lambda/odds_lambda/scheduling/jobs.py::get_bootstrap_function` wraps it to the `(JobContext)` signature. As a result `odds scheduler start` never launches the agent run itself; it only books its next wake-up. `_SMOKE_UNSAFE_JOBS` (`agent-run`) and `_OUTWARD_POSTING_JOBS` (`daily-digest`) drive [`odds scheduler smoke`](local-scheduler.md#cli); smoke skips `agent-run` unless `--include-agent`.
- At import the registry asserts every `_JOB_CRON_MAP` key is a registered job.

### Per-sport jobs and compound names

`_PER_SPORT_JOBS` lists jobs that take a sport: `fetch-odds`, `fetch-scores`, `fetch-oddsportal`, `fetch-oddsportal-results`, `score-predictions`, `daily-digest`, `agent-run`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`. `update-status`, `check-health` and both Polymarket jobs are global.

`_SPORT_SUFFIX_MAP` knows only `epl` (`soccer_epl`) and `mlb` (`baseball_mlb`). `make_compound_job_name("fetch-odds", "soccer_epl")` returns `fetch-odds-epl`, the EventBridge rule suffix and APScheduler schedule id; `packages/odds-lambda/odds_lambda/scheduling/jobs.py::resolve_job_name` reverses it. A sport with no suffix, such as NBA, yields the bare job name. The compound name must equal the rule name, or a reschedule goes nowhere ([Add a scheduler job](howto-add-scheduler-job.md)). Adding a sport touches several tables, listed in [Sources and keys](../collection/sources-and-keys.md).

### JobContext

`packages/odds-lambda/odds_lambda/scheduling/jobs.py::JobContext` is the dataclass every `main(ctx)` receives: `sport`, `include_spreads`, `include_totals`, `dry_run`, `lookback_hours` (24), `lookahead_hours` (48). `packages/odds-lambda/odds_lambda/scheduling/jobs.py::JobContext.from_payload` keeps known keys and drops unknown ones with a debug log, so a typo in an EventBridge payload is silent. Jobs read dry-run from `settings.scheduler.dry_run`; only `backfill-polymarket` reads `ctx.dry_run`.

## Where each job runs

Gates:

- **AWS**: in `deployment/aws/terraform/eventbridge.tf`, `local.sport_configs` (EPL only; MLB commented out), `enable_oddsportal_scraper`, `enable_polymarket` and `betfair_enabled`, plus the always-created `update-status` and `check-health`. These are Terraform variable defaults; see [AWS infrastructure](../operations/aws-infrastructure.md).
- **Local**: `packages/odds-core/odds_core/config.py::SchedulerConfig.bootstrap_jobs` (default `fetch-oddsportal`, `fetch-oddsportal-results`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`; `agent-run` and `daily-digest` are commented out, so opt-in) and `_JOB_CRON_MAP`.

| Job | AWS | Local default | Schedule |
|---|---|---|---|
| `fetch-odds` | scheduler Lambda, per sport | no | self-scheduling by [tier](../storage/fetch-tier.md), 30 min to 48 h |
| `fetch-scores` | scheduler Lambda, per sport | no | self-scheduling, game activity |
| `update-status`, `check-health` | scheduler Lambda, global | no | self-scheduling; health every 60 min |
| `fetch-polymarket` | scheduler Lambda if `enable_polymarket` | no | self-scheduling |
| `backfill-polymarket` | scheduler Lambda if `enable_polymarket` | no | fixed `rate(3 days)` |
| `fetch-betfair-exchange` | scheduler Lambda if `betfair_enabled` | yes, cron `*/30 * * * *` | self-scheduling / cron floor |
| `fetch-oddsportal`, `fetch-oddsportal-results` | scraper Lambda if `enable_oddsportal_scraper` | yes | proximity (30 min to 2 h) / daily 08:00 UTC |
| `fetch-espn-fixtures`, `fetch-mlb-probables` | never | yes, cron `0 6 * * *` | cron |
| `daily-digest` | never | opt-in, cron `0 8 * * *` | cron, EPL only via the allowlist; posts to Discord |
| `agent-run` | never | opt-in | fixture proximity |
| `score-predictions` | never | inline in `fetch-oddsportal` | n/a |

The agent, digest, ESPN and MLB jobs have no rule in `scheduler_rules_map` or `scraper_rules_map` and are always local. `score-predictions` is inline in `fetch-oddsportal` and needs odds-analytics and xgboost, which the scraper image lacks, so it is effectively local-only ([Publishing and scoring](../modelling/model-publishing-and-scoring.md)).

Per-job pages: [fetch-odds](../collection/odds-api-ingestion.md), [fetch-scores, update-status, results](../collection/event-lifecycle.md), [OddsPortal](../collection/oddsportal.md), [Betfair](../collection/betfair-exchange.md), [digest](../operations/reporting-and-daily-digest.md), [health](../operations/alerts-and-health.md), [agent](../agent/betting-agent.md), [ESPN fixtures](../collection/epl-context-data.md), [MLB probables](../collection/mlb-probable-pitchers.md), [Polymarket](../collection/polymarket.md).

### Heartbeats

Which jobs write a heartbeat is a property of `packages/odds-core/odds_core/alerts.py::job_alert_context`, not of the registry. `packages/odds-core/odds_core/config.py::AlertConfig.heartbeat_expectations` (`packages/odds-core/odds_core/config.py::AlertConfig`) expects `fetch-oddsportal-<sport>` every 2 h, `fetch-oddsportal-results-<sport>` every 26 h (EPL and MLB), `daily-digest-epl` every 26 h and `check-health` every 2 h. `fetch-odds`, `fetch-scores`, `update-status` and both Polymarket jobs have none ([Alerts and health](../operations/alerts-and-health.md#heartbeats)).

## Deployed state is not in the repo

Only `terraform.tfvars.example` is committed. The live account may have been applied with different gates, and Terraform pre-creates the self-scheduling rules `DISABLED` and ignores `schedule_expression` and `state` on them ([Scheduler backends](scheduler-backends.md#aws-eventbridge-backend)), so a plan never shows real cadences. Query the account: `odds scheduler list-jobs --backend aws` (needs `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` and credentials). Look for jobs running in both places (scraper enabled on AWS and bootstrapped locally; both would scrape) and for enabled rules whose next-run time is in the past ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)).

A Lambda payload without `sport` runs every configured sport and reschedules under the bare name, which matches no rule ([Scheduler backends](scheduler-backends.md#lambda-handler)).

## Hazards

- **Stale rules.** An enabled rule whose cron year or time is in the past never fires again, and a rule still on the `rate(1 day)` placeholder or `DISABLED` was never self-scheduled; both mean the chain is dead. Check with `odds scheduler list-jobs --backend aws` (look for a past or empty next-run time) or `scripts/verify-schedules.sh`, which fails disabled, placeholder-rate and past-year rules.
- **A job running in both places.** The deploy-time gates (AWS) and `packages/odds-core/odds_core/config.py::SchedulerConfig.bootstrap_jobs` (local) are independent, so a job such as `fetch-oddsportal` or `fetch-betfair-exchange` can run on AWS and on a local scheduler at once and double-fetch. Compare `odds scheduler list-jobs --backend aws` with `odds scheduler list-jobs` on the local host; no code checks for it.
- Heartbeats come from `packages/odds-core/odds_core/alerts.py::job_alert_context`; `fetch-odds`, `fetch-scores`, `fetch-polymarket` and `backfill-polymarket` do not, so they write no heartbeat.
- `fixed_schedule_expressions` in Terraform is empty, so no Lambda job has a fixed cron except the Polymarket backfill rule (`{rule_prefix}-backfill-polymarket-bootstrap`).
- The Betfair cron floor and its self-scheduling share one APScheduler id, so the first self-schedule replaces the cron ([Local scheduler](local-scheduler.md#hazards)).
- The `fetch-mlb-probables` comment in `_JOB_CRON_MAP` says the MCP tool "writes through every call"; the job is the only writer ([MLB probable pitchers](../collection/mlb-probable-pitchers.md)).

## Related

- [Self-scheduling and the decision engine](self-scheduling.md)
- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Conventions](../foundations/conventions.md)
