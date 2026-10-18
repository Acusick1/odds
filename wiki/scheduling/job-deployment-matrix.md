---
type: reference
title: Where each job runs
description: The production host and its job set, the per-job sport and schedule table, the dormant AWS rules, heartbeat writers, and how to check deployed AWS state.
sources:
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.alerts.job_alert_context
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_lambda.scheduling.jobs.resolve_target_sports
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/variables.tf
  - scripts/verify-schedules.sh
status: stable
---

# Where each job runs

Every scheduled job runs on one host: the mini-PC, a WSL2 guest on a Windows machine (Tailscale host `nucboxg10pro`), under the [local scheduler](local-scheduler.md) managed by `odds-scheduler.service` ([Run the scheduler as a systemd service](howto-run-scheduler-service.md)). Its Postgres is the production database ([Databases and environments](../foundations/databases-and-environments.md#environments)). The AWS Lambdas and their EventBridge rules are dormant. The decision and its trade-offs are in [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md). The registry itself is in [Job registry](job-registry.md).

## Where each job runs

Three lists in code decide what the production scheduler runs:

- `odds_core.config.SchedulerConfig.bootstrap_jobs` names the jobs `odds scheduler start` bootstraps. `agent-run` and `daily-digest` are commented out there, so they run only when added to `SCHEDULER_BOOTSTRAP_JOBS`.
- `odds_lambda.scheduling.jobs._JOB_CRON_MAP` gives the jobs installed as a fixed cron (`fetch-polymarket-odds` among them, replaced by its first self-schedule). Every other bootstrapped job runs its bootstrap function once and then self-schedules ([Self-scheduling](self-scheduling.md)).
- `odds_lambda.scheduling.jobs._JOB_SPORTS` limits a per-sport job, cron or self-scheduling, to the configured sports in its allowlist. A per-sport job absent from it runs for every sport in `data_collection.sports` (`odds_lambda.scheduling.jobs.resolve_target_sports`).

| Job | Bootstrapped by default | Sports | Schedule |
|---|---|---|---|
| `fetch-odds` | yes | EPL (allowlist; Odds API quota) | self-scheduling by [tier](../storage/fetch-tier.md), 30 min to 48 h |
| `fetch-scores` | yes | EPL (allowlist; Odds API quota) | self-scheduling, game activity |
| `update-status`, `check-health` | yes | global | self-scheduling; health every 60 min |
| `fetch-oddsportal`, `fetch-oddsportal-results` | yes | every configured sport | proximity (30 min to 2 h) / daily 08:00 UTC |
| `fetch-betfair-exchange` | yes | every Betfair-supported sport | cron `*/30 * * * *` until the first self-schedule replaces it ([cron table](local-scheduler.md#bootstrap-jobs-and-cron-table)) |
| `fetch-polymarket-odds` | no, pending ADR-0013 | every sport with three-way Polymarket match markets (EPL) | cron `*/15 * * * *` until the first self-schedule replaces it ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#the-fetch-polymarket-odds-job)) |
| `fetch-espn-fixtures` | yes | EPL | cron `0 6 * * *` |
| `fetch-mlb-probables` | yes | MLB | cron `0 6 * * *` |
| `daily-digest` | no | EPL | cron `0 8 * * *`; posts to Discord |
| `agent-run` | no | every configured sport | fixture proximity |
| `score-predictions` | inline in `fetch-oddsportal` | per sport with a model | n/a ([Prediction scoring](../modelling/prediction-scoring.md)) |
| `fetch-polymarket`, `backfill-polymarket` | no | global | self-scheduling / AWS `rate(3 days)` |

The betting agent is not scheduled: it is run by hand on the mini-PC with `odds agent run` as the restricted `odds_agent` role ([Agent runtime](../agent/agent-run-job.md)). Collection jobs keep running a sport through its off-season (#487).

Per-job pages: [fetch-odds](../collection/fetch-odds-job.md), [fetch-scores and update-status](../collection/results-jobs.md), [status lifecycle](../collection/event-status-lifecycle.md), [OddsPortal](../collection/fetch-oddsportal-job.md), [Betfair](../collection/betfair-exchange.md), [digest](../operations/daily-digest.md), [health](../operations/health-monitor.md), [agent](../agent/agent-run-job.md), [ESPN fixtures](../collection/epl-context-data.md), [MLB probables](../collection/mlb-probable-pitchers.md), [Polymarket](../collection/polymarket.md), [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md).

Known defects on the production host: OddsPortal scrapes return 0 matches (#484), one hung OddsPortal run ends the local chains (#485), and the ESPN fixtures request returns HTTP 400 (#486).

## Dormant AWS rules

Terraform still declares the AWS jobs: in `deployment/aws/terraform/eventbridge.tf`, `local.sport_configs` (EPL only; MLB commented out) plus the always-created `update-status` and `check-health`, and in `deployment/aws/terraform/variables.tf` the `enable_oddsportal_scraper`, `enable_polymarket` and `betfair_enabled` gates. The rules are disabled in the account, and the Lambdas, ECR images and Terraform state are left in place so the move can be reversed ([AWS infrastructure](../operations/aws-infrastructure.md)).

> [!WARNING]
> The Deploy to Production workflow (`deploy-prod.yml`) is disabled in GitHub; keep it disabled while AWS is dormant (`gh workflow list --all` shows the state). Its post-deploy invocations make every job reschedule itself, which re-enables every rule, so both hosts would collect and the Lambdas would write to the Neon archive ([CI/CD workflows](../operations/ci-cd-workflows.md#prod-deploy)).

No code checks for a job enabled on both hosts, so a re-enabled rule double-fetches alongside the mini-PC.

## Heartbeats

Which jobs write a heartbeat is a property of `odds_core.alerts.job_alert_context`, not of the registry. `fetch-odds`, `fetch-scores` and both Polymarket jobs write none. `update-status` writes one only on runs that execute (`odds_lambda.jobs.update_status.main`); skipped runs write none. Producing a heartbeat is separate from the health check expecting one. The jobs the health check expects a heartbeat from, and how often, are in [Health monitor](../operations/health-monitor.md#heartbeats). All heartbeats land in the production database, which `check-health` reads; nothing outside the mini-PC notices if the host itself stops (#489).

## Deployed state

Deployed AWS state cannot be verified from the repo. Only `terraform.tfvars.example` is committed, so the live account may have been applied with different gates. Terraform pre-creates the self-scheduling rules `DISABLED` and ignores `schedule_expression` and `state` on them ([Scheduler backends](scheduler-backends.md#rules-and-terraform)), so a plan shows neither real cadences nor whether a rule was re-enabled.

Query the account with `odds scheduler list-jobs --backend aws` (needs `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` and credentials). Every rule should be listed as cancelled (disabled). Compare it with `odds scheduler list-jobs` on the mini-PC to find a job running in both places.

- `scripts/verify-schedules.sh` fails disabled, placeholder-rate and past-year rules, so it fails every rule while AWS is dormant. #383 tracks the orphans from when the rules ran.
- A Lambda payload without `sport` runs every configured sport and reschedules under the bare name, which matches no rule ([Lambda handler](scheduler-backends.md#lambda-handler)).

[Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) covers the rule, target and permission commands.

## Related

- [Job registry](job-registry.md)
- [Self-scheduling](self-scheduling.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Code conventions](../foundations/code-conventions.md)
