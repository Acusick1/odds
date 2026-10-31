---
type: reference
title: Where each job runs
description: The production host and its job set, the per-job sport and schedule table, the dormant AWS rules, heartbeat writers, and how to check deployed AWS state.
sources:
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_lambda.scheduling.jobs.resolve_target_sports
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/variables.tf
  - scripts/verify-schedules.sh
mentions:
  - odds_core.alerts.job_alert_context
status: stable
---

# Where each job runs

Every scheduled job runs on one host: the production host, under the [local scheduler](local-scheduler.md) managed by `odds-scheduler.service` ([Run the scheduler as a systemd service](howto-run-scheduler-service.md)). Its Postgres is the production database ([Databases and environments](../foundations/databases-and-environments.md#environments)). The AWS Lambdas and their EventBridge rules are dormant. The decision and its trade-offs are in [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md). The registry itself is in [Job registry](job-registry.md).

## Where each job runs

Three lists in code decide what the production scheduler runs:

- `odds_core.config.SchedulerConfig.bootstrap_jobs` names the jobs `odds scheduler start` bootstraps. `agent-run` is commented out there, so it runs only when added to `SCHEDULER_BOOTSTRAP_JOBS`.
- `odds_lambda.scheduling.jobs._JOB_CRON_MAP` gives the jobs installed as a fixed cron. Every other bootstrapped job self-schedules ([Self-scheduling](self-scheduling.md)); a restart keeps its pending run, and only a job without one runs its bootstrap function ([bootstrap](local-scheduler.md#bootstrap-jobs-and-cron-table)).
- `odds_lambda.scheduling.jobs._JOB_SPORTS` limits a per-sport job, cron or self-scheduling, to the configured sports in its allowlist. A per-sport job absent from it runs for every sport in `data_collection.sports` (`odds_lambda.scheduling.jobs.resolve_target_sports`).

| Job | Bootstrapped by default | Sports | Schedule |
|---|---|---|---|
| `fetch-odds` | yes | EPL and NBA (allowlist; Odds API quota, one key each) | self-scheduling by [tier](../storage/fetch-tier.md), 2 h to 48 h for EPL, per slate for NBA ([NBA schedule](../collection/fetch-odds-job.md#nba-schedule)) |
| `fetch-scores` | yes | EPL (allowlist; Odds API quota) | self-scheduling, game activity |
| `update-status`, `check-health` | yes | global | self-scheduling; health every 60 min |
| `fetch-oddsportal` | yes | every configured sport (EPL, MLB, NBA) | proximity (30 min to 2 h), dormant with no fixture in 28 days |
| `fetch-oddsportal-results` | yes | EPL (allowlist) | daily 08:00 UTC |
| `fetch-mlb-results` | yes | MLB (not per sport) | daily 08:00 UTC |
| `fetch-betfair-exchange` | yes | every Betfair-supported sport (EPL, MLB, NBA) | self-scheduling, 15 min to 2 h, plus a run 5 min before each kickoff, dormant with no fixture in 28 days ([Betfair job](../collection/betfair-exchange.md#flow)) |
| `fetch-polymarket-odds` | yes ([ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md)) | every key of `odds_lambda.polymarket_sports.POLYMARKET_SPORTS` (EPL, NBA) that is in `data_collection.sports` | self-scheduling, 5 min to 2 h, dormant with no event in 28 days ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#the-fetch-polymarket-odds-job)) |
| `run-candidates` | yes | every configured sport (NBA has no candidates, so it only settles trades) | self-scheduling, at the next decision time of an active mechanical candidate, at most 30 min apart ([run-candidates job](../evaluation/run-candidates-job.md#next-run)) |
| `fetch-espn-fixtures` | yes | EPL (fixtures table) and NBA (creates and settles events, [NBA schedule](../collection/polymarket-odds-snapshots.md#nba-schedule-from-espn)) | cron `0 6 * * *` |
| `fetch-mlb-probables` | yes | MLB | cron `0 6 * * *` |
| `fetch-injuries` | yes | NBA (allowlist; the only league with the report) | self-scheduling, every 6 h plus runs 2 h and 30 min before each kickoff, dormant with no stored game in 28 days ([NBA injury job](../collection/nba-injuries-and-stats.md#the-fetch-injuries-job)) |
| `agent-run` | no | every configured sport | fixture proximity |

The betting agent is not scheduled: it is run by hand on the production host with `odds agent run` as the restricted `odds_agent` role ([Agent runtime](../agent/agent-run-job.md)). A collection job goes dormant for a sport with no fixture in the next 28 days, and wakes by itself ([dormancy](decision-engine.md#dormancy)).

Per-job pages: [fetch-odds](../collection/fetch-odds-job.md), [fetch-scores, update-status and the results jobs](../collection/results-jobs.md), [status lifecycle](../collection/event-status-lifecycle.md), [OddsPortal](../collection/fetch-oddsportal-job.md), [Betfair](../collection/betfair-exchange.md), [health](../operations/health-monitor.md), [agent](../agent/agent-run-job.md), [ESPN fixtures](../collection/epl-context-data.md), [MLB probables](../collection/mlb-probable-pitchers.md), [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md), [NBA injury reports](../collection/nba-injuries-and-stats.md#the-fetch-injuries-job), [run-candidates](../evaluation/run-candidates-job.md).

Known defects on the production host: OddsPortal scrapes return 0 matches (#484) and one hung OddsPortal run ends the local chains (#485).

## Dormant AWS rules

Terraform still declares the AWS jobs: in `deployment/aws/terraform/eventbridge.tf`, `local.sport_configs` (EPL only; MLB commented out) plus the always-created `update-status` and `check-health`, and in `deployment/aws/terraform/variables.tf` the `betfair_enabled` gate. No Terraform rule exists for `fetch-oddsportal`, `fetch-oddsportal-results` or `fetch-mlb-results`. The rules are disabled in the account, and the Lambdas, ECR images and Terraform state are left in place so the move can be reversed ([AWS infrastructure](../operations/aws-infrastructure.md)).

> [!WARNING]
> The Deploy to Production workflow (`deploy-prod.yml`) is disabled in GitHub; keep it disabled while AWS is dormant (`gh workflow list --all` shows the state). Its post-deploy invocations make every job reschedule itself, which re-enables every rule, so both hosts would collect and the Lambdas would write to the Neon archive ([CI/CD workflows](../operations/ci-cd-workflows.md#prod-deploy)).

No code checks for a job enabled on both hosts, so a re-enabled rule double-fetches alongside the production host.

## Heartbeats

Which jobs write a heartbeat is a property of [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery), not of the registry, and a run that raises writes none. `update-status` writes one only on runs that execute (`odds_lambda.jobs.update_status.main`); skipped runs write none, and a failed booking alerts through `odds_core.alerts.send_scheduling_failure`. Producing a heartbeat is separate from the health check expecting one. The jobs the health check expects a heartbeat from, and how often, are in [Health monitor](../operations/health-monitor.md#heartbeats). All heartbeats land in the production database, which `check-health` reads; nothing outside the production host notices if the host itself stops (#489). Every local run is bounded by `odds_lambda.scheduling.runner.JOB_TIMEOUT` ([Local scheduler](local-scheduler.md#scheduling-primitives)).

## Deployed state

Deployed AWS state cannot be verified from the repo. Only `terraform.tfvars.example` is committed, so the live account may have been applied with different gates. Terraform pre-creates the self-scheduling rules `DISABLED` and ignores `schedule_expression` and `state` on them ([Scheduler backends](scheduler-backends.md#rules-and-terraform)), so a plan shows neither real cadences nor whether a rule was re-enabled.

Query the account with `odds scheduler list-jobs --backend aws` (needs `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` and credentials). Every rule should be listed as cancelled (disabled). Compare it with `odds scheduler list-jobs` on the production host to find a job running in both places.

- `scripts/verify-schedules.sh` fails disabled, placeholder-rate and past-year rules, so it fails every rule while AWS is dormant. #383 tracks the orphans from when the rules ran.
- A Lambda payload for a per-sport job without `sport` raises `ValueError` ([Lambda handler](scheduler-backends.md#lambda-handler)).

[Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) covers the rule, target and permission commands.

## Related

- [Job registry](job-registry.md)
- [Self-scheduling](self-scheduling.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Code conventions](../foundations/code-conventions.md)
