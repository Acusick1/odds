---
type: reference
title: Where each job runs
description: The AWS and local gates, the per-job placement and schedule table, heartbeat expectations per job, and how to check deployed state.
sources:
  - odds_core.config.SchedulerConfig
  - odds_core.config.AlertConfig
  - odds_core.alerts.job_alert_context
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/variables.tf
  - scripts/verify-schedules.sh
status: stable
---

# Where each job runs

Jobs run on two targets: two AWS Lambdas driven by EventBridge rules, and the [local scheduler](local-scheduler.md). Where a job can run is fixed by the gates below; where it does run is only knowable from the AWS account ([Deployed state](#deployed-state)). The registry itself is in [Job registry](job-registry.md).

## Where each job runs

Gates:

- **AWS**: in `deployment/aws/terraform/eventbridge.tf`, `local.sport_configs` (EPL only; MLB commented out), plus the always-created `update-status` and `check-health`; and in `deployment/aws/terraform/variables.tf` the `enable_oddsportal_scraper`, `enable_polymarket` and `betfair_enabled` variables (each defaults to false). See [AWS infrastructure](../operations/aws-infrastructure.md).
- **Local**: `odds_core.config.SchedulerConfig.bootstrap_jobs` (default `fetch-oddsportal`, `fetch-oddsportal-results`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`; `agent-run` and `daily-digest` are commented out, so opt-in) and `_JOB_CRON_MAP`.

The two sets of gates are independent, and no code checks for a job enabled in both places, so a job such as `fetch-oddsportal` or `fetch-betfair-exchange` can run on AWS and on a local scheduler at once and double-fetch.

| Job | AWS | Local default | Schedule |
|---|---|---|---|
| `fetch-odds` | scheduler Lambda, per sport | no | self-scheduling by [tier](../storage/fetch-tier.md), 30 min to 48 h |
| `fetch-scores` | scheduler Lambda, per sport | no | self-scheduling, game activity |
| `update-status`, `check-health` | scheduler Lambda, global | no | self-scheduling; health every 60 min |
| `fetch-polymarket` | scheduler Lambda if `enable_polymarket` | no | self-scheduling |
| `backfill-polymarket` | scheduler Lambda if `enable_polymarket` | no | fixed `rate(3 days)`; the rule is named `{rule_prefix}-backfill-polymarket-bootstrap` |
| `fetch-betfair-exchange` | scheduler Lambda if `betfair_enabled` | yes, cron `*/30 * * * *` | self-scheduling / cron floor ([replaced by the first self-schedule](local-scheduler.md#bootstrap-jobs-and-cron-table)) |
| `fetch-oddsportal`, `fetch-oddsportal-results` | scraper Lambda if `enable_oddsportal_scraper` | yes | proximity (30 min to 2 h) / daily 08:00 UTC |
| `fetch-espn-fixtures`, `fetch-mlb-probables` | never | yes, cron `0 6 * * *` | cron |
| `daily-digest` | never | opt-in, cron `0 8 * * *` | cron, EPL only via the allowlist; posts to Discord |
| `agent-run` | never | opt-in | fixture proximity |
| `score-predictions` | never | inline in `fetch-oddsportal` | n/a |

`fixed_schedule_expressions` in Terraform is empty, so the only Lambda job with a fixed schedule is the Polymarket backfill rule.

The agent, digest, ESPN and MLB jobs have no rule in `scheduler_rules_map` or `scraper_rules_map` and are always local. `score-predictions` is inline in `fetch-oddsportal` and needs odds-analytics and xgboost, which the scraper image lacks, so it is effectively local-only ([Prediction scoring](../modelling/prediction-scoring.md)).

Per-job pages: [fetch-odds](../collection/fetch-odds-job.md), [fetch-scores and update-status](../collection/results-jobs.md), [status lifecycle](../collection/event-status-lifecycle.md), [OddsPortal](../collection/fetch-oddsportal-job.md), [Betfair](../collection/betfair-exchange.md), [digest](../operations/daily-digest.md), [health](../operations/health-monitor.md), [agent](../agent/agent-run-job.md), [ESPN fixtures](../collection/epl-context-data.md), [MLB probables](../collection/mlb-probable-pitchers.md), [Polymarket](../collection/polymarket.md).

## Heartbeats

Which jobs write a heartbeat is a property of `odds_core.alerts.job_alert_context`, not of the registry. `fetch-odds`, `fetch-scores`, `update-status` and both Polymarket jobs write none. The jobs the health check expects a heartbeat from, and how often, are in [Health monitor](../operations/health-monitor.md#heartbeats).

## Deployed state

Deployed AWS state cannot be verified from the repo. Only `terraform.tfvars.example` is committed, so the live account may have been applied with different gates. Terraform pre-creates the self-scheduling rules `DISABLED` and ignores `schedule_expression` and `state` on them ([Scheduler backends](scheduler-backends.md#rules-and-terraform)), so a plan never shows real cadences.

Query the account with `odds scheduler list-jobs --backend aws` (needs `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` and credentials). Compare it with `odds scheduler list-jobs` on the local host to find jobs running in both places.

- An enabled rule whose cron year or time is in the past never fires again, and a rule still on the `rate(1 day)` placeholder or `DISABLED` was never self-scheduled; both mean the chain is dead. `scripts/verify-schedules.sh` fails disabled, placeholder-rate and past-year rules. #383 tracks the current orphans.
- A chain dies when EventBridge did not trigger, when a Lambda error stopped the job before it booked the next run, or when a rule was edited by hand. The post-deploy invocations in `deploy-prod.yml` re-invoke jobs and re-activate chains ([AWS infrastructure](../operations/aws-infrastructure.md)).
- A Lambda payload without `sport` runs every configured sport and reschedules under the bare name, which matches no rule ([Lambda handler](scheduler-backends.md#lambda-handler)).

[Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) covers the checks step by step, including the rule, target and permission commands.

## Related

- [Job registry](job-registry.md)
- [Self-scheduling](self-scheduling.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Code conventions](../foundations/code-conventions.md)
