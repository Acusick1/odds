---
type: concept
title: Job deployment matrix
summary: Where each job can run (scheduler Lambda, scraper Lambda, local) and on what schedule, which gates decide it, and why the repo does not record the deployed state.
sources:
  - docs/DEPLOYMENT.md#deployment-targets
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - deployment/aws/terraform/eventbridge.tf
verified_at: 7887039
---

# Job deployment matrix

Jobs run on two targets: two AWS Lambdas driven by EventBridge rules, and the [local scheduler](local-scheduler.md). Where a job *can* run is fixed by three inputs; where it *does* run is only knowable from the AWS account.

## Gates

- **AWS**: in `deployment/aws/terraform/eventbridge.tf`, `local.sport_configs` (EPL only; MLB commented out), `enable_oddsportal_scraper`, `enable_polymarket` and `betfair_enabled`, plus the always-created `update-status` and `check-health`.
- **Local**: `SchedulerConfig.bootstrap_jobs` (default `fetch-oddsportal`, `fetch-oddsportal-results`, `fetch-espn-fixtures`, `fetch-betfair-exchange`, `fetch-mlb-probables`; `agent-run` and `daily-digest` are opt-in) and `_JOB_CRON_MAP`.

## Matrix (design)

| Job | AWS | Local default | Schedule |
|---|---|---|---|
| `fetch-odds` | scheduler Lambda, per sport | no | self-scheduling by [tier](fetch-tier.md), 30 min to 48 h |
| `fetch-scores` | scheduler Lambda, per sport | no | self-scheduling, game activity |
| `update-status`, `check-health` | scheduler Lambda, global | no | self-scheduling; health every 60 min |
| `fetch-polymarket` | scheduler Lambda if `enable_polymarket` | no | self-scheduling |
| `backfill-polymarket` | scheduler Lambda if `enable_polymarket` | no | fixed `rate(3 days)` |
| `fetch-betfair-exchange` | scheduler Lambda if `betfair_enabled` | yes, cron `*/30 * * * *` | self-scheduling / cron floor |
| `fetch-oddsportal`, `fetch-oddsportal-results` | scraper Lambda if `enable_oddsportal_scraper` | yes | proximity (30 min to 2 h) / daily 08:00 UTC |
| `fetch-espn-fixtures`, `fetch-mlb-probables` | never | yes, cron `0 6 * * *` | cron |
| `daily-digest` | never | opt-in, cron `0 8 * * *` | cron, EPL |
| `agent-run` | never | opt-in | fixture proximity |
| `score-predictions` | never | inline in `fetch-oddsportal` | n/a |

Jobs: [fetch-odds](job-fetch-odds.md), [fetch-scores](job-fetch-scores.md), [update-status](job-update-status.md), [fetch-oddsportal](job-fetch-oddsportal.md), [results](job-fetch-oddsportal-results.md), [Betfair](job-fetch-betfair-exchange.md), [scoring](job-score-predictions.md), [digest](job-daily-digest.md), [health](health-monitor.md), [agent](job-agent-run.md), [ESPN fixtures](epl-context-data.md), [MLB probables](mlb-probable-pitchers.md), [Polymarket](polymarket-pipeline.md). The agent, digest, ESPN and MLB jobs have no rule in `scheduler_rules_map` or `scraper_rules_map` and are always local.

## Deployed state is not in the repo

Only `terraform.tfvars.example` is committed. The live account may have been applied with different gates, and Terraform ignores `schedule_expression` and `state` on self-scheduling rules ([AWS EventBridge backend](aws-eventbridge-backend.md)), so a plan never shows real cadences. Query the account: `odds scheduler list-jobs --backend aws` (needs `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX` and credentials). Look for jobs running in both places (scraper enabled on AWS and bootstrapped locally) and for enabled rules whose next-run time is in the past.

## Hazards

- `fixed_schedule_expressions` in Terraform is empty, so no Lambda job has a fixed cron except the Polymarket backfill rule (`{rule_prefix}-backfill-polymarket-bootstrap`).
- Scraper-hosted scoring cannot run on the scraper image ([fetch-oddsportal](job-fetch-oddsportal.md)), so `score-predictions` is effectively local-only.

## Related

- [Scheduler backends](scheduler-backends.md), [AWS infrastructure](aws-infrastructure.md), [job registry](job-registry.md).
