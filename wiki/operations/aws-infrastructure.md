---
type: reference
title: AWS infrastructure and deployment
description: The two Lambda functions and images, Terraform and bootstrap layout and variables, IAM, SSM, logs, secrets, constraints and costs.
sources:
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/variables.tf
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/bootstrap/main.tf
  - deployment/aws/bootstrap/README.md
  - deployment/aws/build_and_push.sh
  - deployment/aws/build_and_push_scraper.sh
  - scripts/lambda_usage.py
  - odds_core.database
status: stable
---

# AWS infrastructure

Everything lives under `deployment/aws/`; region `eu-west-1`. Deployed state is not in the repo, and Terraform does not show the live schedule (see [Terraform](#terraform)); the live view is `odds scheduler list-jobs --backend aws` ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). The manual deploy steps are on [howto-deploy-aws](howto-deploy-aws.md).

## Functions

| | Scheduler | Scraper |
|---|---|---|
| Name | `var.project_name` (prod `odds-scheduler`, default `odds-scheduler-dev`) | `{project_name}-scraper` |
| Image | `Dockerfile.lambda`: `odds-core`, `odds-lambda`, `odds-analytics`, `odds-cli` wheels plus xgboost | `Dockerfile.scraper`: `odds-core`, `odds-lambda`, OddsHarvester fork, Playwright and Chromium |
| Memory / timeout | `lambda_memory_size` 512 MB, `lambda_timeout` 300 s (`deployment/aws/terraform/variables.tf`) | 2048 MB, 900 s (`deployment/aws/terraform/scraper.tf`) |
| Created | always | only if `enable_oddsportal_scraper` |

Both run the same handler and registry ([scheduler-backends](../scheduling/scheduler-backends.md), [lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)). The scraper's 900 s timeout applies to the results job, which has no pre-schedule ([results-jobs](../collection/results-jobs.md)). Log groups `/aws/lambda/<function>` keep 14 days. One IAM role covers both: S3 model read, EventBridge rule writes (restricted to `{rule_prefix}-*`; it lacks `events:ListRules`, so listing and health checks work only from a shell with broader credentials; #443), SSM read/write of `/{project_name}/active-api-key-index` (the [API key rotation](../collection/odds-api-client.md#retry-and-rotation) index, set through `SSM_API_KEY_INDEX`), and, if `betfair_enabled`, SSM read of the Betfair cert/key parameters ([Betfair client](../collection/betfair-exchange.md)).

`scraper.tf` sets only `DATABASE_URL`, API keys, `LAMBDA_ARN`, `RULE_PREFIX` and `LOG_LEVEL`: no `ALERT_ENABLED`, `DISCORD_WEBHOOK_URL`, `MODEL_NAME` or `MODEL_BUCKET`, unlike `lambda.tf`, so scraper alerts are off (#443) and scoring is skipped there (#454). The scraper image has no xgboost or `odds-analytics`, so publishing a model does not enable scoring on AWS ([prediction-scoring](../modelling/prediction-scoring.md)). `odds model publish` defaults to the literal bucket `odds-models-685946576110` ([model-store](../modelling/model-store.md)).

## Terraform

`deployment/aws/terraform/`: `main.tf` (provider), `backend.tf` (S3 backend configured per environment by `backend.dev.hcl` / `backend.prod.hcl`), `variables.tf`, `lambda.tf`, `scraper.tf`, `eventbridge.tf`, `terraform.tfvars.example`. `variables.tf` is authoritative; notable defaults: `aws_region` `eu-west-1`, `rule_prefix` `odds`, `enable_oddsportal_scraper`/`enable_polymarket`/`betfair_enabled` false, `model_name` `epl-clv-home`.

EventBridge rules are named `{rule_prefix}-{job}[-{sport suffix}]` and `eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `ignore_changes` on `schedule_expression` and `state`, so a clean `terraform plan` does not mean the rules are active ([scheduler-backends](../scheduling/scheduler-backends.md)). Production Terraform is applied from `.github/workflows/deploy-prod.yml` with `-var-file=terraform.tfvars.example` plus overrides, so the committed tfvars are not a record of what is deployed ([ci-cd-workflows](ci-cd-workflows.md)).

`deployment/aws/bootstrap/` is applied once: the Terraform state bucket and lock table, the ECR repository, and the shared versioned model bucket (`s3_models.tf`, see [model-store](../modelling/model-store.md)). Its own state file is committed.

## Constraints

- Database connections use `NullPool` automatically when `AWS_LAMBDA_FUNCTION_NAME` is set, whatever the scheduler backend (`odds_core.database`; [databases-and-environments](../foundations/databases-and-environments.md#engine-creation)).
- Stateless between invocations apart from the SSM key-rotation index (`SSM_API_KEY_INDEX` in `deployment/aws/terraform/lambda.tf`).

## Costs and usage

Spend cannot be verified from the repo; check it with `aws ce get-cost-and-usage` or the Billing console. The managed PostgreSQL provider is Neon (`deployment/aws/terraform/terraform.tfvars.example`) and is billed separately; check its console. Log groups keep 14 days (`retention_in_days` in `deployment/aws/terraform/lambda.tf` and `deployment/aws/terraform/scraper.tf`). `scripts/lambda_usage.py` pulls `REPORT` lines from CloudWatch for both functions and prints GB-seconds against the 400,000 free tier, daily breakdown, retry analysis and an hourly heatmap (`--days`, `--start`, `--end`). Gaps are filled with [backfills](../collection/odds-api-backfill.md); alerts go to Discord ([alerts](alerts.md)). `scripts/lambda_usage.py` hard-codes memory sizes (0.5 and 2.0 GB), so a change in Terraform does not propagate (#463).

## Secrets

API keys come from environment variables and Terraform variables (`odds_api_key`, `odds_api_keys`, `database_url` in `deployment/aws/terraform/variables.tf`), passed to the functions as environment variables in `deployment/aws/terraform/lambda.tf` and `deployment/aws/terraform/scraper.tf`; only the key-rotation index lives in SSM.

## Related

- [How to deploy to AWS by hand](howto-deploy-aws.md)
- [Debug the AWS scheduler](../scheduling/howto-debug-aws-scheduler.md)
- [CI/CD workflows](ci-cd-workflows.md)
- [Alerts](alerts.md)
