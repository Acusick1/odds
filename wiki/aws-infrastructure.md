---
type: component
title: AWS infrastructure (Lambdas and Terraform)
summary: The two Lambda functions and their images, Terraform layout, variables and bootstrap, plus SSM, CloudWatch, constraints, costs and usage analysis.
sources:
  - docs/DEPLOYMENT.md#aws-lambda
  - docs/DEPLOYMENT.md#functions
  - docs/DEPLOYMENT.md#constraints
  - docs/DEPLOYMENT.md#terraform-layout
  - docs/DEPLOYMENT.md#terraform-variables
  - docs/DEPLOYMENT.md#aws
  - docs/DEPLOYMENT.md#operations
  - docs/DEBUGGING_SCHEDULER.md#where-to-look
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/variables.tf
  - deployment/aws/bootstrap/main.tf
  - scripts/lambda_usage.py
verified_at: 7887039
---

# AWS infrastructure (Lambdas and Terraform)

Everything lives under `deployment/aws/`; region `eu-west-1`. Deployed state is not in the repo (see [job deployment matrix](job-deployment-matrix.md)); the live view is `odds scheduler list-jobs --backend aws`.

## Functions

| | Scheduler | Scraper |
|---|---|---|
| Name | `var.project_name` (prod `odds-scheduler`, default `odds-scheduler-dev`) | `{project_name}-scraper` |
| Image | `Dockerfile.lambda`: `odds-core`, `odds-lambda`, `odds-analytics`, `odds-cli` wheels plus xgboost | `Dockerfile.scraper`: `odds-core`, `odds-lambda`, OddsHarvester fork, Playwright and Chromium |
| Memory / timeout | `lambda_memory_size` 512 MB, `lambda_timeout` 300 s | 2048 MB, 900 s (hard-coded in `scraper.tf`) |
| Created | always | only if `enable_oddsportal_scraper` |

Both run [the same handler](lambda-handler.md) and registry. Log groups `/aws/lambda/<function>` keep 14 days. One IAM role covers both: S3 model read, EventBridge rule writes (restricted to `{rule_prefix}-*`), SSM read/write of `/{project_name}/active-api-key-index` (the [API key rotation](odds-api-client.md) index), and, if `betfair_enabled`, SSM read of the Betfair cert/key parameters ([Betfair client](betfair-exchange-client.md)).

## Terraform

`deployment/aws/terraform/`: `main.tf` (provider), `backend.tf` (S3 backend configured per environment by `backend.dev.hcl` / `backend.prod.hcl`), `variables.tf`, `lambda.tf`, `scraper.tf`, `eventbridge.tf` ([rules](aws-eventbridge-backend.md)), `terraform.tfvars.example`. `variables.tf` is authoritative; notable defaults: `aws_region` `eu-west-1`, `rule_prefix` `odds`, `enable_oddsportal_scraper`/`enable_polymarket`/`betfair_enabled` false, `model_name` `epl-clv-home`.

`deployment/aws/bootstrap/` is applied once: the Terraform state bucket and lock table, the ECR repository, and the shared versioned model bucket (`s3_models.tf`, see [S3 model store](s3-model-store.md)). Its own state file is committed. Deploy steps: [deploy to AWS](howto-deploy-aws.md).

## Constraints

- Database connections use `NullPool` automatically when `AWS_LAMBDA_FUNCTION_NAME` is set, whatever the scheduler backend ([database connections](database-connections.md)).
- Stateless between invocations apart from the SSM index.
- EventBridge limit of 300 rules per region (AWS limit as stated in `docs/DEPLOYMENT.md#constraints`).

## Costs and usage

`docs/DEPLOYMENT.md#aws` estimates ~$0.20/month for the scheduler Lambda, under $0.01 for S3, under $0.10 for logs, free SSM (estimates, not measured here). `scripts/lambda_usage.py` pulls `REPORT` lines from CloudWatch for both functions and prints GB-seconds against the 400,000 free tier, daily breakdown, retry analysis and an hourly heatmap (`--days`, `--start`, `--end`). Backups are the managed database provider's; gaps are filled with [backfills](howto-historical-backfill.md); alerts go to Discord ([alerts module](alerts-module.md)).

## Hazards

- `scraper.tf` sets only `DATABASE_URL`, API keys, `LAMBDA_ARN`, `RULE_PREFIX` and `LOG_LEVEL`: no `ALERT_ENABLED`, `DISCORD_WEBHOOK_URL`, `MODEL_NAME` or `MODEL_BUCKET`, unlike `lambda.tf`.
- `scripts/lambda_usage.py` hard-codes memory sizes (0.5 and 2.0 GB); a change in Terraform does not propagate.
- Production Terraform is applied from a workflow with `-var-file=terraform.tfvars.example` plus overrides ([CI/CD workflows](ci-cd-workflows.md)).

## Related

- [Debug the AWS scheduler](howto-debug-aws-scheduler.md), [configuration](configuration.md).
