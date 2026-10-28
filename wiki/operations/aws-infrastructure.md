---
type: reference
title: AWS infrastructure and deployment
description: The dormant AWS stack — the Lambda function and image, Terraform and bootstrap layout and variables, IAM, SSM, logs, secrets, constraints and costs.
sources:
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/variables.tf
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/bootstrap/main.tf
  - deployment/aws/bootstrap/README.md
  - deployment/aws/build_and_push.sh
  - scripts/lambda_usage.py
  - odds_core.database
status: stable
---

# AWS infrastructure

Everything lives under `deployment/aws/`; region `eu-west-1`. The stack is dormant: its EventBridge rules are disabled and production runs on a single local host, while the Lambdas, ECR images, Terraform state and the S3 model bucket (`deployment/aws/bootstrap/s3_models.tf`) stay in place ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md), [Where each job runs](../scheduling/job-deployment-matrix.md#dormant-aws-rules)). Deployed state is not in the repo, and Terraform does not show the live schedule (see [Terraform](#terraform)); the live view is `odds scheduler list-jobs --backend aws` ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). The manual deploy steps are on [howto-deploy-aws](howto-deploy-aws.md).

## Function

The scheduler function is named `var.project_name` (prod `odds-scheduler`, default `odds-scheduler-dev`). Its image is `Dockerfile.lambda`: `odds-core`, `odds-lambda`, `odds-analytics`, `odds-cli` wheels and their declared dependencies, XGBoost included. Memory is `lambda_memory_size` (512 MB) and timeout `lambda_timeout` (300 s) (`deployment/aws/terraform/variables.tf`). No Lambda runs `fetch-oddsportal` or `fetch-oddsportal-results`; they run on the local scheduler.

The handler and registry are described on [scheduler-backends](../scheduling/scheduler-backends.md#lambda-handler). The log group `/aws/lambda/<function>` keeps 14 days. The IAM role grants: EventBridge rule writes (restricted to `{rule_prefix}-*`; it lacks `events:ListRules`, so listing and health checks work only from a shell with broader credentials; #443), SSM read/write of `/{project_name}/active-api-key-index` (the [API key rotation](../collection/odds-api-client.md#retry-and-rotation) index, set through `SSM_API_KEY_INDEX`), and, if `betfair_enabled`, SSM read of the Betfair cert/key parameters ([Betfair client](../collection/betfair-exchange.md)).

## Terraform

`deployment/aws/terraform/`: `main.tf` (provider), `backend.tf` (S3 backend configured per environment by `backend.dev.hcl` / `backend.prod.hcl`), `variables.tf`, `lambda.tf`, `eventbridge.tf`, `terraform.tfvars.example`. `variables.tf` is authoritative; notable defaults: `aws_region` `eu-west-1`, `rule_prefix` `odds`, `betfair_enabled` false.

EventBridge rules are named `{rule_prefix}-{job}[-{sport suffix}]` and `eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `ignore_changes` on `schedule_expression` and `state`, so a clean `terraform plan` does not mean the rules are active ([scheduler-backends](../scheduling/scheduler-backends.md)). Production Terraform is applied from `.github/workflows/deploy-prod.yml` with `-var-file=terraform.tfvars.example` plus overrides, so the committed tfvars are not a record of what is deployed ([ci-cd-workflows](ci-cd-workflows.md)).

`deployment/aws/bootstrap/` is applied once: the Terraform state bucket and lock table, the ECR repository, and the shared versioned model bucket (`s3_models.tf`). Its own state file is committed.

## Constraints

- Database connections use `NullPool` automatically when `AWS_LAMBDA_FUNCTION_NAME` is set, whatever the scheduler backend (`odds_core.database`; [databases-and-environments](../foundations/databases-and-environments.md#engine-creation)).
- Stateless between invocations apart from the SSM key-rotation index (`SSM_API_KEY_INDEX` in `deployment/aws/terraform/lambda.tf`).

## Costs and usage

Spend cannot be verified from the repo; check it with `aws ce get-cost-and-usage` or the Billing console. The managed PostgreSQL provider is Neon (`deployment/aws/terraform/terraform.tfvars.example`); its database is a frozen archive ([databases-and-environments](../foundations/databases-and-environments.md#environments)), and is billed separately; check its console. Log groups keep 14 days (`retention_in_days` in `deployment/aws/terraform/lambda.tf`). `scripts/lambda_usage.py` pulls `REPORT` lines from CloudWatch for the functions and prints GB-seconds against the 400,000 free tier, daily breakdown, retry analysis and an hourly heatmap (`--days`, `--start`, `--end`). Gaps are filled with [backfills](../collection/odds-api-backfill.md); alerts go to Discord ([alerts](alerts.md)). `scripts/lambda_usage.py` hard-codes memory sizes (0.5 and 2.0 GB), so a change in Terraform does not propagate (#463).

## Secrets

API keys come from environment variables and Terraform variables (`odds_api_key`, `odds_api_keys`, `database_url` in `deployment/aws/terraform/variables.tf`), passed to the functions as environment variables in `deployment/aws/terraform/lambda.tf`; only the key-rotation index lives in SSM.

## Related

- [How to deploy to AWS by hand](howto-deploy-aws.md)
- [Debug the AWS scheduler](../scheduling/howto-debug-aws-scheduler.md)
- [CI/CD workflows](ci-cd-workflows.md)
- [Alerts](alerts.md)
