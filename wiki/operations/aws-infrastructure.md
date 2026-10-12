---
type: component
title: AWS infrastructure and deployment
description: The two Lambda functions and images, Terraform and bootstrap layout and variables, IAM/SSM/logs and costs, and the manual first-deploy and per-deploy steps, including activating self-scheduling rules.
sources:
  - docs/DEPLOYMENT.md#aws-lambda
  - docs/DEPLOYMENT.md#functions
  - docs/DEPLOYMENT.md#constraints
  - docs/DEPLOYMENT.md#terraform-layout
  - docs/DEPLOYMENT.md#terraform-variables
  - docs/DEPLOYMENT.md#aws
  - docs/DEPLOYMENT.md#operations
  - docs/DEPLOYMENT.md#deploying
  - docs/DEBUGGING_SCHEDULER.md#where-to-look
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/variables.tf
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/bootstrap/main.tf
  - deployment/aws/bootstrap/README.md
  - deployment/aws/build_and_push.sh
  - deployment/aws/build_and_push_scraper.sh
  - scripts/lambda_usage.py
verified: "7887039"
status: stable
---

# AWS infrastructure and deployment

Everything lives under `deployment/aws/`; region `eu-west-1`. Deployed state is not in the repo; the live view is `odds scheduler list-jobs --backend aws` ([jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs)).

## Functions

| | Scheduler | Scraper |
|---|---|---|
| Name | `var.project_name` (prod `odds-scheduler`, default `odds-scheduler-dev`) | `{project_name}-scraper` |
| Image | `Dockerfile.lambda`: `odds-core`, `odds-lambda`, `odds-analytics`, `odds-cli` wheels plus xgboost | `Dockerfile.scraper`: `odds-core`, `odds-lambda`, OddsHarvester fork, Playwright and Chromium |
| Memory / timeout | `lambda_memory_size` 512 MB, `lambda_timeout` 300 s | 2048 MB, 900 s (hard-coded in `scraper.tf`) |
| Created | always | only if `enable_oddsportal_scraper` |

Both run the same handler and registry ([scheduler backends and Lambda handler](../scheduling/scheduler-backends.md)). The scraper's 900 s timeout matters because the results job has no pre-schedule, so a timeout breaks its chain ([event lifecycle](../collection/event-lifecycle.md)). Log groups `/aws/lambda/<function>` keep 14 days. One IAM role covers both: S3 model read, EventBridge rule writes (restricted to `{rule_prefix}-*`; it lacks `events:ListRules`, so listing rules works only from a local shell), SSM read/write of `/{project_name}/active-api-key-index` (the [API key rotation](../collection/odds-api-client-and-quota.md) index, set through `SSM_API_KEY_INDEX`), and, if `betfair_enabled`, SSM read of the Betfair cert/key parameters ([Betfair client](../collection/betfair-exchange.md)).

The scraper image has no xgboost or `odds-analytics`, so publishing a model does not enable scoring on AWS ([publishing and scoring](../modelling/model-publishing-and-scoring.md)). `odds model publish` defaults to the literal bucket `odds-models-685946576110`.

## Terraform

`deployment/aws/terraform/`: `main.tf` (provider), `backend.tf` (S3 backend configured per environment by `backend.dev.hcl` / `backend.prod.hcl`), `variables.tf`, `lambda.tf`, `scraper.tf`, `eventbridge.tf`, `terraform.tfvars.example`. `variables.tf` is authoritative; notable defaults: `aws_region` `eu-west-1`, `rule_prefix` `odds`, `enable_oddsportal_scraper`/`enable_polymarket`/`betfair_enabled` false, `model_name` `epl-clv-home`.

EventBridge rules are named `{rule_prefix}-{job}[-{sport suffix}]` and `eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `ignore_changes` on `schedule_expression` and `state`, so a clean `terraform plan` does not mean the rules are active ([EventBridge backend](../scheduling/scheduler-backends.md)).

`deployment/aws/bootstrap/` is applied once: the Terraform state bucket and lock table, the ECR repository, and the shared versioned model bucket (`s3_models.tf`, see [publishing and scoring](../modelling/model-publishing-and-scoring.md)). Its own state file is committed.

## Constraints

- Database connections use `NullPool` automatically when `AWS_LAMBDA_FUNCTION_NAME` is set, whatever the scheduler backend ([databases and environments](../foundations/databases-and-environments.md)).
- Stateless between invocations apart from the SSM index.
- EventBridge limit of 300 rules per region (AWS limit as stated in `docs/DEPLOYMENT.md#constraints`).

## Deploying

Normal path: let GitHub Actions deploy ([CI/CD workflows](ci-cd-workflows.md)). `ci-cd.yml` deploys an ephemeral dev stack on every relevant push to `main`; `deploy-prod.yml` is run manually from the Actions tab and waits for approval on the `production` environment. Use the manual steps for a first deploy or a debugging session.

**First deployment only**

1. Apply `deployment/aws/bootstrap/` once: state bucket, lock table, ECR repository, model bucket. Follow `deployment/aws/bootstrap/README.md` to write `backend.dev.hcl` and `backend.prod.hcl` in `deployment/aws/terraform/`.
2. Set `database_url`, `odds_api_key` (or `odds_api_keys`) and any gates (`enable_oddsportal_scraper`, `enable_polymarket`, `betfair_enabled`) in your tfvars; keep the file out of git.

**Each deployment**

1. Build and push images from `deployment/aws`:
   ```bash
   cd deployment/aws
   ./build_and_push.sh dev eu-west-1 <account-id>
   ./build_and_push_scraper.sh dev eu-west-1 <account-id>
   ```
   - Arguments are environment (`dev` or `prod`), region and account id; the scripts fail on anything else.
   - The scraper's ECR repository is created by its script, not by Terraform.
2. Apply infrastructure:
   ```bash
   cd terraform
   terraform init -backend-config=backend.dev.hcl
   terraform plan
   terraform apply
   ```
   - Use `backend.prod.hcl` and set `project_name=odds-scheduler`, `rule_prefix=odds`, `environment=production` for production; this mirrors `.github/workflows/deploy-prod.yml`.
   - Pass the pushed tags through `image_tag` and `scraper_image_tag`.
3. Run migrations against the dev database: `uv run alembic upgrade head` ([database migration](../foundations/howto-database-migration.md)). The deploy workflows apply Terraform before `alembic upgrade head`, so migrations must be backward-compatible with the running release. Production migrations run only inside `deploy-prod.yml`.
4. Activate self-scheduling: invoke each scheduler and scraper job once (`scripts/ci/test_lambda.py --lambda-name <function> --jobs <csv>`). Rules start `DISABLED` on a `rate(1 day)` placeholder and only become real schedules after a job reschedules itself ([self-scheduling](../scheduling/self-scheduling.md)).
5. Verify: `bash scripts/verify-schedules.sh --rule-prefix odds --region eu-west-1`, and `odds scheduler list-jobs --backend aws`.
6. If scoring is wanted: `uv run odds model publish --name epl-clv-home --path <model.pkl>`.

Never run migrations or `odds fetch` against production from a laptop; deploy through the production workflow ([debug the AWS scheduler](../scheduling/howto-debug-aws-scheduler.md) for the read-only rules).

## Costs and usage

`docs/DEPLOYMENT.md#aws` estimates ~$0.20/month for the scheduler Lambda, under $0.01 for S3, under $0.10 for logs, free SSM (estimates, not measured here). `scripts/lambda_usage.py` pulls `REPORT` lines from CloudWatch for both functions and prints GB-seconds against the 400,000 free tier, daily breakdown, retry analysis and an hourly heatmap (`--days`, `--start`, `--end`). Backups are the managed database provider's; gaps are filled with [backfills](../collection/howto-load-historical-odds.md); alerts go to Discord ([alerts and health](alerts-and-health.md)).

## Hazards

- **Scraper has no alert or model env vars.** `scraper.tf` sets only `DATABASE_URL`, API keys, `LAMBDA_ARN`, `RULE_PREFIX` and `LOG_LEVEL`: no `ALERT_ENABLED`, `DISCORD_WEBHOOK_URL`, `MODEL_NAME` or `MODEL_BUCKET`, unlike `lambda.tf`. Scraper-side alerts are off by default ([configuration](../foundations/configuration.md#hazards)).
- `scripts/lambda_usage.py` hard-codes memory sizes (0.5 and 2.0 GB); a change in Terraform does not propagate.
- Production Terraform is applied from a workflow with `-var-file=terraform.tfvars.example` plus overrides, so the committed tfvars are not a record of what is deployed.
- Terraform does not show the live schedule (see above).

## Related

- [Debug the AWS scheduler](../scheduling/howto-debug-aws-scheduler.md), [CI/CD workflows](ci-cd-workflows.md), [Alerts and health](alerts-and-health.md)
