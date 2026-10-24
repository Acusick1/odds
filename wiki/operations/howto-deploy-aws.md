---
type: howto
title: Deploy to AWS by hand
description: Deploy the dormant AWS stack by hand — first-deploy bootstrap, then build, apply, migrate, activate rules and verify.
sources:
  - deployment/aws/bootstrap/main.tf
  - deployment/aws/bootstrap/README.md
  - deployment/aws/build_and_push.sh
  - deployment/aws/build_and_push_scraper.sh
  - deployment/aws/terraform/eventbridge.tf
  - .github/workflows/deploy-prod.yml
  - .github/workflows/deploy-dev.yml
  - scripts/ci/test_lambda.py
  - scripts/verify-schedules.sh
status: stable
---

# Deploy to AWS by hand

The AWS stack is dormant and production runs on a single local host ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). `ci-cd.yml` still deploys and destroys an ephemeral dev stack on every relevant push to main ([ci-cd-workflows](ci-cd-workflows.md)). Use these manual steps for a dev stack, a debugging session, or to reverse the move. What the infrastructure contains is on [aws-infrastructure](aws-infrastructure.md).

> [!WARNING]
> Step 4 against the production stack re-enables its rules, so the Lambdas collect alongside the production host and write to the Neon archive. Do it only to move production back to AWS, after stopping the production host's jobs. Never run migrations or `odds fetch` against the production database from a development machine ([databases-and-environments](../foundations/databases-and-environments.md#environments)).

## First deployment only

1. Apply `deployment/aws/bootstrap/` once.
   - It creates the state bucket, lock table, ECR repository and model bucket.
   - Follow `deployment/aws/bootstrap/README.md` to write `backend.dev.hcl` and `backend.prod.hcl` in `deployment/aws/terraform/`.
2. Set `database_url`, `odds_api_key` (or `odds_api_keys`) and any gates (`enable_oddsportal_scraper`, `enable_polymarket`, `betfair_enabled`) in your tfvars.
   - Keep the tfvars file out of git: it holds the database URL and API keys, which Terraform passes to the functions as environment variables ([aws-infrastructure](aws-infrastructure.md#secrets)).
   - Keep the file out of git; it holds secrets.

## Each deployment

1. Build and push images from `deployment/aws`.
   ```bash
   cd deployment/aws
   ./build_and_push.sh dev eu-west-1 <account-id>
   ./build_and_push_scraper.sh dev eu-west-1 <account-id>
   ```
   - Arguments are environment (`dev` or `prod`), region and account id; the scripts fail on anything else.
   - The scraper's ECR repository is created by its script, not by Terraform.
2. Apply infrastructure.
   ```bash
   cd terraform
   terraform init -backend-config=backend.dev.hcl
   terraform plan
   terraform apply
   ```
   - Use `backend.prod.hcl` and set `project_name=odds-scheduler`, `rule_prefix=odds`, `environment=production` for production; this mirrors `.github/workflows/deploy-prod.yml`.
   - Pass the pushed tags through `image_tag` and `scraper_image_tag`.
3. Run migrations against the dev database: `uv run alembic upgrade head` ([howto-database-migration](../foundations/howto-database-migration.md)).
   - The deploy workflows apply Terraform before `alembic upgrade head`, so new code can briefly run against the old schema (#443); keep migrations backward-compatible with the running release `(inferred)`.
   - The production Terraform stack's database is the Neon archive; the production database itself migrates only through [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) on the production host.
4. Activate self-scheduling: invoke each scheduler and scraper job once.
   ```bash
   uv run python scripts/ci/test_lambda.py --lambda-name <function> --jobs <csv>
   ```
   - Rules start `DISABLED` on a `rate(1 day)` placeholder and only become real schedules after a job reschedules itself ([self-scheduling](../scheduling/self-scheduling.md)).
5. Verify the rules.
   ```bash
   bash scripts/verify-schedules.sh --rule-prefix odds --region eu-west-1
   uv run odds scheduler list-jobs --backend aws
   ```
   - Terraform does not show the live schedule, so `list-jobs` is the only record of what runs ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)).
6. If scoring is wanted, publish a model: `uv run odds model publish --name epl-clv-home --path <model.pkl>` ([model-store](../modelling/model-store.md)).
   - Scoring runs only where the scheduler image has `odds-analytics` ([prediction-scoring](../modelling/prediction-scoring.md)).

## Related

- [AWS infrastructure](aws-infrastructure.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md), for the read-only rules
- [CI/CD workflows](ci-cd-workflows.md)
