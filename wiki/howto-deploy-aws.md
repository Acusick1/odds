---
type: howto
title: Deploy to AWS
summary: Build and push both images, apply Terraform per environment, bootstrap and publish a model on first deploy, and how CI deploys.
sources:
  - docs/DEPLOYMENT.md#deploying
  - deployment/aws/build_and_push.sh
  - deployment/aws/build_and_push_scraper.sh
  - deployment/aws/bootstrap/README.md
verified_at: 7887039
---

# Deploy to AWS

Normal path: let GitHub Actions deploy ([CI/CD workflows](ci-cd-workflows.md)). `ci-cd.yml` deploys an ephemeral dev stack on every relevant push to `main`; `deploy-prod.yml` is run manually from the Actions tab and waits for approval on the `production` environment. Use the manual steps below for a first deploy or a debugging session. Architecture: [AWS infrastructure](aws-infrastructure.md).

## First deployment only

1. Apply `deployment/aws/bootstrap/` once: state bucket, lock table, ECR repository, model bucket. Follow `deployment/aws/bootstrap/README.md` to write `backend.dev.hcl` and `backend.prod.hcl` in `deployment/aws/terraform/`.
2. Set `database_url`, `odds_api_key` (or `odds_api_keys`) and any gates (`enable_oddsportal_scraper`, `enable_polymarket`, `betfair_enabled`) in your tfvars; keep the file out of git.

## Each deployment

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
3. Run migrations against the dev database: `uv run alembic upgrade head` ([database migration](howto-database-migration.md)). Production migrations run only inside `deploy-prod.yml`.
4. Activate self-scheduling: invoke each scheduler and scraper job once (`scripts/ci/test_lambda.py --lambda-name <function> --jobs <csv>`). Rules start `DISABLED` on a `rate(1 day)` placeholder and only become real schedules after a job reschedules itself ([self-scheduling](self-scheduling.md)).
5. Verify: `bash scripts/verify-schedules.sh --rule-prefix odds --region eu-west-1`, and `odds scheduler list-jobs --backend aws`.
6. If scoring is wanted: `uv run odds model publish --name epl-clv-home --path <model.pkl>` ([S3 model store](s3-model-store.md), [train and publish a model](howto-train-and-publish-model.md)).

## Gotchas

- Never run migrations or `odds fetch` against production from a laptop; deploy through the production workflow ([debug the AWS scheduler](howto-debug-aws-scheduler.md) for the read-only rules).
- Terraform does not show the live schedule, so a clean plan does not mean the rules are active ([AWS EventBridge backend](aws-eventbridge-backend.md)).
- The scraper image has no xgboost, so model publishing does not enable scoring on AWS ([fetch-oddsportal](job-fetch-oddsportal.md)).
