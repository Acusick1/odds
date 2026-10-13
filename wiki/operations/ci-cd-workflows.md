---
type: component
title: CI/CD workflows
description: What each GitHub Actions workflow runs and when — tests, lint, ephemeral dev deploy, gated prod deploy, daily validation, live-API and schedule health checks — and what they do not exercise.
sources:
  - .github/workflows/ci-cd.yml
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - .github/workflows/test.yml
  - .github/workflows/lint.yml
  - .github/workflows/daily-validation.yml
  - .github/workflows/live-api.yml
  - .pre-commit-config.yaml
  - .github/workflows/schedule-health-check.yml
  - scripts/ci/test_lambda.py
  - scripts/ci/verify_database.py
  - scripts/verify-schedules.sh
status: stable
---

# CI/CD workflows

Workflows live in `.github/workflows/`. Secrets come from the `development` and `production` GitHub environments. CI unit and integration tests run on py-pglite with `-m "not live_api"` ([local development](../foundations/howto-develop-locally.md)).

| Workflow | Trigger | What it does |
|---|---|---|
| `test.yml` | push to any branch but main; `workflow_call` | `uv run pytest tests/unit`, then `tests/integration -m "not live_api"` |
| `lint.yml` | push to non-main branches | `uvx pre-commit run --all-files` (ruff, Vale, lychee, wiki-lint) ([conventions](../foundations/conventions.md)) |
| `ci-cd.yml` | push to main touching `packages/**/*.py`, `tests/**`, `migrations/**`, `deployment/aws/**`, workflows or lockfiles | calls `test.yml`, then `deploy-dev.yml` |
| `deploy-dev.yml` | called by `ci-cd.yml` or manual | ephemeral dev stack |
| `deploy-prod.yml` | manual, or a published release; `production` environment approval | persistent prod deploy |
| `daily-validation.yml` | 09:00 UTC cron | `odds validate daily --check-discovery --output-json validation_<yesterday>.json`, artifact kept 30 days |
| `live-api.yml` | Monday 06:00 UTC | `pytest -m live_api` against third-party APIs |
| `schedule-health-check.yml` | 10:00 UTC cron, `production` environment | `scripts/verify-schedules.sh --rule-prefix odds` |

`daily-validation` checks the previous day, so a file dated 20251103 covers 3 November. Its discovery check is NBA-only ([data validation](../storage/data-validation.md), [debug the AWS scheduler](../scheduling/howto-debug-aws-scheduler.md)).

## Dev deploy

`deploy-dev.yml` builds both images (tag `dev-<sha>`), runs `terraform apply` with `terraform.tfvars.example` plus overrides (scraper and Betfair on), runs `alembic upgrade head` and all of `pytest tests/migration`, invokes `check-health` and `fetch-betfair-exchange-epl` (a smoke test) through `scripts/ci/test_lambda.py`, runs `scripts/ci/verify_database.py` and `verify-schedules.sh --jobs check-health`, then always `terraform destroy`s ([database migration](../foundations/howto-database-migration.md)).

## Prod deploy

`deploy-prod.yml` sets `project_name=odds-scheduler`, `rule_prefix=odds`, applies, migrates, runs only read-only schema and CLI tests under `tests/migration`, then invokes every scheduler job and, as `continue-on-error`, every scraper job from the Terraform outputs. Those invocations make each job reschedule itself, which is what activates the `DISABLED` placeholder rules ([self-scheduling](../scheduling/self-scheduling.md), [scheduler backends](../scheduling/scheduler-backends.md)). It then runs `verify-schedules.sh` and `verify_database.py`. Manual deploy steps live on [AWS infrastructure and deployment](aws-infrastructure.md#deploying).

## Schedule verification

`scripts/verify-schedules.sh` fails a rule that is not `ENABLED`, still has a `rate(...)` expression, or has a `cron(...)` year earlier than the current one. Without `--jobs` it discovers rules by prefix and skips names ending `-bootstrap`.

## Hazards

- `test.yml` and `live-api.yml` set up Python 3.11 while every package requires `>=3.12` (`pyproject.toml`); `deploy-*` use 3.12.
- `.github/workflows/README.md` says production deploys from `master` and uses a `-fetch-odds` log group; the workflows trigger on main and the log group is `/aws/lambda/odds-scheduler`.
- The scraper test step in prod is `continue-on-error`, so a broken scraper does not fail a deploy.
- Dev deploy exercises only `check-health` and Betfair; `fetch-odds` is not invoked there.
- Daily validation's discovery check is NBA-only, so EPL and MLB days are checked against NBA discovery.

## Related

- [Jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md), [Scheduler backends and Lambda handler](../scheduling/scheduler-backends.md), [AWS infrastructure](aws-infrastructure.md)
