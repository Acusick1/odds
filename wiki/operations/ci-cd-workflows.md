---
type: reference
title: CI/CD workflows
description: What each GitHub Actions workflow runs and when — tests, lint, ephemeral dev deploy, the AWS prod deploy that must not run, daily validation, live-API and schedule health checks — and what they do not exercise.
sources:
  - .github/workflows/ci-cd.yml
  - .github/workflows/deploy-dev.yml
  - .github/workflows/deploy-prod.yml
  - .github/workflows/test.yml
  - .github/workflows/lint.yml
  - .github/workflows/live-api.yml
  - .pre-commit-config.yaml
  - .github/workflows/schedule-health-check.yml
  - scripts/ci/test_lambda.py
  - scripts/ci/verify_database.py
  - scripts/verify-schedules.sh
status: stable
---

# CI/CD workflows

Workflows live in `.github/workflows/`. Secrets come from the `development` and `production` GitHub environments. No workflow deploys to production, which is a single host deployed by [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). The `production` environment names the dormant AWS stack and its Neon database (secret values are not verifiable from the repo; check Settings, Environments). `test.yml` and `live-api.yml` install uv with `astral-sh/setup-uv` and set up no Python themselves, so `uv sync` picks an interpreter that satisfies `requires-python` in `pyproject.toml`. CI unit and integration tests run on py-pglite with `-m "not live_api"` ([test-suite](../foundations/test-suite.md), [howto-develop-locally](../foundations/howto-develop-locally.md)).

| Workflow | Trigger | What it does |
|---|---|---|
| `test.yml` | push to any branch but main; `workflow_call` | `deptry` per package, `uv run pytest tests/unit`, then `tests/integration -m "not live_api"` |
| `lint.yml` | push to non-main branches | `uvx pre-commit run --all-files` (ruff with all `DTZ` datetime rules, per the root `pyproject.toml`; Vale, lychee, wiki-lint, candidate-versions, which compares against the merge-base with `origin/main`) ([code-conventions](../foundations/code-conventions.md)) |
| `ci-cd.yml` | push to main touching `packages/**/*.py`, `tests/**`, `migrations/**`, `deployment/aws/**`, workflows or lockfiles | calls `test.yml`, then `deploy-dev.yml` |
| `deploy-dev.yml` | called by `ci-cd.yml` or manual | ephemeral dev stack |
| `deploy-prod.yml` | manual, or a published release; `production` environment approval | AWS stack deploy; disabled while AWS is dormant ([Prod deploy](#prod-deploy)) |
| `live-api.yml` | Monday 06:00 UTC | `pytest -m live_api` against third-party APIs |
| `schedule-health-check.yml` | 10:00 UTC cron, `production` environment; disabled | `scripts/verify-schedules.sh --rule-prefix odds` |

`schedule-health-check` fails every disabled rule ([Schedule verification](#schedule-verification)), so it would fail while AWS is dormant. It and `deploy-prod.yml` are disabled in GitHub; that state is not in the repo, so check it with `gh workflow list --all` and restore one with `gh workflow enable`.

## Dev deploy

`deploy-dev.yml` builds the Lambda image (tag `dev-<sha>`), runs `terraform apply` with `terraform.tfvars.example` plus overrides (Betfair on), runs `alembic upgrade head` and all of `pytest tests/migration`, invokes `check-health` and `fetch-betfair-exchange-epl` (a smoke test) through `scripts/ci/test_lambda.py`, runs `scripts/ci/verify_database.py` (events, recent snapshots and fetch logs) and `verify-schedules.sh --jobs check-health`, then always `terraform destroy`s ([howto-database-migration](../foundations/howto-database-migration.md)).

## Prod deploy

> [!WARNING]
> `deploy-prod.yml` is disabled in GitHub. Do not re-enable it, dispatch it or publish a release while AWS is dormant. Its job invocations re-enable every dormant EventBridge rule, so the Lambdas collect alongside the production host and write to the Neon archive ([Where each job runs](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).

`deploy-prod.yml` sets `project_name=odds-scheduler`, `rule_prefix=odds`, applies, migrates, runs only read-only schema and CLI tests under `tests/migration`, then invokes every scheduler job from the Terraform outputs. Those invocations make each job reschedule itself, which is what activates the `DISABLED` placeholder rules ([self-scheduling](../scheduling/self-scheduling.md), [scheduler-backends](../scheduling/scheduler-backends.md)). It then runs `verify-schedules.sh` and `verify_database.py`. Manual deploy steps live on [howto-deploy-aws](howto-deploy-aws.md).

## Schedule verification

`scripts/verify-schedules.sh` fails a rule that is not `ENABLED`, still has a `rate(...)` expression, or has a `cron(...)` year earlier than the current one. Without `--jobs` it discovers rules by prefix and skips names ending `-bootstrap`.

## Known gaps

- The dev deploy exercises only `check-health` and Betfair; `fetch-odds` is not invoked there.
- The workflows trigger on main (not `master`), and the scheduler log group is `/aws/lambda/odds-scheduler`.

## Related

- [Job deployment matrix](../scheduling/job-deployment-matrix.md)
- [Scheduler backends](../scheduling/scheduler-backends.md)
- [AWS infrastructure](aws-infrastructure.md)
