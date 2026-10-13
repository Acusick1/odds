---
type: component
title: Scheduler backends, EventBridge rules and the Lambda handler
description: The SchedulerBackend interface and selection, how the AWS backend rewrites placeholder EventBridge rules, the Lambda handler's payload routing and extra failure alert, and the Railway no-op shim.
sources:
  - packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend
  - packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWS_SCHEDULE_FORMAT
  - packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend
  - packages/odds-lambda/odds_lambda/scheduling/exceptions.py::BackendUnavailableError
  - packages/odds-lambda/odds_lambda/scheduling/health_check.py::HealthCheckBuilder
  - packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::resolve_job_name
  - packages/odds-core/odds_core/config.py::AWSConfig
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
status: stable
---

# Scheduler backends, EventBridge rules and the Lambda handler

`packages/odds-lambda/odds_lambda/scheduling/backends/` holds one ABC and three implementations (AWS EventBridge, local APScheduler, a Railway shim). Jobs never import a concrete backend; they call `packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend`, usually through [self-scheduling](self-scheduling.md). The Lambda handler is the entry point EventBridge invokes.

## The ABC

`packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend` requires:

- `schedule_next_execution(job_name, next_time, payload=None)`, a one-shot booking; `payload` is merged next to `{"job": job_name}`.
- `cancel_scheduled_execution`, `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs`, `get_job_status`.
- `health_check() -> BackendHealth`, `validate_configuration() -> ValidationResult`, `get_backend_name()`.

The constructor takes `dry_run` and a `packages/odds-lambda/odds_lambda/scheduling/backends/base.py::RetryConfig`. In dry-run mode every implementation logs and returns; the AWS and local listings return `[]`. Return types are frozen dataclasses (`packages/odds-lambda/odds_lambda/scheduling/backends/base.py::ScheduledJob`, `packages/odds-lambda/odds_lambda/scheduling/backends/base.py::BackendHealth`, `packages/odds-lambda/odds_lambda/scheduling/backends/base.py::ValidationResult`).

`packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend` (`packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py`) reads `settings.scheduler.backend` (`SCHEDULER_BACKEND`, default `local`) unless `backend_type` is passed, lowercases it, and lazy-imports the class. An unknown value raises `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::ConfigurationError`. The AWS branch fills region, ARN and prefix from `settings.aws`; the local branch is [the local scheduler](local-scheduler.md).

`packages/odds-lambda/odds_lambda/scheduling/health_check.py` provides `packages/odds-lambda/odds_lambda/scheduling/health_check.py::HealthCheckBuilder` (`pass_check`, `fail_check`, `check_condition`, `build`) and `packages/odds-lambda/odds_lambda/scheduling/health_check.py::ValidationBuilder`. A backend is healthy when no check failed. `odds scheduler health` and `odds scheduler test-backend` use the configured backend only; they do not accept `--backend`.

## Railway shim

`packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend` is not a deployment target. `schedule_next_execution` and `cancel_scheduled_execution` only log. `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` and `get_job_status` raise `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::BackendUnavailableError` (`packages/odds-lambda/odds_lambda/scheduling/exceptions.py`), which `odds scheduler list-jobs` catches and prints as "not supported". Jobs still run their own gate logic under it, but cannot move their own cadence.

## AWS EventBridge backend

`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend` (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`) rewrites one pre-created EventBridge rule per job. Terraform owns rule creation; the Lambda owns the schedule.

`schedule_next_execution` (retried with tenacity per `packages/odds-lambda/odds_lambda/scheduling/backends/base.py::RetryConfig`, default 3 attempts) does two calls on rule `{rule_prefix}-{job_name}`:

1. `put_rule` with `State="ENABLED"` and a one-time expression from `packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::_format_aws_schedule`: `cron(min hour day month ? year)` in UTC. Seconds are dropped.
2. `put_targets` with Lambda ARN `self.lambda_arn` and `Input` = `{"job": job_name, **payload}`. The target retry policy is 0 attempts, 60 s max event age.

`cancel_scheduled_execution` disables the rule. Failure after retries raises `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::SchedulingFailedError`.

### Rules and Terraform

`deployment/aws/terraform/eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `lifecycle { ignore_changes = [schedule_expression, state] }`, so `terraform plan` never shows live cadences. A rule is named `{rule_prefix}-{job}[-{sport suffix}]`, which is why jobs use [compound names](jobs-and-deployment-matrix.md#per-sport-jobs-and-compound-names). The post-deploy Lambda invocations in the deploy workflows activate the placeholders (prod invocations of the scraper are `continue-on-error`) ([CI/CD workflows](../operations/ci-cd-workflows.md)). Live state is not in the repo: run `odds scheduler list-jobs --backend aws`.

`packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` lists rules by prefix and parses expressions with `_parse_schedule_expression`; a `rate(...)` placeholder yields `next_run_time=None` plus a warning log, and `ENABLED` maps to `SCHEDULED`, everything else to `CANCELLED`. The parser builds naive datetimes ([Conventions](../foundations/conventions.md#timezones)). The MCP tool `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` never goes through this path: it reads only the local APScheduler store ([MCP server](../agent/mcp-server.md)).

### Env-var contract

Constructor arguments win, then unprefixed env vars: `AWS_REGION`, `LAMBDA_ARN`, `RULE_PREFIX`. Terraform sets `LAMBDA_ARN` and `RULE_PREFIX` on each function (`deployment/aws/terraform/lambda.tf`); the scraper's `LAMBDA_ARN` is its own ARN, so scraper jobs reschedule onto the scraper. `packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend` first passes `settings.aws.*` (`packages/odds-core/odds_core/config.py::AWSConfig`, prefix `AWS_`: `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX`), which is what a local `list-jobs --backend aws` uses ([Configuration](../foundations/configuration.md#hazards)). Missing region, ARN or prefix raises `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::ConfigurationError` in the constructor. `health_check` results are cached for 5 minutes.

## Lambda handler

`packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler` is the CMD of both Lambda images (`deployment/aws/Dockerfile.lambda`, `deployment/aws/Dockerfile.scraper`); the scheduler and scraper functions differ only in image contents and rules ([AWS infrastructure](../operations/aws-infrastructure.md)).

EventBridge delivers the target `Input` written by the AWS backend or by Terraform:

```json
{"job": "fetch-odds", "sport": "soccer_epl"}
```

`job` may be a compound name (`fetch-odds-epl`). The handler:

1. Raises `ValueError` if `job` is missing.
2. Calls `packages/odds-lambda/odds_lambda/scheduling/jobs.py::resolve_job_name` to split it into base job and suffix-derived sport. An explicit `sport` in the payload wins over the suffix.
3. Binds `job`, `request_id` and `sport` into structlog context so CloudWatch can filter on them (`configure_logging(..., json_output=True)` runs at import).
4. Builds `JobContext.from_payload` from every payload key except `job`, then runs `get_job_function(base)(ctx)` under `asyncio.run`.
5. Returns `{"statusCode": 200, "body": ...}`, or 500 with `error` and `error_type` on any exception.

Always pass `sport` when invoking manually ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)): with a bare `fetch-odds` and no sport, the job runs every configured sport and reschedules under the bare name, which matches no rule.

On an exception the handler logs `lambda_failed` and calls `packages/odds-core/odds_core/alerts.py::send_critical` directly with the job name. That path does not touch `packages/odds-core/odds_core/models.py::AlertHistory`, so it is never rate limited.

## Hazards

- **Double alert.** A job wrapped in `packages/odds-core/odds_core/alerts.py::job_alert_context` that raises sends its own rate-limited critical, then the handler sends another unconditional one. Jobs without the wrapper (for example `fetch-odds`) send their own `packages/odds-core/odds_core/alerts.py::send_critical` before re-raising ([Alerts and health](../operations/alerts-and-health.md)).
- The 500 response is returned rather than raised, so Lambda records the invocation as successful at the platform level; EventBridge retry policy is 0 anyway.
- `RailwayBackend.health_check` fails its "Running in Railway environment" check unless `RAILWAY_ENVIRONMENT` is set, so it reports unhealthy anywhere else. The `packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend` docstring calls Railway an "alternative production" backend; `packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend` is a no-op shim.
- `packages/odds-lambda/odds_lambda/scheduling/backends/base.py::RetryConfig` is consumed only by the AWS backend, and only `max_attempts`, `initial_delay_seconds` and `max_delay_seconds`; `exponential_base` and `jitter` are read by nothing. Each `packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule` call builds a fresh backend (and boto3 clients on AWS); there is no shared instance.
- Terraform names the fixed Polymarket rule `{rule_prefix}-backfill-polymarket-bootstrap` (`deployment/aws/terraform/eventbridge.tf`), not `odds-backfill-polymarket`.
- `packages/odds-lambda/odds_lambda/scheduling/backends/aws.py` carries an unused `AWS_SCHEDULE_FORMAT = "at({timestamp})"` constant and its class docstring says `at(...)`; the implementation only emits `cron(...)`.
- The Lambda role in `deployment/aws/terraform/lambda.tf` grants `PutRule`, `PutTargets`, `DisableRule`, `EnableRule`, `DescribeRule` but not `events:ListRules`, which `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` and `health_check` call. Those work only from a local shell with broader credentials. `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` makes one `list_rules` call and does not follow `NextToken`.
- The handler module docstring names `AWS_LAMBDA_ARN` as the Lambda env var; the deployed variable the backend reads is `LAMBDA_ARN`.
- Alert delivery inside Lambda depends on `ALERT_ENABLED` and `DISCORD_WEBHOOK_URL`, which `lambda.tf` sets for the scheduler but `deployment/aws/terraform/scraper.tf` does not set for the scraper function, so scraper-side alerts are off by default config. Deployed values are not verifiable from the repo; check with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`.

## Related

- [Self-scheduling and the decision engine](self-scheduling.md)
- [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md)
- [Local scheduler and always-on host](local-scheduler.md)
