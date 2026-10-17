---
type: reference
title: Scheduler backends
description: The SchedulerBackend interface and selection, the AWS EventBridge backend and its env contract, the Railway shim, health checks, and the Lambda handler EventBridge invokes.
sources:
  - odds_lambda.scheduling.backends.base.SchedulerBackend
  - odds_lambda.scheduling.backends.base.RetryConfig
  - odds_lambda.scheduling.backends.get_scheduler_backend
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
  - odds_lambda.scheduling.backends.aws.AWS_SCHEDULE_FORMAT
  - odds_lambda.scheduling.backends.railway.RailwayBackend
  - odds_lambda.scheduling.exceptions.BackendUnavailableError
  - odds_lambda.scheduling.exceptions.ConfigurationError
  - odds_lambda.scheduling.exceptions.SchedulingFailedError
  - odds_lambda.scheduling.health_check.HealthCheckBuilder
  - odds_lambda.scheduling.health_check.ValidationBuilder
  - odds_lambda.lambda_handler.lambda_handler
  - odds_lambda.scheduling.jobs.resolve_job_name
  - odds_lambda.scheduling.jobs.JobContext.from_payload
  - odds_core.config.AWSConfig
  - odds_core.alerts.send_critical
  - odds_core.alerts.job_alert_context
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/Dockerfile.lambda
  - deployment/aws/Dockerfile.scraper
status: stable
---

# Scheduler backends

`packages/odds-lambda/odds_lambda/scheduling/backends/` holds one ABC and three implementations (AWS EventBridge, local APScheduler, a Railway shim). Jobs never import a concrete backend; they call `odds_lambda.scheduling.backends.get_scheduler_backend`, usually through [self-scheduling](self-scheduling.md). The Lambda handler is the entry point EventBridge invokes. Production runs on the local backend; the AWS backend's rules are disabled and its Lambdas dormant ([Where each job runs](job-deployment-matrix.md#dormant-aws-rules), [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)).

## The ABC

`odds_lambda.scheduling.backends.base.SchedulerBackend` requires:

- `odds_lambda.scheduling.backends.base.SchedulerBackend.schedule_next_execution(job_name, next_time, payload=None)`, a one-shot booking; `payload` is merged next to `{"job": job_name}`.
- `odds_lambda.scheduling.backends.base.SchedulerBackend.cancel_scheduled_execution`, `odds_lambda.scheduling.backends.base.SchedulerBackend.get_scheduled_jobs`, `odds_lambda.scheduling.backends.base.SchedulerBackend.get_job_status`.
- `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check() -> BackendHealth`, `odds_lambda.scheduling.backends.base.SchedulerBackend.validate_configuration() -> ValidationResult`, `odds_lambda.scheduling.backends.base.SchedulerBackend.get_backend_name()`.

The constructor takes a dry-run flag and a `odds_lambda.scheduling.backends.base.RetryConfig`. In dry-run mode the AWS and local implementations log and return, and their listings return `[]`; `odds_lambda.scheduling.backends.railway.RailwayBackend.get_scheduled_jobs` raises regardless. `odds_lambda.scheduling.backends.get_scheduler_backend` defaults the dry-run flag to false, so a listing built without it (as `odds scheduler list-jobs` does) is never emptied by dry-run. Return types are frozen dataclasses (`odds_lambda.scheduling.backends.base.ScheduledJob`, `odds_lambda.scheduling.backends.base.BackendHealth`, `odds_lambda.scheduling.backends.base.ValidationResult`).

`odds_lambda.scheduling.backends.base.RetryConfig` is consumed only by the AWS backend, and only `max_attempts`, `initial_delay_seconds` and `max_delay_seconds`; `exponential_base` and `jitter` are read by nothing (#473). Each `odds_lambda.scheduling.helpers.self_schedule` call builds a fresh backend (and boto3 clients on AWS); there is no shared instance.

`odds_lambda.scheduling.backends.get_scheduler_backend` reads `settings.scheduler.backend` (`SCHEDULER_BACKEND`, default `local`) unless `backend_type` is passed, lowercases it, and lazy-imports the class. An unknown value raises `odds_lambda.scheduling.exceptions.ConfigurationError`. The AWS branch fills region, ARN and prefix from `settings.aws`; the local branch is [the local scheduler](local-scheduler.md).

## Health checks

`odds_lambda.scheduling.health_check` provides `odds_lambda.scheduling.health_check.HealthCheckBuilder` (`odds_lambda.scheduling.health_check.HealthCheckBuilder.pass_check`, `odds_lambda.scheduling.health_check.HealthCheckBuilder.fail_check`, `odds_lambda.scheduling.health_check.HealthCheckBuilder.check_condition`, `odds_lambda.scheduling.health_check.HealthCheckBuilder.build`) and `odds_lambda.scheduling.health_check.ValidationBuilder`. A backend is healthy when no check failed. `odds scheduler health` and `odds scheduler test-backend` use the configured backend only; they do not accept `--backend`.

## Railway shim

`odds_lambda.scheduling.backends.railway.RailwayBackend` is not a deployment target. `odds_lambda.scheduling.backends.railway.RailwayBackend.schedule_next_execution` and `odds_lambda.scheduling.backends.railway.RailwayBackend.cancel_scheduled_execution` only log. `odds_lambda.scheduling.backends.railway.RailwayBackend.get_scheduled_jobs` and `odds_lambda.scheduling.backends.railway.RailwayBackend.get_job_status` raise `odds_lambda.scheduling.exceptions.BackendUnavailableError`, which `odds scheduler list-jobs` catches and prints as "not supported". Jobs still run their own gate logic under it, but cannot move their own cadence. Its `odds_lambda.scheduling.backends.railway.RailwayBackend.health_check` fails the "Running in Railway environment" check unless `RAILWAY_ENVIRONMENT` is set, so it reports unhealthy anywhere else.

## AWS EventBridge backend

`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend` rewrites one pre-created EventBridge rule per job. Terraform owns rule creation; the Lambda owns the schedule.

`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.schedule_next_execution` (retried with tenacity per `odds_lambda.scheduling.backends.base.RetryConfig`, default 3 attempts) does two calls on rule `{rule_prefix}-{job_name}`:

1. `put_rule` with `State="ENABLED"` and a one-time expression from `odds_lambda.scheduling.backends.aws._format_aws_schedule`: `cron(min hour day month ? year)` in UTC. Seconds are dropped.
2. `put_targets` with Lambda ARN `self.lambda_arn` and `Input` = `{"job": job_name, **payload}`. The target retry policy is 0 attempts, 60 s max event age.

`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.cancel_scheduled_execution` disables the rule. Failure after retries raises `odds_lambda.scheduling.exceptions.SchedulingFailedError`. `odds_lambda.scheduling.backends.aws.AWS_SCHEDULE_FORMAT` (`at({timestamp})`) is read by nothing and the class docstring mentions `at(...)`, but the backend only emits `cron(...)` (`odds_lambda.scheduling.backends.aws._format_aws_schedule`).

`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.get_scheduled_jobs` lists rules by prefix with one `list_rules` call and does not follow `NextToken` (#443). It parses expressions with `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._parse_schedule_expression`; a `rate(...)` placeholder yields `next_run_time=None` plus a warning log, and `ENABLED` maps to `SCHEDULED`, everything else to `CANCELLED`. The parser builds naive datetimes ([Timezones](../foundations/timezones.md)). The MCP tool `odds_mcp.server.get_scheduled_jobs` never goes through this path: it reads only the local APScheduler store ([MCP scheduling tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).

The Lambda role in `deployment/aws/terraform/lambda.tf` grants `PutRule`, `PutTargets`, `DisableRule`, `EnableRule` and `DescribeRule` but not `events:ListRules`, which `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.get_scheduled_jobs` and `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.health_check` call. Listing and health checks therefore work only from a shell with broader credentials (#443).

### Rules and Terraform

`deployment/aws/terraform/eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `lifecycle { ignore_changes = [schedule_expression, state] }`, so `terraform plan` never shows live cadences. A rule is named `{rule_prefix}-{job}[-{sport suffix}]`, which is why jobs use [compound names](job-registry.md#compound-job-names). The post-deploy Lambda invocations in the deploy workflows activate the placeholders (prod invocations of the scraper are `continue-on-error`), so a run of `deploy-prod.yml` re-enables every dormant rule ([CI/CD workflows](../operations/ci-cd-workflows.md)). Live state is not in the repo: run `odds scheduler list-jobs --backend aws` ([Deployed state](job-deployment-matrix.md#deployed-state)).

### Env-var contract

Constructor arguments win, then unprefixed env vars: `AWS_REGION`, `LAMBDA_ARN`, `RULE_PREFIX`. Terraform sets `LAMBDA_ARN` and `RULE_PREFIX` on each function (`deployment/aws/terraform/lambda.tf`); the scraper's `LAMBDA_ARN` is its own ARN, so scraper jobs reschedule onto the scraper. `odds_lambda.scheduling.backends.get_scheduler_backend` first passes `settings.aws.*` (`odds_core.config.AWSConfig`, prefix `AWS_`: `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX`), which is what a local `list-jobs --backend aws` uses ([Configuration](../foundations/configuration.md#variables-read-outside-settings)). Missing region, ARN or prefix raises `odds_lambda.scheduling.exceptions.ConfigurationError` in the constructor. `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend.health_check` results are cached for 5 minutes.

## Lambda handler

`odds_lambda.lambda_handler.lambda_handler` is the CMD of both Lambda images (`deployment/aws/Dockerfile.lambda`, `deployment/aws/Dockerfile.scraper`); the scheduler and scraper functions differ only in image contents and rules ([AWS infrastructure](../operations/aws-infrastructure.md)). The Lambda env var holding the function's own ARN is `LAMBDA_ARN`, the variable the backend reads ([Env-var contract](#env-var-contract)). Alert environment variables per function are on [AWS infrastructure](../operations/aws-infrastructure.md).

EventBridge delivers the target `Input` written by the AWS backend or by Terraform:

```json
{"job": "fetch-odds", "sport": "soccer_epl"}
```

`job` may be a compound name (`fetch-odds-epl`). The handler:

1. Raises `ValueError` if `job` is missing.
2. Calls `odds_lambda.scheduling.jobs.resolve_job_name` to split it into base job and suffix-derived sport. An explicit `sport` in the payload wins over the suffix.
3. Binds `job`, `request_id` and `sport` into structlog context so CloudWatch can filter on them (`configure_logging(..., json_output=True)` runs at import).
4. Builds `odds_lambda.scheduling.jobs.JobContext.from_payload` from every payload key except `job`, then runs `get_job_function(base)(ctx)` under `asyncio.run`.
5. Returns `{"statusCode": 200, "body": ...}`, or 500 with `error` and `error_type` on any exception.

Failures return a 500 body rather than raising, so Lambda counts the invocation as successful (#442). EventBridge's target retry policy is 0 attempts anyway.

On an exception the handler logs `lambda_failed` and calls `odds_core.alerts.send_critical` directly with the job name, in addition to the job's own alert (#442). That path does not touch `odds_core.models.AlertHistory`, so it is never rate limited. A job wrapped in `odds_core.alerts.job_alert_context` that raises sends its own rate-limited critical first; jobs without the wrapper (for example `fetch-odds`) send their own `odds_core.alerts.send_critical` before re-raising ([Alerts](../operations/alerts.md)).

With a bare `fetch-odds` and no `sport`, the job runs every configured sport and reschedules under the bare name, which matches no rule, so manual invocations always pass `sport` ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md#invoke-manually)).

## Related

- [Self-scheduling](self-scheduling.md)
- [Where each job runs](job-deployment-matrix.md)
- [Local scheduler](local-scheduler.md)
