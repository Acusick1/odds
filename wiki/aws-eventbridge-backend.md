---
type: component
title: AWS EventBridge backend
summary: "How the AWS backend reschedules each job's EventBridge rule: rule naming, placeholder rules, ignore_changes, and the backend's env-var contract."
sources:
  - docs/ARCHITECTURE.md#aws-lambda-backend
  - docs/DEPLOYMENT.md#eventbridge-rules
  - docs/DEPLOYMENT.md#configuration
  - docs/DEVELOPMENT.md#aws-settings-awsconfig
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend
  - packages/odds-core/odds_core/config.py::AWSConfig
  - deployment/aws/terraform/eventbridge.tf
  - deployment/aws/terraform/lambda.tf
verified_at: 7887039
---

# AWS EventBridge backend

`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend` implements [the backend ABC](scheduler-backends.md) by rewriting one pre-created EventBridge rule per job. Terraform owns rule creation; the Lambda owns the schedule.

## Rewriting a rule

`schedule_next_execution` (retried with tenacity per `RetryConfig`, default 3 attempts) does two calls on rule `{rule_prefix}-{job_name}`:

1. `put_rule` with `State="ENABLED"` and a one-time expression from `_format_aws_schedule`: `cron(min hour day month ? year)` in UTC. Seconds are dropped.
2. `put_targets` with Lambda ARN `self.lambda_arn` and `Input` = `{"job": job_name, **payload}` (the [Lambda handler](lambda-handler.md) reads this). The target retry policy is 0 attempts, 60 s max event age.

`cancel_scheduled_execution` disables the rule. Failure after retries raises `SchedulingFailedError`.

## Rules and Terraform

`deployment/aws/terraform/eventbridge.tf` pre-creates each self-scheduling rule `DISABLED` with a `rate(1 day)` placeholder and `lifecycle { ignore_changes = [schedule_expression, state] }`, so `terraform plan` never shows live cadences. A rule is named `{rule_prefix}-{job}[-{sport suffix}]`, which is why jobs use [compound names](job-registry.md). The post-deploy Lambda invocation in the deploy workflows activates them (see [CI/CD workflows](ci-cd-workflows.md)). Live state is not in the repo: run `odds scheduler list-jobs --backend aws`.

`get_scheduled_jobs` lists rules by prefix and parses expressions with `_parse_schedule_expression`; a `rate(...)` placeholder yields `next_run_time=None` plus a warning log, and `ENABLED` maps to `SCHEDULED`, everything else to `CANCELLED`.

## Env-var contract

Constructor arguments win, then unprefixed env vars: `AWS_REGION`, `LAMBDA_ARN`, `RULE_PREFIX`. Terraform sets `LAMBDA_ARN` and `RULE_PREFIX` on each function (`deployment/aws/terraform/lambda.tf`); the scraper's `LAMBDA_ARN` is its own ARN, so scraper jobs reschedule onto the scraper. `get_scheduler_backend` first passes `settings.aws.*` (`packages/odds-core/odds_core/config.py::AWSConfig`, prefix `AWS_`: `AWS_REGION`, `AWS_LAMBDA_ARN`, `AWS_RULE_PREFIX`), which is what a local `list-jobs --backend aws` uses. Missing region, ARN or prefix raises `ConfigurationError` in the constructor. `health_check` results are cached for 5 minutes.

## Hazards

- `docs/DEPLOYMENT.md#eventbridge-rules` names the fixed Polymarket rule `odds-backfill-polymarket`; Terraform names it `{rule_prefix}-backfill-polymarket-bootstrap`.
- `aws.py` carries an unused `AWS_SCHEDULE_FORMAT = "at({timestamp})"` constant and its class docstring says `at(...)`; the implementation only emits `cron(...)` (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWS_SCHEDULE_FORMAT`).
- The Lambda role in `deployment/aws/terraform/lambda.tf` grants `PutRule`, `PutTargets`, `DisableRule`, `EnableRule`, `DescribeRule` but not `events:ListRules`, which `get_scheduled_jobs` and `health_check` call. Those work from a local shell with broader credentials.
- `get_scheduled_jobs` makes one `list_rules` call and does not follow `NextToken`.
- The handler docstring lists `AWS_LAMBDA_ARN` as the Lambda env var; the deployed variable is `LAMBDA_ARN` (`packages/odds-lambda/odds_lambda/lambda_handler.py`).

## Related

- [Self-scheduling](self-scheduling.md), [AWS infrastructure](aws-infrastructure.md), [job deployment matrix](job-deployment-matrix.md), [configuration](configuration.md).
