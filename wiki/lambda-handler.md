---
type: component
title: Lambda handler
summary: "The single Lambda entry point: EventBridge payload format, job and sport routing, and its extra failure alert that skips rate limiting."
sources:
  - docs/DEBUGGING_SCHEDULER.md#invoking-a-job-manually
  - docs/DEPLOYMENT.md#functions
  - packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::resolve_job_name
verified_at: 7887039
---

# Lambda handler

`packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler` is the CMD of both Lambda images (`deployment/aws/Dockerfile.lambda`, `deployment/aws/Dockerfile.scraper`); the scheduler and scraper functions differ only in image contents and rules ([AWS infrastructure](aws-infrastructure.md)).

## Payload

EventBridge delivers the target `Input` written by the [AWS backend](aws-eventbridge-backend.md) or by Terraform:

```json
{"job": "fetch-odds", "sport": "soccer_epl"}
```

`job` may be a compound name (`fetch-odds-epl`). The handler:

1. Raises `ValueError` if `job` is missing.
2. Calls `resolve_job_name` to split it into base job and suffix-derived sport. An explicit `sport` in the payload wins over the suffix.
3. Binds `job`, `request_id` and `sport` into structlog context, so CloudWatch can filter on them (`configure_logging(..., json_output=True)` runs at import).
4. Builds `JobContext.from_payload` from every payload key except `job`, then runs `get_job_function(base)(ctx)` under `asyncio.run`.
5. Returns `{"statusCode": 200, "body": ...}`, or 500 with `error` and `error_type` on any exception.

Manual invocation and logs are in [debug the AWS scheduler](howto-debug-aws-scheduler.md). Pass `sport`: with a bare `fetch-odds` and no sport, the job runs every configured sport and reschedules under the bare name, which matches no rule ([job registry](job-registry.md)).

## Failure path

On an exception the handler logs `lambda_failed` and calls `send_critical` directly with the job name. That path does not touch `AlertHistory`, so it is never rate limited.

## Hazards

- Double alert: a job wrapped in `job_alert_context` that raises sends its own rate-limited critical, then the handler sends another unconditional one. Jobs without the wrapper (for example `fetch-odds`) send their own `send_critical` too before re-raising. See [alerts module](alerts-module.md).
- The 500 response is returned rather than raised, so Lambda records the invocation as successful at the platform level; EventBridge retry policy is 0 anyway (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend`).
- The module docstring names `AWS_LAMBDA_ARN` as a required env var; the deployed variable the backend reads is `LAMBDA_ARN`.
- Alert delivery inside Lambda depends on `ALERT_ENABLED` and `DISCORD_WEBHOOK_URL`, which `deployment/aws/terraform/lambda.tf` sets for the scheduler but `deployment/aws/terraform/scraper.tf` does not set for the scraper function, so scraper-side alerts are disabled by default config (deployed values are not verifiable from the repo; check with `aws lambda get-function-configuration --function-name odds-scheduler-scraper`).

## Related

- [Self-scheduling](self-scheduling.md), [scheduler backends](scheduler-backends.md).
