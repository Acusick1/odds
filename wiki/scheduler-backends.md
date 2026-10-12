---
type: component
title: Scheduler backend interface
summary: The SchedulerBackend ABC, backend selection with SCHEDULER_BACKEND, the Railway no-op shim, BackendUnavailableError and backend health checks.
sources:
  - docs/ARCHITECTURE.md#multi-backend-scheduler
  - docs/ARCHITECTURE.md#backend-interface
  - docs/ARCHITECTURE.md#railway-backend
  - docs/DEPLOYMENT.md#deployment-targets
  - packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend
  - packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend
  - packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend
  - packages/odds-lambda/odds_lambda/scheduling/exceptions.py::BackendUnavailableError
  - packages/odds-lambda/odds_lambda/scheduling/health_check.py::HealthCheckBuilder
verified_at: 7887039
---

# Scheduler backend interface

`packages/odds-lambda/odds_lambda/scheduling/backends/` holds one ABC and three implementations. Jobs never import a concrete backend; they call `get_scheduler_backend()` (usually through [self-scheduling](self-scheduling.md)).

## The ABC

`packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend` requires:

- `schedule_next_execution(job_name, next_time, payload=None)` is a one-shot booking; `payload` is merged next to `{"job": job_name}`.
- `cancel_scheduled_execution`, `get_scheduled_jobs`, `get_job_status`.
- `health_check() -> BackendHealth`, `validate_configuration() -> ValidationResult`, `get_backend_name()`.

The constructor takes `dry_run` and a `RetryConfig`. In dry-run mode every implementation logs and returns; the AWS and local listings return `[]`. Return types are frozen dataclasses (`ScheduledJob`, `BackendHealth`, `ValidationResult`).

## Selection

`packages/odds-lambda/odds_lambda/scheduling/backends/__init__.py::get_scheduler_backend` reads `settings.scheduler.backend` (`SCHEDULER_BACKEND`, default `local`) unless `backend_type` is passed, lowercases it, and lazy-imports the class. An unknown value raises `ConfigurationError`. The AWS branch fills region, ARN and prefix from `settings.aws` (see [AWS EventBridge backend](aws-eventbridge-backend.md)); the local branch is described in [local scheduler](local-scheduler.md).

## Railway shim

`packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend` is not a deployment target. `schedule_next_execution` and `cancel_scheduled_execution` only log. `get_scheduled_jobs` and `get_job_status` raise `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::BackendUnavailableError`, which `odds scheduler list-jobs` catches and prints as "not supported". Jobs still run their own gate logic under it, but cannot move their own cadence.

## Health and validation builders

`packages/odds-lambda/odds_lambda/scheduling/health_check.py` provides `HealthCheckBuilder` (`pass_check`, `fail_check`, `check_condition`, `build`) and `ValidationBuilder`. A backend is healthy when no check failed. `odds scheduler health` and `odds scheduler test-backend` call `get_scheduler_backend()` with the configured backend only; they do not accept `--backend`.

## Hazards

- `RailwayBackend.health_check` fails its "Running in Railway environment" check unless `RAILWAY_ENVIRONMENT` is set, so the backend reports unhealthy anywhere else (`packages/odds-lambda/odds_lambda/scheduling/backends/railway.py::RailwayBackend`).
- The `get_scheduler_backend` docstring calls Railway an "alternative production" backend; `docs/ARCHITECTURE.md#railway-backend` and the code say it is a no-op shim.
- `RetryConfig` is only consumed by the AWS backend, and only `max_attempts`, `initial_delay_seconds` and `max_delay_seconds`; `exponential_base` and `jitter` are read by nothing (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend`).
- Each `self_schedule` call builds a fresh backend (and boto3 clients on AWS); there is no shared instance.

## Related

- [Job deployment matrix](job-deployment-matrix.md) for which targets run which jobs.
- [Job registry](job-registry.md) for the names backends schedule.
