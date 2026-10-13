---
type: howto
title: Add a scheduler job
description: Steps to add a job (entry point, registry, per-sport or cron registration, self-scheduling, alert wrapper and heartbeat, local test, Terraform rule and matrix update) and the traps in each.
sources:
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_BOOTSTRAP_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_PER_SPORT_JOBS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-core/odds_core/alerts.py::job_alert_context
  - packages/odds-core/odds_core/config.py::AlertConfig
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Add a scheduler job

Steps for a new job under `packages/odds-lambda/odds_lambda/jobs/`. Concepts: [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md), [Self-scheduling and the decision engine](self-scheduling.md).

1. **Entry point.** Create `jobs/<name>.py` with `async def main(ctx: JobContext) -> None`. Read `ctx.sport` when the job is per sport and bail out on unsupported sports.
2. **Register.** Add `"<job-name>": ("odds_lambda.jobs.<name>", "main")` to `_JOB_MODULE_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`. If local bootstrap needs a different function, add it to `_JOB_BOOTSTRAP_MAP`.
3. **Per sport?** Add the name to `_PER_SPORT_JOBS`. Compound names then resolve via `_SPORT_SUFFIX_MAP` (`epl`, `mlb`); a new sport needs a suffix there or its compound name stays bare.
4. **Schedule shape.** Pick one:
   - Proximity-driven: use `packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_forward_resilient` or `packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward_resilient` with a `packages/odds-lambda/odds_lambda/scheduling/decision.py::CadenceConfig`, pre-schedule before the work, re-schedule after, via `self_schedule(job_name=make_compound_job_name(...), ...)`. Choose a failure policy from the [table](self-scheduling.md#failure-policy-by-job); none is enforced for you.
   - Fixed local cron: add `(cron, allowed_sports_or_None)` to `_JOB_CRON_MAP`. Use an allowlist for jobs that raise on other sports, or the daily fire raises inside `packages/odds-core/odds_core/alerts.py::job_alert_context` and alerts every day.
5. **Alerts.** Wrap the work, not the scheduling, in `async with job_alert_context(compound_name):` so failures send a rate-limited critical and success writes `heartbeat:<compound_name>` ([Alerts and health](../operations/alerts-and-health.md#heartbeats)). To get a missed-run alert, add the same compound name to `packages/odds-core/odds_core/config.py::AlertConfig.heartbeat_expectations` in `packages/odds-core/odds_core/config.py`. Nothing validates the string match: a key mismatch passes silently, and a job with no heartbeat history is never alerted.
6. **Test locally.** `uv run odds scheduler run-once <job>` (compound names work), then `uv run odds scheduler start` ([Local scheduler](local-scheduler.md)). Add the job to `SCHEDULER_BOOTSTRAP_JOBS` to include it in `odds scheduler smoke`.
7. **AWS (optional).** Add the job to the Terraform job lists in `deployment/aws/terraform/eventbridge.tf` so a placeholder rule exists. The rule name must equal the compound name, or the reschedule goes nowhere ([Scheduler backends](scheduler-backends.md#aws-eventbridge-backend)).
8. **Docs.** Update [the matrix](jobs-and-deployment-matrix.md#where-each-job-runs) and add a wiki page for the job.

## Hazards

- Pre-schedule first; a job that only schedules after its work loses its chain on any exception. Several existing jobs do exactly this ([hazards](self-scheduling.md#hazards)).
- Registering the job does not give it an EventBridge rule or a local schedule; both are explicit.
- Keep imports of heavy dependencies inside functions: the registry imports lazily and the scraper image has no xgboost or analytics, so your module must not import them at module level ([Packages and the odds CLI](../foundations/packages-and-cli.md#hazards)).

## Related

- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)
- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md)
