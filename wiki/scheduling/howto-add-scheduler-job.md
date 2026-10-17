---
type: howto
title: Add a scheduler job
description: Steps to add a job (entry point, registry, per-sport or cron registration, self-scheduling, alert wrapper and heartbeat, local test, Terraform rule and matrix update) and the traps in each.
sources:
  - odds_lambda.scheduling.jobs._JOB_MODULE_MAP
  - odds_lambda.scheduling.jobs._JOB_BOOTSTRAP_MAP
  - odds_lambda.scheduling.jobs._PER_SPORT_JOBS
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_core.alerts.job_alert_context
  - odds_core.config.AlertConfig
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Add a scheduler job

Steps for a new job under `packages/odds-lambda/odds_lambda/jobs/`. Concepts: [Job registry](job-registry.md), [Where each job runs](job-deployment-matrix.md), [Self-scheduling](self-scheduling.md).

1. **Entry point.** Create `jobs/<name>.py` with `async def main(ctx: JobContext) -> None`. Read `ctx.sport` when the job is per sport and bail out on unsupported sports. Keep imports of heavy dependencies inside functions: the registry imports lazily and the scraper image has no xgboost or analytics, so the module must not import them at module level (#454, [Workspace packages](../foundations/workspace-packages.md)).
2. **Register.** Add `"<job-name>": ("odds_lambda.jobs.<name>", "main")` to `_JOB_MODULE_MAP` in `odds_lambda.scheduling.jobs`. If local bootstrap needs a different function, add it to `_JOB_BOOTSTRAP_MAP`.
3. **Per sport?** Add the name to `_PER_SPORT_JOBS`. Compound names then resolve via `_SPORT_SUFFIX_MAP` (`epl`, `mlb`); a new sport needs a suffix there or its compound name stays bare.
4. **Schedule shape.** Pick one:
   - Proximity-driven: use `odds_lambda.scheduling.decision.decide_forward_resilient` or `odds_lambda.scheduling.decision.decide_backward_resilient` with a `odds_lambda.scheduling.decision.CadenceConfig`, pre-schedule before the work, re-schedule after, via `self_schedule(job_name=make_compound_job_name(...), ...)`. Pre-schedule first: a job that schedules only after its work loses its chain on any exception, as three existing jobs do (#438). Choose a failure policy from the [table](self-scheduling.md#failure-policy-by-job); none is enforced for you.
   - Fixed local cron: add `(cron, allowed_sports_or_None)` to `_JOB_CRON_MAP`. Use an allowlist for jobs that raise on other sports, or the daily fire raises inside `odds_core.alerts.job_alert_context` and alerts every day.
5. **Alerts.** Wrap the work, not the scheduling, in `async with job_alert_context(compound_name):` so failures send a rate-limited critical and success writes `heartbeat:<compound_name>` ([Alerting and heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)). To get a missed-run alert, add the same compound name to `odds_core.config.AlertConfig.heartbeat_expectations` in `odds_core.config`. Nothing validates the string match: a key mismatch passes silently, and a job with no heartbeat history is never alerted.
6. **Test locally.** `uv run odds scheduler run-once <job>` (compound names work), then `uv run odds scheduler start` ([Local scheduler](local-scheduler.md#cli)). Add the job to `SCHEDULER_BOOTSTRAP_JOBS` to include it in `odds scheduler smoke`.
7. **AWS (optional).** Add the job to the Terraform job lists in `deployment/aws/terraform/eventbridge.tf` so a placeholder rule exists. Registering the job in step 2 creates no EventBridge rule or local schedule, so this step and the local bootstrap in step 6 are both explicit. The rule name must equal the compound name, or the reschedule goes nowhere ([Scheduler backends](scheduler-backends.md#rules-and-terraform)).
8. **Docs.** Update [the matrix](job-deployment-matrix.md#where-each-job-runs) and add a wiki page for the job.

## Related

- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
