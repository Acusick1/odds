---
type: howto
title: Add a scheduler job
summary: Entry point, registry entry, self-scheduling, job_alert_context, per-sport or cron registration, local testing and the deployment-matrix update.
sources:
  - docs/DEVELOPMENT.md#adding-scheduler-jobs
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-core/odds_core/alerts.py::job_alert_context
verified_at: 7887039
---

# Add a scheduler job

Steps for a new job under `packages/odds-lambda/odds_lambda/jobs/`. Concepts: [job registry](job-registry.md), [self-scheduling](self-scheduling.md).

1. **Entry point.** Create `jobs/<name>.py` with `async def main(ctx: JobContext) -> None`. Read `ctx.sport` when the job is per sport and bail out on unsupported sports ([JobContext](job-registry.md)).
2. **Register.** Add `"<job-name>": ("odds_lambda.jobs.<name>", "main")` to `_JOB_MODULE_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`. If local bootstrap needs a different function, add it to `_JOB_BOOTSTRAP_MAP`.
3. **Per sport?** Add the name to `_PER_SPORT_JOBS`. Compound names then resolve via `_SPORT_SUFFIX_MAP` (`epl`, `mlb`); a new sport needs a suffix there or its compound name stays bare.
4. **Schedule shape.** Pick one:
   - Proximity-driven: use [`decide_forward_resilient`](scheduling-decision-engine.md) or `decide_backward_resilient` with a `CadenceConfig`, pre-schedule before the work, re-schedule after, via `self_schedule(job_name=make_compound_job_name(...), ...)`.
   - Fixed local cron: add `(cron, allowed_sports_or_None)` to `_JOB_CRON_MAP`. Use an allowlist for jobs whose `main` raises on other sports, or the daily fire raises inside `job_alert_context` and alerts every day.
5. **Alerts.** Wrap the work, not the scheduling, in `async with job_alert_context(compound_name):` so failures send a rate-limited critical and success writes `heartbeat:<compound_name>` ([alerts module](alerts-module.md)). If you want a missed-run alert, add the same compound name to `AlertConfig.heartbeat_expectations` in `packages/odds-core/odds_core/config.py`; nothing validates the string match ([health monitor](health-monitor.md)).
6. **Test locally.** `uv run odds scheduler run-once <job>` (compound names work), then `uv run odds scheduler start` ([local scheduler](local-scheduler.md)). Add the job to `SCHEDULER_BOOTSTRAP_JOBS` to include it in `odds scheduler smoke`.
7. **AWS (optional).** Add the job to the Terraform job lists in `deployment/aws/terraform/eventbridge.tf` so a placeholder rule exists; rule name must equal the compound name ([EventBridge backend](aws-eventbridge-backend.md)).
8. **Docs.** Update the job matrix in `docs/DEPLOYMENT.md#job-deployment-matrix` ([job deployment matrix](job-deployment-matrix.md)) and add a wiki page for the job.

## Gotchas

- Pre-schedule first; a job that only schedules after its work loses its chain on any exception. Several existing jobs do exactly this ([hazards](self-scheduling.md#hazards)).
- Registering the job does not give it an EventBridge rule or a local schedule; both are explicit.
- `docs/DEVELOPMENT.md#adding-scheduler-jobs` omits `_JOB_BOOTSTRAP_MAP`, the heartbeat expectation and Terraform steps.
- Keep imports of heavy dependencies inside functions: the registry imports lazily so the scraper image does not need xgboost, and your module must not break that.
