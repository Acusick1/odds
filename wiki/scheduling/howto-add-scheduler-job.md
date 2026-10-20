---
type: howto
title: Add a scheduler job
description: Steps to add a job (entry point, registry, per-sport, sport allowlist or cron registration, self-scheduling, alert wrapper and heartbeat, production bootstrap and matrix update) and the traps in each.
sources:
  - odds_lambda.scheduling.jobs._JOB_MODULE_MAP
  - odds_lambda.scheduling.jobs._JOB_BOOTSTRAP_MAP
  - odds_lambda.scheduling.jobs._PER_SPORT_JOBS
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.alerts.job_alert_context
  - odds_core.config.AlertConfig
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Add a scheduler job

Steps for a new job under `packages/odds-lambda/odds_lambda/jobs/`. Concepts: [Job registry](job-registry.md), [Where each job runs](job-deployment-matrix.md), [Self-scheduling](self-scheduling.md).

1. **Entry point.** Create `jobs/<name>.py` with `async def main(ctx: JobContext) -> None`. Read `ctx.sport` when the job is per sport and bail out on unsupported sports. Keep imports of heavy dependencies inside functions: the registry imports lazily and the scraper image has no xgboost or analytics, so the module must not import them at module level (#454, [Workspace packages](../foundations/workspace-packages.md)).
2. **Register.** Add `"<job-name>": ("odds_lambda.jobs.<name>", "main")` to `_JOB_MODULE_MAP` in `odds_lambda.scheduling.jobs`. If local bootstrap needs a different function, add it to `_JOB_BOOTSTRAP_MAP`. Per-sport jobs such as `fetch-polymarket-odds` also belong in `_PER_SPORT_JOBS` and `_JOB_SPORTS`.
3. **Per sport?** Add the name to `_PER_SPORT_JOBS`. Compound names then resolve via `_SPORT_SUFFIX_MAP` (`epl`, `mlb`); a new sport needs a suffix there or its compound name stays bare. If the job raises on some sports or should not spend resources on every configured sport, add an allowlist to `_JOB_SPORTS`; only configured sports that are in it get scheduled. Derive the allowlist from the sport map the job reads when one exists (`fetch-betfair-exchange` and `fetch-polymarket-odds` do), so a new sport needs one edit. Without one, a job that raises on a sport alerts on every fire inside `odds_core.alerts.job_alert_context`.
4. **Schedule shape.** Pick one:
   - Proximity-driven: use `odds_lambda.scheduling.decision.decide_forward_resilient` or `odds_lambda.scheduling.decision.decide_backward_resilient` with a `odds_lambda.scheduling.decision.CadenceConfig`, pre-schedule before the work, re-schedule after, via `self_schedule(job_name=make_compound_job_name(...), ...)`. Pre-schedule first: a job that schedules only after its work loses its chain on any exception, as three existing jobs do (#438). Choose a failure policy from the [table](self-scheduling.md#failure-policy-by-job); none is enforced for you. Set `close_lead` on the cadence when the job must also run just before each kickoff ([close run](decision-engine.md#close-run)). A job that creates its own events must not gate on stored events alone: decide through `odds_lambda.collection_calendar.decide_collection_run` (`stored_only` when it only pairs with others' events), and when `should_execute` is false write the heartbeat and return ([dormancy](decision-engine.md#dormancy)).
   - Fixed local cron: add `"<job-name>": "<cron>"` to `_JOB_CRON_MAP`. The two shapes are exclusive: bootstrap re-applies every cron on start, and the local backend refuses a self-schedule for a job with a cron ([bootstrap](local-scheduler.md#bootstrap-jobs-and-cron-table)).
5. **Alerts.** Wrap the work, not the scheduling, in `async with job_alert_context(compound_name):` so a failure sends a rate-limited critical and success writes `heartbeat:<compound_name>` ([Alerting and heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)). To get a missed-run alert, add the same compound name, with the allowed hours between runs (longer than the overnight pause for jobs that have one; `fetch-mlb-probables` uses 30 h and `fetch-espn-fixtures` 26 h for their daily runs; a dormant job still writes its heartbeat each check-in, so size the expectation to the check-in), to `odds_core.config.AlertConfig.heartbeat_expectations`. Alerts are silent under `SCHEDULER_DRY_RUN`, so the deploy smoke never pages. Nothing validates the string match: a key mismatch passes silently, and a job with no heartbeat history is never alerted. A job that writes snapshots also declares an `odds_lambda.collection_calendar.CollectionSource` (snapshot source tag, cadence, per-sport fixture horizons) and calls `odds_lambda.collection_calendar.require_rows_for_fixtures` inside the alert context, so a run that stores nothing while fixtures are scheduled fails, and adds the source to `odds_lambda.health_monitor.collection_sources` for the per-source stale-data check ([Stale-data check](../operations/health-monitor.md#stale-data-check)). Checks tied to one job's output, such as the EPL close of `fetch-betfair-exchange` or unsettled events, are separate methods of `odds_lambda.health_monitor.HealthMonitor`, called from `odds_lambda.health_monitor.HealthMonitor.check_system_health` with its clock, that report after the fact ([Betfair close check](../operations/health-monitor.md#betfair-close-check)). A bootstrapped self-scheduling job is covered by the [schedule chain check](../operations/health-monitor.md#schedule-chain-check); if it returns without booking a run while disabled by a setting, add that setting to `odds_lambda.health_monitor._JOB_ENABLED`. The local scheduler cancels a run that exceeds `odds_lambda.scheduling.runner.JOB_TIMEOUT` and re-books a failed run that booked nothing, so the job must finish within the timeout ([Local scheduler](local-scheduler.md#scheduling-primitives)).
6. **Test locally.** `uv run odds scheduler run-once <job>` (compound names work), then `uv run odds scheduler start` with the job in `SCHEDULER_BOOTSTRAP_JOBS` ([Local scheduler](local-scheduler.md#cli)).
7. **Run it in production.** Add the job to the default `odds_core.config.SchedulerConfig.bootstrap_jobs` and redeploy the mini-PC with [deploy.sh](howto-run-scheduler-service.md#redeploy). Registering the job in step 2 creates no schedule, and a registered job left out of the list runs only when `SCHEDULER_BOOTSTRAP_JOBS` names it. The bootstrap list also makes `odds scheduler smoke` run the job on every deploy. The AWS stack is dormant, so a Terraform rule in `deployment/aws/terraform/eventbridge.tf` is needed only to move production back to AWS; its name must equal the compound name ([Scheduler backends](scheduler-backends.md#rules-and-terraform)).
8. **Docs.** Update [the matrix](job-deployment-matrix.md#where-each-job-runs) and add a wiki page for the job.

## Related

- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
