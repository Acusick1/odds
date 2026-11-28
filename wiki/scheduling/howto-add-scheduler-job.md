---
type: howto
title: Add a scheduler job
description: Steps to add a job (entry point, registry, per-league, league allowlist or cron registration, self-scheduling, alert wrapper and heartbeat, production bootstrap and matrix update) and the traps in each.
sources:
  - odds_core.job_specs.JOB_SPECS
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.alerts.job_alert_context
  - odds_core.config.AlertConfig
  - deployment/aws/terraform/eventbridge.tf
mentions:
  - odds_lambda.health_monitor.HealthMonitor
status: stable
---

# Add a scheduler job

Steps for a new job under `packages/odds-lambda/odds_lambda/jobs/`. Concepts: [Job registry](job-registry.md), [Where each job runs](job-deployment-matrix.md), [Self-scheduling](self-scheduling.md).

1. **Entry point.** Create `jobs/<name>.py` with `async def main(ctx: JobContext) -> None`. When the job is per league, call `odds_lambda.scheduling.jobs.require_sport(ctx, "<job-name>")` first (a run without a sport must fail, not run every league) and bail out on unsupported leagues. The registry imports job modules lazily by name ([Job registry](job-registry.md#registry)).
2. **Register.** Add one `odds_core.job_specs.JobSpec` to `odds_core.job_specs.JOB_SPECS`, with the lazy module name and entry point, its `cron` or its `cadence` (one interval per fetch tier), its `sports` allowlist and its `heartbeats` as the job needs. Set its bootstrap override when startup should book a run instead of doing the work. Mark its smoke side effects with `agent_bets`, `candidate_bets` or `paid_api` so deploy smoke requires the corresponding opt-in.
3. **Per league?** Set `per_sport` and, when restricted, `sports` on the record. Only configured leagues in that allowlist get scheduled. A new league also needs a suffix in `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` (which holds `epl`, `mlb`, `nba` and the football leagues' `championship`, `laliga`, `seriea`, `bundesliga` and `ligue1`); see [Add a league](../collection/howto-add-sport.md).
4. **Schedule shape.** Pick one:
   - Proximity-driven: use `odds_lambda.scheduling.decision.decide_forward_resilient` or `odds_lambda.scheduling.decision.decide_backward_resilient` with cadence values selected from `odds_core.job_specs.job_cadence`, pre-schedule before the work, re-schedule after, via `self_schedule(job_name=make_compound_job_name(...), ...)`. Pre-schedule first: a job that schedules only after its work loses its chain on any exception, as `check-health` does (#438). Use `odds_lambda.scheduling.collection.run_collection` for an odds collector and choose its postbooking policy from the [table](self-scheduling.md#failure-policy-by-job). Pass `leads` and the run's start time as `run_started` to the decision when the job must also run a set time before each kick-off ([lead runs](decision-engine.md#lead-runs)). A job that creates its own events must not gate on stored events alone: decide through `odds_lambda.collection_calendar.decide_collection_run` (`stored_only` when it only pairs with others' events, `stored_kickoff` when its kick-offs are not the events' `commence_time`), and when `should_execute` is false write the heartbeat and return ([dormancy](decision-engine.md#dormancy)).
   - Fixed local cron: set `cron` on the job record. The two shapes are exclusive: bootstrap re-applies every cron on start, and the local backend refuses a self-schedule for a job with a cron ([bootstrap](local-scheduler.md#bootstrap-jobs-and-cron-table)).
5. **Alerts.** Wrap the work, not the scheduling, in `async with job_alert_context(compound_name):` so a failure sends a critical alert, once per outage, and success writes `heartbeat:<compound_name>` ([Alerting and heartbeats](../operations/alerting-and-heartbeats.md#heartbeats), [dedup](../operations/alerts.md#alert-history)).
   - For a missed-run alert, add the same compound name to the record's `heartbeats` (one `odds_core.job_specs.Heartbeat` per league suffix for a per-league job), with the allowed hours between runs: longer than any overnight pause, and sized to the check-in for a dormant job, which still writes its heartbeat. The current values are in the [heartbeat table](../operations/health-monitor.md#heartbeats). Nothing checks that the names match, and a job with no heartbeat history is never alerted.
   - A job that writes snapshots also declares an `odds_lambda.collection_calendar.CollectionSource`, calls `odds_lambda.collection_calendar.require_rows_for_games` inside the alert context ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)), and adds the source to `odds_lambda.health_readers.collection_sources` for the [stale-data check](../operations/health-monitor.md#stale-data-check) and the web UI's status strip.
   - A check tied to one job's output is a separate method of `odds_lambda.health_monitor.HealthMonitor`, called from `odds_lambda.health_monitor.HealthMonitor.check_system_health` with its clock ([Betfair close check](../operations/health-monitor.md#betfair-close-check)). A check of what a job found wrong, rather than of its output, reads a record the job writes for it ([candidate check](../operations/health-monitor.md#candidate-check)).
   - A bootstrapped self-scheduling job is covered by the [schedule chain check](../operations/health-monitor.md#schedule-chain-check). If a setting can switch it off, set the record's `enabled` predicate on that setting and have `main` return early when `odds_lambda.scheduling.jobs.is_job_enabled` is false, so the check skips the job while it is off ([registry](job-registry.md#registry)).
   - The job must finish within `odds_lambda.scheduling.runner.JOB_TIMEOUT` ([Local scheduler](local-scheduler.md#scheduling-primitives)). Once it runs, its durations against the time between its runs are on the web UI's [performance page](../operations/odds-web.md#performance-page).
6. **Test locally.** `uv run odds scheduler run-once <job>` (compound names work; a global job such as `fetch-mlb-results` takes its bare name), then `uv run odds scheduler start` with the job in `SCHEDULER_BOOTSTRAP_JOBS` ([Local scheduler](local-scheduler.md#cli)).
7. **Run it in production.** Set `bootstrap=True` on its record and redeploy the production host with [deploy.sh](howto-run-scheduler-service.md#redeploy). Registering the job in step 2 creates no schedule, and a registered job left out of the list runs only when `SCHEDULER_BOOTSTRAP_JOBS` names it. The bootstrap list also makes `odds scheduler smoke` run the job on every deploy, unless its record marks smoke side effects. The AWS stack is dormant, so a Terraform rule in `deployment/aws/terraform/eventbridge.tf` is needed only to move production back to AWS; its name must equal the compound name ([Scheduler backends](scheduler-backends.md#rules-and-terraform)).
8. **Docs.** Update [the matrix](job-deployment-matrix.md#where-each-job-runs) and add a wiki page for the job.

## Related

- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
