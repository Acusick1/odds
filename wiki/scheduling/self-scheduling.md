---
type: explanation
title: Self-scheduling
description: Why jobs book their own next run instead of a fixed cron, the pre-schedule, work and re-schedule pattern, and how each job's failure policy can end its chain.
sources:
  - odds_lambda.scheduling.helpers.self_schedule
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.scheduling.backends.base.SchedulerBackend.schedule_next_execution
  - odds_lambda.scheduling.decision.decide_forward_resilient
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.jobs.fetch_scores.main
  - odds_lambda.jobs.update_status.main
  - odds_lambda.jobs.fetch_oddsportal_results.main
  - odds_lambda.jobs.fetch_polymarket._schedule
  - odds_lambda.health_monitor.check_system_health
  - deployment/aws/terraform/scraper.tf
status: stable
---

# Self-scheduling

There is no fixed external cron for the data jobs. Each run asks the [decision engine](decision-engine.md) when to run next and books that time through `odds_lambda.scheduling.backends.base.SchedulerBackend.schedule_next_execution` ([backends](scheduler-backends.md)). On AWS that rewrites one EventBridge rule; locally it replaces an APScheduler entry ([local scheduler](local-scheduler.md)). The shared helper is `odds_lambda.scheduling.helpers.self_schedule`, which builds a backend, passes `{"sport": ...}` as payload and logs `job_next_scheduled`. Its `job_name` must be the [compound name](job-registry.md#compound-job-names) so it matches the rule or schedule id.

The cadence of a data job follows how close the next kickoff is, which only the job's own run can see (the decision functions read the stored events). A job therefore picks its next time at the end of each run instead of keeping a fixed rate `(inferred)`.

## Pattern

1. Decide: `odds_lambda.scheduling.decision.decide_forward_resilient` or `odds_lambda.scheduling.decision.decide_backward_resilient`.
2. Pre-schedule the next run before any work, so a crash or Lambda timeout leaves the chain intact.
3. If the gate says skip, return.
4. Do the work (usually inside `odds_core.alerts.job_alert_context`, see [Alerts](../operations/alerts.md)).
5. Re-decide and reschedule, because ingestion may have created events that move the tier.

`fetch-odds` syncs events from the free `/events` endpoint before deciding, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees upstream fixtures ([fetch-odds job](../collection/fetch-odds-job.md)).

## Failure policy by job

Five different failure behaviours exist across the jobs, and no code states which job class gets which.

| Job | Scheduling shape | If scheduling fails |
|---|---|---|
| `fetch-odds`, `fetch-scores`, `update-status` | pre and post | log and `odds_core.alerts.send_error`, continue |
| `fetch-oddsportal`, `fetch-betfair-exchange` | pre and post | pre-schedule failure raises; post logs only |
| `fetch-oddsportal-results` | fixed 08:00 UTC next day in `finally`, no pre-schedule | log and `odds_core.alerts.send_critical` |
| `check-health` | after the work only, +60 min, raw backend call | log and `odds_core.alerts.send_error` |
| `fetch-polymarket` | after the work only, raw backend call | log and `odds_core.alerts.send_error` |

`fetch-oddsportal-results`, `check-health` and `fetch-polymarket` schedule only after their work, so a kill or exception ends the chain (#438). A hard kill skips the results job's `finally` (the scraper Lambda timeout is 900 s, `deployment/aws/terraform/scraper.tf`). An exception in `odds_lambda.health_monitor.check_system_health` or in Polymarket's fetch body skips the reschedule. A rule that was never rescheduled stays `DISABLED` on its `rate(1 day)` placeholder; see [Deployed state](job-deployment-matrix.md#deployed-state) for how to find those. The deploy workflows invoke every job after `terraform apply` to (re)start chains ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md), [CI/CD workflows](../operations/ci-cd-workflows.md)).

`odds_lambda.jobs.fetch_odds.main`, `odds_lambda.jobs.fetch_scores.main` and `odds_lambda.jobs.update_status.main` each define their own copy of a `odds_lambda.jobs.fetch_polymarket._schedule` closure, and `odds_lambda.jobs.fetch_polymarket._schedule` is a module-level variant (#463).

## Related

- [Decision engine](decision-engine.md)
- [Job registry](job-registry.md)
- [Where each job runs](job-deployment-matrix.md)
- [Scheduler backends](scheduler-backends.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
