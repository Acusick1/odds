---
type: process
title: Self-scheduling
summary: Each job invocation computes and books its own next run (an EventBridge rule or an APScheduler entry), pre-schedules so the chain survives a crash, and follows a per-job failure policy.
sources:
  - docs/ARCHITECTURE.md#backend-interface
  - docs/DEPLOYMENT.md#eventbridge-rules
  - docs/DEVELOPMENT.md#adding-scheduler-jobs
  - docs/DEBUGGING_SCHEDULER.md#missing-tier-coverage
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::get_next_kickoff
  - packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::main
verified_at: 7887039
---

# Self-scheduling

There is no fixed external cron for the data jobs. Each run asks the [decision engine](scheduling-decision-engine.md) when to run next and books that time through `SchedulerBackend.schedule_next_execution` ([backends](scheduler-backends.md)). On AWS that rewrites one [EventBridge rule](aws-eventbridge-backend.md); locally it replaces an APScheduler entry ([local scheduler](local-scheduler.md)). The shared helper is `packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule(job_name, next_time, dry_run, sport, ...)`, which builds a backend, passes `{"sport": ...}` as payload and logs `job_next_scheduled`. The `job_name` must be the [compound name](job-registry.md) so it matches the rule or schedule id.

## Pattern

1. Decide: `decide_forward_resilient` or `decide_backward_resilient`.
2. Pre-schedule the next run before any work, so a crash or Lambda timeout leaves the chain intact.
3. If the gate says skip, return.
4. Do the work (usually inside `job_alert_context`, see [alerts module](alerts-module.md)).
5. Re-decide and reschedule, because ingestion may have created events that move the tier.

## Failure policy by job

| Job | Scheduling shape | If scheduling fails |
|---|---|---|
| `fetch-odds`, `fetch-scores`, `update-status` | pre and post | log and `send_error`, continue |
| `fetch-oddsportal`, `fetch-betfair-exchange` | pre and post | pre-schedule failure raises; post logs only |
| `fetch-oddsportal-results` | fixed 08:00 UTC next day in `finally`, no pre-schedule | log and `send_critical` |
| `check-health` | after the work only, +60 min, raw backend call | log and `send_error` |
| `fetch-polymarket` | after the work only, raw backend call | log and `send_error` |

## Hazards

- `fetch-oddsportal-results`, `check-health` and `fetch-polymarket` never pre-schedule. A hard kill (Lambda timeout) skips `finally` for the results job; an exception in `check_system_health` or in Polymarket's fetch body skips the reschedule, and a disabled or past-dated rule stays dead. The deploy workflows re-invoke jobs after `terraform apply` to restart chains ([debug the AWS scheduler](howto-debug-aws-scheduler.md)).
- Five different failure behaviours exist across these jobs and no code states which job class gets which; each job carries a copy of the `_schedule` closure (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`).
- A rule that was never rescheduled still shows `rate(1 day)` and is `DISABLED`; `scripts/verify-schedules.sh` fails such rules, disabled rules and cron years in the past ([CI/CD workflows](ci-cd-workflows.md)).
- `get_next_kickoff` only sees SCHEDULED events, so a sport with no stored future events sleeps for `no_game` hours even if fixtures exist upstream ([fetch-odds job](job-fetch-odds.md) syncs events first to avoid this).

## Related

- [Add a scheduler job](howto-add-scheduler-job.md), [fetch tier](fetch-tier.md), [lambda handler](lambda-handler.md).
