---
type: component
title: update-status job
summary: Marks SCHEDULED events that have started as LIVE, so pre-game fetching stops.
sources:
  - docs/ARCHITECTURE.md#update-status-job
  - packages/odds-lambda/odds_lambda/jobs/update_status.py::main
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward_resilient
verified_at: 7887039
---

# update-status job

`packages/odds-lambda/odds_lambda/jobs/update_status.py::main(ctx)` flips SCHEDULED events whose `commence_time` has passed to LIVE. FINAL is set elsewhere ([fetch-scores](job-fetch-scores.md) and [fetch-oddsportal-results](job-fetch-oddsportal-results.md)). The lifecycle is in [event lifecycle](event-lifecycle.md). Because the [fetch-odds gate](job-fetch-odds.md) keys off SCHEDULED events only, flipping to LIVE removes a started game from pre-game proximity decisions.

## Deployment

A global (not per-sport) job: one rule `odds-update-status` on the scheduler Lambda, always created by Terraform, not in the default local `bootstrap_jobs` ([deployment matrix](job-deployment-matrix.md)). It runs for `data_collection.sports` when no sport is given.

## Flow

1. `_status_decision` calls `decide_backward_resilient` with a 4-hour window, 1 h active and 6 h idle intervals, counting only `{EventStatus.SCHEDULED}` ([decision engine](scheduling-decision-engine.md)).
2. Pre-schedule under `update-status`, skip if idle ([self-scheduling](self-scheduling.md)).
3. Inside `job_alert_context("update-status")`, `_update_event_statuses` reads SCHEDULED events with `commence_time` in the last four hours and calls `OddsWriter.update_event_status(LIVE)` for each that has started ([odds writer](odds-writer.md)).
4. Re-decide and post-schedule.

## Hazards

- `_update_event_statuses` does not filter by sport, while the gate does; when a per-sport schedule fires, it flips events of every sport. Harmless in the current single global rule, but the gate and the action disagree on scope.
- Only events that kicked off within the last 4 hours are eligible. If the job was down longer, older SCHEDULED events stay SCHEDULED until a results job marks them FINAL.
- The function docstring cites a nonexistent scheduler jobs module (old path scheduler/jobs) as its origin.
- Heartbeat key is `heartbeat:update-status`, which is not in the default `heartbeat_expectations` ([health monitor](health-monitor.md)).

## Related

- [Job registry](job-registry.md).
