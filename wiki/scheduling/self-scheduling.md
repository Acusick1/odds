---
type: process
title: Self-scheduling and the decision engine
description: How each job decides and books its own next run, the forward/backward decisions, every job's cadence and overnight window, and the per-job failure policies that can break a chain.
sources:
  - docs/ARCHITECTURE.md#backend-interface
  - docs/ARCHITECTURE.md#intelligent-scheduling-system
  - docs/DEPLOYMENT.md#eventbridge-rules
  - docs/DEPLOYMENT.md#smoke-check
  - docs/DEVELOPMENT.md#adding-scheduler-jobs
  - docs/DEBUGGING_SCHEDULER.md#missing-tier-coverage
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::get_next_kickoff
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::apply_overnight_skip
  - packages/odds-lambda/odds_lambda/scheduling/backends/base.py::SchedulerBackend
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_forward
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_forward_resilient
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward_resilient
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::CadenceConfig
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LeagueSpec
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE
  - deployment/aws/terraform/scraper.tf
verified: "7887039"
status: stable
---

# Self-scheduling and the decision engine

There is no fixed external cron for the data jobs. Each run asks the decision engine when to run next and books that time through `SchedulerBackend.schedule_next_execution` ([backends](scheduler-backends.md)). On AWS that rewrites one EventBridge rule; locally it replaces an APScheduler entry ([local scheduler](local-scheduler.md)). The shared helper is `packages/odds-lambda/odds_lambda/scheduling/helpers.py::self_schedule(job_name, next_time, dry_run, sport, ...)`, which builds a backend, passes `{"sport": ...}` as payload and logs `job_next_scheduled`. The `job_name` must be the [compound name](jobs-and-deployment-matrix.md#per-sport-jobs-and-compound-names) so it matches the rule or schedule id.

## Pattern

1. Decide: `decide_forward_resilient` or `decide_backward_resilient`.
2. Pre-schedule the next run before any work, so a crash or Lambda timeout leaves the chain intact.
3. If the gate says skip, return.
4. Do the work (usually inside `job_alert_context`, see [Alerts and health](../operations/alerts-and-health.md)).
5. Re-decide and reschedule, because ingestion may have created events that move the tier.

`fetch-odds` syncs events from the free `/events` endpoint before deciding, so `get_next_kickoff` sees upstream fixtures ([Odds API ingestion](../collection/odds-api-ingestion.md)).

## The decision engine

`packages/odds-lambda/odds_lambda/scheduling/decision.py` maps proximity to a polling interval and a run/skip gate. Each function returns a frozen `ScheduleDecision(should_execute, reason, next_execution, tier)`.

### Forward decisions (odds, scrapes, Betfair, agent)

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`get_next_kickoff`). No kickoff within `lookahead_days` gives `should_execute=False` and `cadence.no_game`. Otherwise `_classify(hours_until)` picks a [fetch tier](../storage/fetch-tier.md) and the cadence interval.
- `decide_forward_resilient(sport_keys, cadence, ...)` uses the soonest kickoff across sports; if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- Always-run jobs (`fetch-oddsportal`, `fetch-betfair-exchange`) ignore `should_execute` and use only `next_execution`. So does `agent-run`, which uses `decide_forward_resilient` for its cadence and ignores the gate; a requested wake-up overrides it ([Betting agent](../agent/betting-agent.md)).

### Backward decisions (scores, status)

`decide_backward` runs while a game that kicked off within `window` is not yet resolved; `decide_backward_resilient` wraps it across sports with the same DB-failure fallback. Without `statuses_needing_update` any non-FINAL event counts; `update-status` passes `{SCHEDULED}`. An idle result still returns a `next_execution` at `idle_interval` ([Event lifecycle](../collection/event-lifecycle.md)).

### Per-job cadence (hours)

| Job | closing | pregame | sharp | early | opening | no game |
|---|---|---|---|---|---|---|
| `fetch-odds` | 0.5 | 3 | 12 | 24 | 48 | 24 |
| `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 2 |
| `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 2 |
| `agent-run` | 1 | 4 | 4 | 12 | 24 | 12 |

Values are each job module's `CADENCE` (`packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE` for the agent); they replace `interval_hours`, which nothing reads. `CadenceConfig.interval_for` maps `IN_PLAY` to `closing`. Scores: 3-day window, 6 h active, 12 h idle. Status: 4 h window, 1 h active, 6 h idle.

### Overnight skip

`OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC and `baseball_mlb` 05:00-14:00 UTC via `apply_overnight_skip`, which pushes a candidate time to the resume hour; a sport missing from the map raises. It applies only if the caller passes `overnight`: oddsportal passes its `LeagueSpec` window, Betfair passes `OVERNIGHT_WINDOWS` for single-sport runs, and `fetch-odds`, scores and status pass none. `lookahead_days` comes from `SchedulerConfig.lookahead_days` only in `fetch-odds`; the scrape and Betfair jobs use the function default of 7.

## Failure policy by job

| Job | Scheduling shape | If scheduling fails |
|---|---|---|
| `fetch-odds`, `fetch-scores`, `update-status` | pre and post | log and `send_error`, continue |
| `fetch-oddsportal`, `fetch-betfair-exchange` | pre and post | pre-schedule failure raises; post logs only |
| `fetch-oddsportal-results` | fixed 08:00 UTC next day in `finally`, no pre-schedule | log and `send_critical` |
| `check-health` | after the work only, +60 min, raw backend call | log and `send_error` |
| `fetch-polymarket` | after the work only, raw backend call | log and `send_error` |

## Hazards

- `fetch-oddsportal-results`, `check-health` and `fetch-polymarket` never pre-schedule. A hard kill skips `finally` for the results job (the scraper Lambda timeout is 900 s, `deployment/aws/terraform/scraper.tf`); an exception in `check_system_health` or in Polymarket's fetch body skips the reschedule, and a disabled or past-dated rule stays dead. The deploy workflows invoke every job after `terraform apply` to (re)start chains ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md), [CI/CD workflows](../operations/ci-cd-workflows.md)).
- Five different failure behaviours exist across these jobs and no code states which job class gets which; each job carries a copy of the `_schedule` closure (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::main`).
- A rule that was never rescheduled still shows `rate(1 day)` and is `DISABLED`; `scripts/verify-schedules.sh` fails such rules, disabled rules and cron years in the past ([AWS infrastructure](../operations/aws-infrastructure.md)).
- `get_next_kickoff` only sees future SCHEDULED events, so a sport with no stored future events sleeps for `no_game` hours even if fixtures exist upstream, and `FetchTier.IN_PLAY` is never produced by forward decisions.
- **Tier boundary.** `_classify` uses strict `<` on `FetchTier.max_hours`; `calculate_tier` uses `<=`. A kickoff exactly 3.0 h away is PREGAME here and CLOSING when stored ([Fetch tier](../storage/fetch-tier.md)).
- Overnight windows live in two places: `OVERNIGHT_WINDOWS` and the `overnight_*_utc` fields of `LeagueSpec` (`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LeagueSpec`).
- `decide_backward` counts every non-FINAL event, so a postponed or cancelled event keeps `fetch-scores` on its 6 h cadence for three days.
- Docstrings in `decision.py` refer to deleted engines (`SchedulingIntelligence`, `should_execute_scores`); searching for those names finds nothing.
- The smoke check only reaches a job's fetch body when `should_execute` is true ([Local scheduler](local-scheduler.md#cli)).

## Related

- [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md)
- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Fetch tier](../storage/fetch-tier.md)
