---
type: process
title: Self-scheduling and the decision engine
description: How each job decides and books its own next run, the forward/backward decisions, every job's cadence and overnight window, and the per-job failure policies that can break a chain.
sources:
  - odds_lambda.scheduling.helpers.self_schedule
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.scheduling.helpers.apply_overnight_skip
  - odds_lambda.scheduling.backends.base.SchedulerBackend
  - odds_lambda.scheduling.decision.decide_forward
  - odds_lambda.scheduling.decision.decide_forward_resilient
  - odds_lambda.scheduling.decision.decide_backward
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_lambda.scheduling.decision.CadenceConfig
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.jobs.fetch_oddsportal_results.main
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_lambda.jobs.agent_run.CADENCE
  - deployment/aws/terraform/scraper.tf
status: stable
---

# Self-scheduling and the decision engine

There is no fixed external cron for the data jobs. Each run asks the decision engine when to run next and books that time through `odds_lambda.scheduling.backends.base.SchedulerBackend.schedule_next_execution` ([backends](scheduler-backends.md)). On AWS that rewrites one EventBridge rule; locally it replaces an APScheduler entry ([local scheduler](local-scheduler.md)). The shared helper is `odds_lambda.scheduling.helpers.self_schedule(job_name, next_time, dry_run, sport, ...)`, which builds a backend, passes `{"sport": ...}` as payload and logs `job_next_scheduled`. The `job_name` must be the [compound name](jobs-and-deployment-matrix.md#per-sport-jobs-and-compound-names) so it matches the rule or schedule id.

## Pattern

1. Decide: `odds_lambda.scheduling.decision.decide_forward_resilient` or `odds_lambda.scheduling.decision.decide_backward_resilient`.
2. Pre-schedule the next run before any work, so a crash or Lambda timeout leaves the chain intact.
3. If the gate says skip, return.
4. Do the work (usually inside `odds_core.alerts.job_alert_context`, see [Alerts and health](../operations/alerts-and-health.md)).
5. Re-decide and reschedule, because ingestion may have created events that move the tier.

`fetch-odds` syncs events from the free `/events` endpoint before deciding, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees upstream fixtures ([Odds API ingestion](../collection/odds-api-ingestion.md)).

## The decision engine

`odds_lambda.scheduling.decision` maps proximity to a polling interval and a run/skip gate. Each function returns a frozen `ScheduleDecision(should_execute, reason, next_execution, tier)`.

### Forward decisions (odds, scrapes, Betfair, agent)

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`odds_lambda.scheduling.helpers.get_next_kickoff`). No kickoff within `lookahead_days` gives `should_execute=False` and `cadence.no_game`. Otherwise `_classify(hours_until)` picks a [fetch tier](../storage/fetch-tier.md) and the cadence interval.
- `decide_forward_resilient(sport_keys, cadence, ...)` uses the soonest kickoff across sports; if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- Always-run jobs (`fetch-oddsportal`, `fetch-betfair-exchange`) ignore `should_execute` and use only `next_execution`. So does `agent-run`, which uses `odds_lambda.scheduling.decision.decide_forward_resilient` for its cadence and ignores the gate; a requested wake-up overrides it ([Betting agent](../agent/betting-agent.md)).

### Backward decisions (scores, status)

`odds_lambda.scheduling.decision.decide_backward` runs while a game that kicked off within `window` is not yet resolved; `odds_lambda.scheduling.decision.decide_backward_resilient` wraps it across sports with the same DB-failure fallback. Without `statuses_needing_update` any non-FINAL event counts; `update-status` passes `{SCHEDULED}`. An idle result still returns a `next_execution` at `idle_interval` ([Event lifecycle](../collection/event-lifecycle.md)).

### Per-job cadence (hours)

| Job | closing | pregame | sharp | early | opening | no game |
|---|---|---|---|---|---|---|
| `fetch-odds` | 0.5 | 3 | 12 | 24 | 48 | 24 |
| `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 2 |
| `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 2 |
| `agent-run` | 1 | 4 | 4 | 12 | 24 | 12 |

Values are each job module's cadence constant (`odds_lambda.jobs.fetch_odds.CADENCE`, `odds_lambda.jobs.fetch_oddsportal.CADENCE`, `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`, `odds_lambda.jobs.agent_run.CADENCE`); they replace the `interval_hours` argument of `odds_lambda.scheduling.helpers.self_schedule`, which is only logged. `odds_lambda.scheduling.decision.CadenceConfig.interval_for` maps `IN_PLAY` to `closing`. Scores: 3-day window, 6 h active, 12 h idle. Status: 4 h window, 1 h active, 6 h idle.

### Overnight skip

`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC and `baseball_mlb` 05:00-14:00 UTC via `odds_lambda.scheduling.helpers.apply_overnight_skip`, which pushes a candidate time to the resume hour; a sport missing from the map raises. It applies only if the caller passes `overnight`: oddsportal passes its `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` window, Betfair passes `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` for single-sport runs, and `fetch-odds`, scores and status pass none. `lookahead_days` comes from `odds_core.config.SchedulerConfig.lookahead_days` only in `fetch-odds`; the scrape and Betfair jobs use the function default of 7.

## Failure policy by job

| Job | Scheduling shape | If scheduling fails |
|---|---|---|
| `fetch-odds`, `fetch-scores`, `update-status` | pre and post | log and `odds_core.alerts.send_error`, continue |
| `fetch-oddsportal`, `fetch-betfair-exchange` | pre and post | pre-schedule failure raises; post logs only |
| `fetch-oddsportal-results` | fixed 08:00 UTC next day in `finally`, no pre-schedule | log and `odds_core.alerts.send_critical` |
| `check-health` | after the work only, +60 min, raw backend call | log and `odds_core.alerts.send_error` |
| `fetch-polymarket` | after the work only, raw backend call | log and `odds_core.alerts.send_error` |

## Hazards

- `fetch-oddsportal-results`, `check-health` and `fetch-polymarket` never pre-schedule. A hard kill skips `finally` for the results job (the scraper Lambda timeout is 900 s, `deployment/aws/terraform/scraper.tf`); an exception in `odds_lambda.health_monitor.check_system_health` or in Polymarket's fetch body skips the reschedule, and a disabled or past-dated rule stays dead. The deploy workflows invoke every job after `terraform apply` to (re)start chains ([Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md), [CI/CD workflows](../operations/ci-cd-workflows.md)).
- Five different failure behaviours exist across these jobs and no code states which job class gets which; `odds_lambda.jobs.fetch_odds.main`, `odds_lambda.jobs.fetch_scores.main` and `odds_lambda.jobs.update_status.main` each define their own copy of a `odds_lambda.jobs.fetch_polymarket._schedule` closure, and `odds_lambda.jobs.fetch_polymarket._schedule` is a module-level variant.
- A rule that was never rescheduled still shows `rate(1 day)` and is `DISABLED`; `scripts/verify-schedules.sh` fails such rules, disabled rules and cron years in the past ([AWS infrastructure](../operations/aws-infrastructure.md)).
- `odds_lambda.scheduling.helpers.get_next_kickoff` only sees future SCHEDULED events, so a sport with no stored future events sleeps for `no_game` hours even if fixtures exist upstream, and `odds_lambda.fetch_tier.FetchTier.IN_PLAY` is never produced by forward decisions.
- **Tier boundary.** `odds_lambda.scheduling.decision._classify` uses strict `<` on `odds_lambda.fetch_tier.FetchTier.max_hours`; `odds_lambda.tier_utils.calculate_tier` uses `<=`. A kickoff exactly 3.0 h away is PREGAME here and CLOSING when stored ([Fetch tier](../storage/fetch-tier.md)).
- Overnight windows live in two places: `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` and the `overnight_*_utc` fields of `odds_lambda.jobs.fetch_oddsportal.LeagueSpec`.
- `odds_lambda.scheduling.decision.decide_backward` counts every non-FINAL event, so a postponed or cancelled event keeps `fetch-scores` on its 6 h cadence for three days.
- Docstrings in `odds_lambda.scheduling.decision` refer to deleted engines (`SchedulingIntelligence`, `should_execute_scores`); searching for those names finds nothing.
- The smoke check only reaches a job's fetch body when `should_execute` is true ([Local scheduler](local-scheduler.md#cli)).

## Related

- [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md)
- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md)
- [Add a scheduler job](howto-add-scheduler-job.md)
- [Fetch tier](../storage/fetch-tier.md)
