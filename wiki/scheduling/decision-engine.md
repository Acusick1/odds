---
type: reference
title: Scheduling decision engine
description: decide_forward and decide_backward and their resilient wrappers, CadenceConfig and per-job cadences, overnight windows, and the self_schedule and kickoff helpers.
sources:
  - odds_lambda.scheduling.decision.decide_forward
  - odds_lambda.scheduling.decision.decide_forward_resilient
  - odds_lambda.scheduling.decision.decide_backward
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_lambda.scheduling.decision.CadenceConfig
  - odds_lambda.scheduling.decision.ScheduleDecision
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.scheduling.decision._classify
  - odds_lambda.scheduling.helpers.self_schedule
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.scheduling.helpers.apply_overnight_skip
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.jobs.agent_run.CADENCE
  - odds_lambda.jobs.agent_run.schedule_next
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_core.config.SchedulerConfig.lookahead_days
status: stable
---

# Scheduling decision engine

`odds_lambda.scheduling.decision` maps proximity to a polling interval and a run/skip gate. Each function returns a frozen `odds_lambda.scheduling.decision.ScheduleDecision(should_execute, reason, next_execution, tier)`. Jobs call it as the first step of [self-scheduling](self-scheduling.md).

## Helpers

- `odds_lambda.scheduling.helpers.self_schedule(job_name, next_time, dry_run, sport, ...)` builds a backend and books `next_time` with `{"sport": ...}` as payload ([backends](scheduler-backends.md)). Each call builds a fresh backend ([backends](scheduler-backends.md#the-abc)).
- `odds_lambda.scheduling.helpers.get_next_kickoff` sees only stored future SCHEDULED events, so a sport with no stored future events sleeps for `no_game` hours even if fixtures exist upstream, and `odds_lambda.fetch_tier.FetchTier.IN_PLAY` is never produced by forward decisions.

## Forward decisions

Used by odds, scrapes, Betfair and the agent.

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`odds_lambda.scheduling.helpers.get_next_kickoff`). No kickoff within `lookahead_days` gives `should_execute=False` and `cadence.no_game`. Otherwise `odds_lambda.scheduling.decision._classify(hours_until)` picks a [fetch tier](../storage/fetch-tier.md) and the cadence interval. `odds_lambda.scheduling.decision._classify` uses strict `<` on `odds_lambda.fetch_tier.FetchTier.max_hours` where stored tiers use `<=`, so a kickoff exactly 3.0 h away is PREGAME here and CLOSING when stored (#446, [boundary encodings](../storage/fetch-tier.md#boundary-encodings)).
- `odds_lambda.scheduling.decision.decide_forward_resilient(sport_keys, cadence, ...)` uses the soonest kickoff across sports; if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- Always-run jobs (`fetch-oddsportal`, `fetch-betfair-exchange`) ignore `should_execute` and use only `next_execution`. So does `agent-run`, which uses `odds_lambda.scheduling.decision.decide_forward_resilient` for its cadence and ignores the gate; a requested wake-up overrides it ([wake-up requests](../agent/agent-run-job.md#wake-up-requests)).

## Backward decisions

Used by scores and status.

`odds_lambda.scheduling.decision.decide_backward` runs while a game that kicked off within `window` is not yet resolved; `odds_lambda.scheduling.decision.decide_backward_resilient` wraps it across sports with the same DB-failure fallback. Without `statuses_needing_update` any non-FINAL event counts; `update-status` passes `{SCHEDULED}`. Any non-FINAL event counts, so a game that never reaches FINAL keeps `fetch-scores` on its active cadence until it leaves the `window` (`odds_lambda.scheduling.decision.decide_backward`). An idle result still returns a `next_execution` at `idle_interval` ([results jobs](../collection/results-jobs.md)).

## Cadence

Hours per tier:

| Job | closing | pregame | sharp | early | opening | no game |
|---|---|---|---|---|---|---|
| `fetch-odds` | 0.5 | 3 | 12 | 24 | 48 | 24 |
| `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 2 |
| `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 2 |
| `agent-run` | 1 | 4 | 4 | 12 | 24 | 12 |

Values are each job module's cadence constant (`odds_lambda.jobs.fetch_odds.CADENCE`, `odds_lambda.jobs.fetch_oddsportal.CADENCE`, `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`, `odds_lambda.jobs.agent_run.CADENCE`); they replace the `interval_hours` argument of `odds_lambda.scheduling.helpers.self_schedule`, which is only logged. `odds_lambda.scheduling.decision.CadenceConfig.interval_for` maps `IN_PLAY` to `closing`. Scores: 3-day window, 6 h active, 12 h idle. Status: 4 h window, 1 h active, 6 h idle.

## Overnight windows

`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC and `baseball_mlb` 05:00-14:00 UTC via `odds_lambda.scheduling.helpers.apply_overnight_skip`, which pushes a candidate time to the resume hour and never raises. It applies only if the caller passes `overnight`: the fetch-oddsportal job passes its `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` window, Betfair looks the sport up with `.get()` for single-sport runs (no window for a sport missing from the map), `agent-run` indexes the map and raises `ValueError` for a sport missing from it (`odds_lambda.jobs.agent_run.schedule_next`), and `fetch-odds`, scores and status pass none. The same windows are also kept in the `overnight_*_utc` fields of `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` (#463).

`lookahead_days` comes from `odds_core.config.SchedulerConfig.lookahead_days` only in `fetch-odds`; the scrape and Betfair jobs use the function default of 7.

## Related

- [Self-scheduling](self-scheduling.md)
- [Where each job runs](job-deployment-matrix.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Local scheduler](local-scheduler.md)
