---
type: component
title: Scheduling decision engine
summary: "decision.py: decide_forward, decide_forward_resilient and decide_backward, each job's CadenceConfig, the run/skip gate and overnight skip windows."
sources:
  - docs/ARCHITECTURE.md#intelligent-scheduling-system
  - docs/DEPLOYMENT.md#smoke-check
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_forward
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_forward_resilient
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::decide_backward_resilient
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::CadenceConfig
  - packages/odds-lambda/odds_lambda/scheduling/helpers.py::apply_overnight_skip
verified_at: 7887039
---

# Scheduling decision engine

`packages/odds-lambda/odds_lambda/scheduling/decision.py` maps proximity to a polling interval and a run/skip gate for every [self-scheduling](self-scheduling.md) job. Each function returns a frozen `ScheduleDecision(should_execute, reason, next_execution, tier)`.

## Forward decisions (odds, scrapes, Betfair)

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`get_next_kickoff` in `packages/odds-lambda/odds_lambda/scheduling/helpers.py`). No kickoff within `lookahead_days` gives `should_execute=False` and `cadence.no_game`. Otherwise `_classify(hours_until)` picks a [fetch tier](fetch-tier.md) and the cadence interval.
- `decide_forward_resilient(sport_keys, cadence, ...)` uses the soonest kickoff across sports; if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- Always-run jobs (`fetch-oddsportal`, `fetch-betfair-exchange`) ignore `should_execute` and use only `next_execution`.

## Backward decisions (scores, status)

`decide_backward` runs while a game that kicked off within `window` is not yet resolved; `decide_backward_resilient` wraps it across sports with the same DB-failure fallback. Without `statuses_needing_update` any non-FINAL event counts; `update-status` passes `{SCHEDULED}`. An idle result still returns a `next_execution` at `idle_interval`.

## Per-job cadence (hours)

| Job | closing | pregame | sharp | early | opening | no game |
|---|---|---|---|---|---|---|
| `fetch-odds` | 0.5 | 3 | 12 | 24 | 48 | 24 |
| `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 2 |
| `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 2 |

Values are each job module's `CADENCE`. `CadenceConfig.interval_for` maps `IN_PLAY` to `closing`. Scores: 3-day window, 6 h active, 12 h idle. Status: 4 h window, 1 h active, 6 h idle.

## Overnight skip

`OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC and `baseball_mlb` 05:00-14:00 UTC via `apply_overnight_skip`, which pushes a candidate time to the resume hour. It applies only if the caller passes `overnight`: oddsportal passes its `LeagueSpec` window, Betfair passes `OVERNIGHT_WINDOWS` for single-sport runs, and `fetch-odds`, scores and status pass none.

## Hazards

- Overnight windows live in two places: `OVERNIGHT_WINDOWS` and the `overnight_*_utc` fields of `LeagueSpec` in `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LeagueSpec`.
- `_classify` uses strict `<` on `FetchTier.max_hours`; `calculate_tier` uses `<=`. A kickoff exactly 3.0 h away is PREGAME here and CLOSING when stored ([fetch tier](fetch-tier.md)).
- `get_next_kickoff` only returns future SCHEDULED events, so `FetchTier.IN_PLAY` is never produced by forward decisions.
- `decide_backward` counts every non-FINAL event, so a postponed or cancelled event keeps `fetch-scores` on its 6 h cadence for three days.
- `lookahead_days` comes from `SchedulerConfig.lookahead_days` only in `fetch-odds`; the scrape and Betfair jobs use the function default of 7.
- Docstrings in this module refer to deleted engines (`SchedulingIntelligence`, `should_execute_scores`); search for those names finds nothing.
- The smoke check only reaches a job's fetch body when `should_execute` is true ([local scheduler](local-scheduler.md)).

## Related

- [fetch-odds job](job-fetch-odds.md), [fetch-oddsportal job](job-fetch-oddsportal.md), [fetch-betfair-exchange job](job-fetch-betfair-exchange.md), [fetch-scores job](job-fetch-scores.md), [update-status job](job-update-status.md).
