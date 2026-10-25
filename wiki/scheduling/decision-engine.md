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
  - odds_lambda.scheduling.decision.classify_tier
  - odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS
  - odds_lambda.scheduling.decision.KickoffLookup
  - odds_lambda.collection_calendar.decide_collection_run
  - odds_lambda.collection_calendar.next_fixture_kickoff
  - odds_lambda.collection_calendar.published_schedule_kickoff
  - odds_lambda.scheduling.helpers.self_schedule
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.scheduling.helpers.apply_overnight_skip
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.jobs.fetch_polymarket_odds.CADENCE
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
- `odds_lambda.scheduling.helpers.get_next_kickoff` sees only stored future SCHEDULED events, so a sport with no stored future events sleeps for `no_game` hours even if fixtures exist upstream, and `odds_lambda.fetch_tier.FetchTier.IN_PLAY` is never produced by forward decisions. A job that must wake without help from stored events passes another source as `next_kickoff_lookup` ([Dormancy](#dormancy)).

## Forward decisions

Used by odds, scrapes, Betfair and the agent.

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`odds_lambda.scheduling.helpers.get_next_kickoff`). No kickoff within `lookahead_days` gives `should_execute=False` and a check-in at the earlier of `cadence.no_game` hours from now and the moment a known later kickoff enters `lookahead_days`; the check-in ignores the overnight window. Otherwise `odds_lambda.scheduling.decision.classify_tier(hours_until)` picks a [fetch tier](../storage/fetch-tier.md) and the cadence interval. `odds_lambda.scheduling.decision.classify_tier` uses strict `<` on `odds_lambda.fetch_tier.FetchTier.max_hours` where stored tiers use `<=`, so a kickoff exactly 3.0 h away is PREGAME here and CLOSING when stored (#446, [boundary encodings](../storage/fetch-tier.md#boundary-encodings)).
- `odds_lambda.scheduling.decision.decide_forward_resilient(sport_keys, cadence, ..., next_kickoff_lookup=None)` uses the soonest kickoff across sports, from `odds_lambda.scheduling.helpers.get_next_kickoff` unless a `odds_lambda.scheduling.decision.KickoffLookup` is given; if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- `fetch-oddsportal`, `fetch-betfair-exchange` and `fetch-polymarket-odds` run only while `should_execute` holds and otherwise only book a check-in ([Dormancy](#dormancy)). `agent-run` uses `odds_lambda.scheduling.decision.decide_forward_resilient` for its cadence, ignores the gate and uses only `next_execution`; a requested wake-up overrides it ([wake-up requests](../agent/agent-run-job.md#wake-up-requests)).

## Dormancy

A collector that creates its own events cannot be gated on stored events alone: stopped, it would never see the next season. `odds_lambda.collection_calendar.decide_collection_run` is `odds_lambda.scheduling.decision.decide_forward_resilient` with `lookahead_days` set to `odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS` (28, longer than the roughly 20-day gaps between two Premier League fixtures) and `odds_lambda.collection_calendar.next_fixture_kickoff` as the lookup. Its `stored_kickoff` argument replaces the stored events' `commence_time` in that lookup and in the close run; only `fetch-betfair-exchange` passes one ([close run](#close-run)). That is the earlier of the stored SCHEDULED events and the sport's published schedule (`odds_lambda.collection_calendar.published_schedule_kickoff`):

- EPL: the ESPN Premier League fixtures that are not postponed, written by the daily `fetch-espn-fixtures` cron.
- MLB: the MLB Stats API games of the latest `fetch-mlb-probables` fetch, under the same freshness and game-type rules as the [stale-data calendar](../operations/health-monitor.md#stale-data-check). The feed covers about four days, so MLB wakes that far ahead.

`fetch-oddsportal`, `fetch-betfair-exchange` and `fetch-polymarket-odds` pre-schedule from this decision. When `should_execute` is false they write their heartbeat and return without scraping, so the [heartbeat check](../operations/health-monitor.md#heartbeats) sees a live chain; their `no_game` check-in is 6 h, inside every heartbeat expectation. `fetch-polymarket-odds` pairs Gamma matches with stored events, so it passes `stored_only` to `odds_lambda.collection_calendar.decide_collection_run`. The published-schedule crons never go dormant, and a cron or the chain keeps its place across a scheduler restart ([bootstrap](local-scheduler.md#bootstrap-jobs-and-cron-table)). `fetch-odds` is gated on stored events and syncs the free `/events` listing before it decides, so it discovers fixtures without any other job.

## Backward decisions

Used by scores and status.

`odds_lambda.scheduling.decision.decide_backward` runs while a game that kicked off within `window` is not yet resolved; `odds_lambda.scheduling.decision.decide_backward_resilient` wraps it across sports with the same DB-failure fallback. Without `statuses_needing_update` any non-FINAL event counts; `update-status` passes `{SCHEDULED}`. Any non-FINAL event counts, so a game that never reaches FINAL keeps `fetch-scores` on its active cadence until it leaves the `window` (`odds_lambda.scheduling.decision.decide_backward`). An idle result still returns a `next_execution` at `idle_interval` ([results jobs](../collection/results-jobs.md)).

## Cadence

Hours per tier:

| Job | closing | pregame | sharp | early | opening | no game |
|---|---|---|---|---|---|---|
| `fetch-odds` | 2 | 3 | 12 | 24 | 48 | 24 |
| `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 6 |
| `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 6 |
| `fetch-polymarket-odds` | 0.083 | 0.25 | 0.25 | 0.5 | 1 | 6 |
| `agent-run` | 1 | 4 | 4 | 12 | 24 | 12 |

Values are each job module's cadence constant (`odds_lambda.jobs.fetch_odds.CADENCE`, `odds_lambda.jobs.fetch_oddsportal.CADENCE`, `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`, `odds_lambda.jobs.fetch_polymarket_odds.CADENCE`, `odds_lambda.jobs.agent_run.CADENCE`); they replace the `interval_hours` argument of `odds_lambda.scheduling.helpers.self_schedule`, which is only logged. `odds_lambda.scheduling.decision.CadenceConfig.interval_for` maps `IN_PLAY` to `closing`. Scores: 3-day window, 6 h active, 12 h idle. Status: 4 h window, 1 h active, 6 h idle.

## Close run

`odds_lambda.scheduling.decision.CadenceConfig.close_lead`, when set, adds a run just before each kickoff. The forward decisions look up the earliest kickoff more than `close_lead` away, through the `close_kickoff_lookup` of `odds_lambda.scheduling.decision.decide_forward_resilient` (the stored events' `commence_time` by default; `odds_lambda.collection_calendar.decide_collection_run` passes its `stored_kickoff`). If the cadence's next time falls after that kickoff minus `close_lead`, the run moves to that time, and the overnight skip does not apply to it. A close run less than `odds_lambda.scheduling.decision.MIN_CLOSE_RUN_GAP` (60 s) away is dropped, because the run deciding it is already at the close. Only `odds_lambda.jobs.fetch_betfair_exchange.CADENCE` sets it, at 5 min, so the Betfair close does not depend on where the 15-minute cadence happens to fall. That job's kickoffs come from `odds_lambda.betfair.kickoff.next_betfair_kickoff`, which adds an earlier Betfair start as a target while it is ahead, so an event can get a close run before it as well as before its `commence_time` ([Betfair job](../collection/betfair-exchange.md#flow)).

## Overnight windows

`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC and `baseball_mlb` 05:00-14:00 UTC via `odds_lambda.scheduling.helpers.apply_overnight_skip`, which pushes a candidate time to the resume hour and never raises. It applies only if the caller passes `overnight`: the fetch-oddsportal job passes its `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight window, Betfair and Polymarket odds look the sport up with `.get()` (no window for a sport missing from the map), `agent-run` indexes the map and raises `ValueError` for a sport missing from it (`odds_lambda.jobs.agent_run.schedule_next`), and `fetch-odds`, scores and status pass none. The same windows are also kept in the `overnight_*_utc` fields of `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` (#463).

`lookahead_days` comes from `odds_core.config.SchedulerConfig.lookahead_days` in `fetch-odds`, from `odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS` in the three collectors, and is the function default of 7 elsewhere. Only an active run applies the overnight skip; a dormant check-in does not.

## Related

- [Self-scheduling](self-scheduling.md)
- [Where each job runs](job-deployment-matrix.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Local scheduler](local-scheduler.md)
