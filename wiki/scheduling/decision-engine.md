---
type: reference
title: Scheduling decision engine
description: decide_forward and decide_backward and their resilient wrappers, CadenceConfig and per-job cadences, lead runs before kickoffs, overnight windows, and the self_schedule and kickoff helpers.
sources:
  - odds_lambda.scheduling.decision.decide_forward
  - odds_lambda.scheduling.decision.decide_forward_resilient
  - odds_lambda.scheduling.decision.decide_backward
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_lambda.scheduling.decision.CadenceConfig
  - odds_lambda.scheduling.decision.ScheduleDecision
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.tier_utils.calculate_tier
  - odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS
  - odds_lambda.scheduling.decision.KickoffLookup
  - odds_lambda.scheduling.decision.LeadRun
  - odds_lambda.scheduling.decision.next_lead_run
  - odds_lambda.scheduling.decision.MIN_LEAD_RUN_GAP
  - odds_lambda.collection_calendar.decide_collection_run
  - odds_lambda.collection_calendar.next_fixture_kickoff
  - odds_lambda.collection_calendar.published_schedule_kickoff
  - odds_lambda.scheduling.helpers.self_schedule
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.scheduling.helpers.apply_overnight_skip
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CLOSE_LEAD
  - odds_lambda.jobs.fetch_oddsportal.candidate_scrape_leads
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

- `decide_forward(sport_key, cadence, overnight=None, lookahead_days=7, ...)` takes the next SCHEDULED kickoff for the sport (`odds_lambda.scheduling.helpers.get_next_kickoff`). No kickoff within `lookahead_days` gives `should_execute=False` and a check-in at the earlier of `cadence.no_game` hours from now and the moment a known later kickoff enters `lookahead_days`; the check-in ignores the overnight window. Otherwise `odds_lambda.tier_utils.calculate_tier(hours_until)`, the one hours-to-tier rule, picks a [fetch tier](../storage/fetch-tier.md#the-one-hours-to-tier-rule) and the cadence interval: a kickoff exactly on a `odds_lambda.fetch_tier.FetchTier.max_hours` boundary (3.0 h away, say) is in the nearer tier, as it is wherever hours alone decide the tier; a stored snapshot's tier also depends on its time basis ([the tier a snapshot covers](../storage/fetch-tier.md#the-tier-a-snapshot-covers)).
- `odds_lambda.scheduling.decision.decide_forward_resilient(sport_keys, cadence, ..., next_kickoff_lookup=None, leads=())` uses the soonest kickoff across sports, from `odds_lambda.scheduling.helpers.get_next_kickoff` unless a `odds_lambda.scheduling.decision.KickoffLookup` is given, and moves the next run earlier for any of `leads` ([lead runs](#lead-runs)); if the kickoff query raises it returns `should_execute=True` at `cadence.db_fallback` (default 1 h) so the chain survives a database outage.
- `fetch-oddsportal`, `fetch-betfair-exchange` and `fetch-polymarket-odds` run only while `should_execute` holds and otherwise only book a check-in ([Dormancy](#dormancy)). `agent-run` uses `odds_lambda.scheduling.decision.decide_forward_resilient` for its cadence, ignores the gate and uses only `next_execution`; a requested wake-up overrides it ([wake-up requests](../agent/agent-run-job.md#wake-up-requests)).

## Dormancy

A collector that creates its own events cannot be gated on stored events alone: stopped, it would never see the next season. `odds_lambda.collection_calendar.decide_collection_run` is `odds_lambda.scheduling.decision.decide_forward_resilient` with `lookahead_days` set to `odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS` (28, longer than the roughly 20-day gaps between two Premier League fixtures) and `odds_lambda.collection_calendar.next_fixture_kickoff` as the lookup. Its `stored_kickoff` argument replaces the stored events' `commence_time` in that lookup and in the lead runs; only `fetch-betfair-exchange` passes one ([lead runs](#lead-runs)). That is the earlier of the stored SCHEDULED events and the sport's published schedule (`odds_lambda.collection_calendar.published_schedule_kickoff`):

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

## Lead runs

A lead (`odds_lambda.scheduling.decision.Lead`) is a time before kickoff at which a job must run whatever its cadence (`before`), and how long before kickoff a run that fell behind is still worth taking (`until`). `odds_lambda.scheduling.decision.decide_forward_resilient` and `odds_lambda.collection_calendar.decide_collection_run` take a sequence of `leads` and `run_started`, the start of the run that is deciding (default now). A run serves a lead time when it started at most `odds_lambda.scheduling.decision.MIN_LEAD_RUN_GAP` (60 s) before it, or after it: its snapshots are as recent as the lead run's. For each lead and sport, `odds_lambda.scheduling.decision.next_lead_run` looks up the earliest kickoff whose lead time the deciding run did not serve and that is more than `until` away. Its run is at the lead time, or now when that has passed, so a cadence run that started before a lead time but finished after it never cancels the lead run. Kickoffs come from the `lead_kickoff_lookup` of `odds_lambda.scheduling.decision.decide_forward_resilient`: the stored SCHEDULED events' `commence_time` (`odds_lambda.scheduling.helpers.get_next_kickoff`) by default, and the `stored_kickoff` that `odds_lambda.collection_calendar.decide_collection_run` is given. The earliest of those runs is an `odds_lambda.scheduling.decision.LeadRun`. If the cadence's next time falls after it, the run moves to it, and the overnight skip does not apply to it. A dormant decision has no lead run. `odds_lambda.scheduling.decision.decide_forward` takes no leads.

Two jobs pass leads:

- `fetch-betfair-exchange` passes `odds_lambda.jobs.fetch_betfair_exchange.CLOSE_LEAD` (5 min before kickoff, late until kickoff), so the Betfair close does not depend on where the 15-minute cadence happens to fall. Its kickoffs come from `odds_lambda.betfair.kickoff.next_betfair_kickoff`, which adds an earlier market start (`odds_snapshots.market_start_time` of the event's latest `betfair` match-odds snapshot) as a target while it is ahead, so an event can get a close run before it as well as before its `commence_time` ([Betfair job](../collection/betfair-exchange.md#flow)).
- `fetch-oddsportal` passes `odds_lambda.jobs.fetch_oddsportal.candidate_scrape_leads`: 5 min before each active mechanical candidate's decision time, late until that decision time, so every candidate decision reads a snapshot taken just before it ([fetch-oddsportal job](../collection/fetch-oddsportal-job.md#flow)).

## Overnight windows

`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` suppresses runs in `soccer_epl` 22:00-06:00 UTC, `baseball_mlb` 05:00-14:00 UTC and `basketball_nba` 07:00-15:00 UTC (the last tip-off is about 03:30 UTC and the game ends by about 06:00; the earliest weekend tip-offs are about 16:00 UTC) via `odds_lambda.scheduling.helpers.apply_overnight_skip`, which pushes a candidate time to the resume hour and never raises. It applies only if the caller passes `overnight`: the fetch-oddsportal job passes its `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` overnight window, Betfair and Polymarket odds look the sport up with `.get()` (no window for a sport missing from the map), `agent-run` indexes the map and raises `ValueError` for a sport missing from it (`odds_lambda.jobs.agent_run.schedule_next`), and `fetch-odds`, scores and status pass none. The same windows are also kept in the `overnight_*_utc` fields of `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` (#463).

`lookahead_days` comes from `odds_core.config.SchedulerConfig.lookahead_days` in `fetch-odds`, from `odds_lambda.scheduling.decision.DORMANCY_HORIZON_DAYS` in the three collectors, and is the function default of 7 elsewhere. Only an active run applies the overnight skip; a dormant check-in does not.

## Related

- [Self-scheduling](self-scheduling.md)
- [Where each job runs](job-deployment-matrix.md)
- [Fetch tier](../storage/fetch-tier.md)
- [Local scheduler](local-scheduler.md)
