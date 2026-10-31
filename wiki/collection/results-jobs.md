---
type: reference
title: Results and status jobs
description: "update-status, fetch-scores, fetch-oddsportal-results, fetch-mlb-results and odds fetch scores: cadence, flow, heartbeats, failure handling."
sources:
  - odds_lambda.jobs.update_status.main
  - odds_lambda.jobs.update_status._update_event_statuses
  - odds_lambda.jobs.update_status._status_decision
  - odds_lambda.jobs.fetch_scores.main
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_lambda.jobs.fetch_scores.SCORES_WINDOW
  - odds_lambda.jobs.fetch_oddsportal_results.main
  - odds_lambda.jobs.fetch_oddsportal_results.process_results
  - odds_lambda.jobs.fetch_oddsportal_results._raise_if_feed_parsed_nothing
  - odds_lambda.jobs.fetch_mlb_results
  - odds_lambda.results_run
  - odds_lambda.storage.event_merge.find_match_id
  - odds_cli.commands.fetch.fetch_results
  - odds_cli.commands.events.repair_scores_command
  - odds_lambda.storage.mlb_score_repair.check_scores
  - odds_lambda.storage.mlb_score_repair.repair_scores
  - odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results
  - odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_game
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_results
  - odds_lambda.oddsportal_client.parse_results_page
  - odds_lambda.oddsportal_client.CancelledMatch
  - odds_lambda.oddsportal_client.parse_archive_path
  - odds_lambda.scheduling.decision.decide_backward
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_cli.commands.fetch.fetch_scores
status: stable
---

# Results and status jobs

Four jobs and one command move events to LIVE, FINAL, POSTPONED and CANCELLED ([event-status-lifecycle](event-status-lifecycle.md)). All four jobs are in the default local `bootstrap_jobs` and run on the production host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)); their AWS rules are dormant.

## update-status

`odds_lambda.jobs.update_status.main(ctx)` flips SCHEDULED events whose `commence_time` has passed to LIVE. It is a global job (not per-sport): one schedule `update-status` (on AWS the always-created rule `odds-update-status`). It runs for `data_collection.sports` when no sport is given.

1. `odds_lambda.jobs.update_status._status_decision` calls `odds_lambda.scheduling.decision.decide_backward_resilient` with a 4-hour window, 1 h active and 6 h idle intervals, counting only `{EventStatus.SCHEDULED}`.
2. Pre-schedule under `update-status`, skip if idle.
3. Inside `job_alert_context("update-status")`, `odds_lambda.jobs.update_status._update_event_statuses` reads SCHEDULED events with `commence_time` in the last four hours and calls `update_event_status(LIVE)` for each that has started.
4. Re-decide and post-schedule.

`odds_lambda.jobs.update_status._update_event_statuses` does not filter by sport while the gate does, so when a per-sport schedule fires it flips events of every sport. Only events that kicked off within the last 4 hours are eligible, so if the job was down longer, older SCHEDULED events can only reach FINAL through the scores or results jobs.

## fetch-scores

`odds_lambda.jobs.fetch_scores.main(ctx)` is the Odds API `/scores` job. It is per sport and its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist limits its schedules to EPL to save quota, so no scheduled job fetches MLB or NBA scores from the Odds API; MLB events reach FINAL through [fetch-mlb-results](#fetch-mlb-results).

1. Gate with `odds_lambda.scheduling.decision.decide_backward_resilient` using `odds_lambda.jobs.fetch_scores.SCORES_WINDOW` (3 days), `odds_lambda.jobs.fetch_scores.SCORES_ACTIVE_INTERVAL` 6 h and `odds_lambda.jobs.fetch_scores.SCORES_IDLE_INTERVAL` 12 h: it runs while any event that kicked off in the last three days is not FINAL. Any non-FINAL event counts (`odds_lambda.scheduling.decision.decide_backward`), including POSTPONED and CANCELLED ones.
2. Pre-schedule, then skip if idle.
3. `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`: for each sport call `TheOddsAPIClient.get_scores(sport, days_from=3)`, and for every `completed` entry with parseable scores find the event the entry's Odds API id names (`odds_lambda.odds_api_events.find_event_by_odds_api_id`: the event holding it as an `odds_api` external id, else none) and call `update_event_status(FINAL, home_score, away_score)` on it, committing per sport. An entry naming no stored event is logged (`score_event_not_found`) and skipped.
4. Re-decide and post-schedule.

Per-event failures are logged and skipped. A whole-job failure re-raises through [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery), which sends the `job_failure:fetch-scores-<suffix>` critical once per outage. Each `/scores` call costs 2 units because the client sends `daysFrom` (#441; [odds-api-client](odds-api-client.md#cost-per-call)). Every run after the pre-schedule, gated ones included, writes `heartbeat:fetch-scores-<suffix>`.

`odds fetch scores` repeats the same loop inline (logging `score_event_not_found` for an entry that names no stored event) with `--sport` (default `basketball_nba`) and `--days` (default 3); it does not call the job and does not gate or reschedule. The update loop is duplicated in `odds_cli.commands.fetch._fetch_scores` and `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`, so a change to result parsing needs both (#463). The job loops sports and alerts; the CLI takes one sport.

## fetch-oddsportal-results

`odds_lambda.jobs.fetch_oddsportal_results.main(ctx)` reads finished EPL matches from OddsPortal's results page over HTTP, with no browser, and sets the matching events FINAL with scores. It stores no odds. `odds_lambda.scheduling.jobs._JOB_SPORTS` limits its schedules to EPL, and `odds_lambda.jobs.fetch_oddsportal_results.process_results` raises `ValueError` for MLB, which [fetch-mlb-results](#fetch-mlb-results) settles. League config is `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)); only its `listing_path` and `sport_key` are used, and only the EPL spec is settled (the MLB and NBA specs are in `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` for odds). `odds_lambda.jobs.fetch_oddsportal_results.process_results` uses three session scopes so no connection is held across the fetch (Neon closes idle connections):

1. **Read**: `odds_lambda.results_run.get_pending_events` returns plain `odds_lambda.results_run.PendingEvent` dataclasses (with the event's integer id) for events of the sport with `commence_time` in the past, and within `odds_lambda.results_run.SCHEDULED_LOOKBACK` (14 days) unless the lookback is lifted, that are SCHEDULED, LIVE, POSTPONED, or FINAL without both scores. None means exit.
2. **Fetch**: `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` loads `<listing_path>results/` for the archive feed URL in its RSC payload (`odds_lambda.oddsportal_client.parse_archive_path`; a page without one raises `odds_lambda.oddsportal_client.OddsPortalError`), then pages through `<feed>/page/N/`, newest first, until a page holds a kick-off older than the oldest pending event minus 2 h or the pages run out. A page that fails after the first keeps the earlier pages and the run settles from them; a failing first page raises. `odds_lambda.oddsportal_client.parse_results_page` keeps rows with status 3 (finished) and digit scores as matches and rows with status 5 ("Canceled") as `odds_lambda.oddsportal_client.CancelledMatch`; scheduled, in-play and unscored rows are dropped. The pages are the current season's, so a pending event from an earlier season stays pending.
3. **Write**: each finished match has its teams normalised with `odds_core.team.normalize_team_name` and is looked up with `odds_lambda.storage.event_merge.find_match_id`, the read-only finder that the other loaders use and that returns the event's integer id (same teams, within `odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW` of 2 h, of several events, the one a merge would keep). It applies only when that event is pending: `update_event_status(FINAL, scores)` sets it, or `update_event_status(CANCELLED)` for a cancelled match whose event can still become CANCELLED (a scoreless FINAL event cannot). Commit per event with rollback on error.

An event with no finished or cancelled match stays pending and is counted in `events_unresolved`; that count does not alert on its own ([event-matching](../storage/event-matching.md)). `odds_lambda.jobs.fetch_oddsportal_results._raise_if_feed_parsed_nothing` raises when pending events older than 24 h exist, the feed returned rows and none parsed as finished or cancelled, because that means the parser cannot read the page. `odds fetch results --sport S [--lookback-days N]` (`odds_cli.commands.fetch.fetch_results`) runs the sport's settle with a chosen lookback, or with every pending event when the option is omitted, for backfills: OddsPortal for EPL, [fetch-mlb-results](#fetch-mlb-results)' settle for `baseball_mlb`. `odds_lambda.results_run.ResultsStats.events_cancelled` counts the cancellations. Rerunning is idempotent because settled events leave the pending query. No postponed row has been observed on the EPL results page, so a postponed match stays pending until ESPN marks it POSTPONED ([event-status-lifecycle](event-status-lifecycle.md#who-sets-what)).

`main` runs the settle through `odds_lambda.results_run.run_daily_results_job`, described in [Run, alerts and booking](#run-alerts-and-booking). It requires `ctx.sport` and raises `ValueError` without one.

## fetch-mlb-results

`odds_lambda.jobs.fetch_mlb_results.main(ctx)` settles MLB events from the MLB Stats API schedule by their `gamePk` ([event matching](../storage/event-matching.md#live-mlb-events)). It is one job, not per sport (`fetch-mlb-results`, no suffix), and books its runs without a sport. `odds_lambda.jobs.fetch_mlb_results.process_results`:

1. **Read**: `odds_lambda.results_run.get_pending_events` for `baseball_mlb`, as above. Events without a `gamePk` are never touched. One still SCHEDULED or LIVE a day after kick-off is reported by the `unsettled_events:baseball_mlb` check unless it is marked as having no game, which makes it UNMATCHED (`odds events mark-unassignable`) ([health monitor](../operations/health-monitor.md)).
2. **Fetch** (`odds_lambda.jobs.fetch_mlb_results._fetch_slots`): one linescore schedule call (`odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results`) from the day before the earliest pending event's MLB date (US Eastern) to the day after today, then one `odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_game` call per game with no played slot in that range, which finds a replay booked later than today. A failed call fails the run.
3. **Decide** (`odds_lambda.jobs.fetch_mlb_results.game_outcome`), from all the game's slots: the played slot is the one neither postponed, cancelled nor a resumed continuation (`odds_lambda.mlb_game_resolver.played_slots`).
   - Final (`odds_lambda.mlb_game_resolver.final_scores`): FINAL with MLB's home and away score.
   - A played slot not yet Final (a replay still to come, or a game in progress): no status change.
   - No played slot: CANCELLED when a slot is cancelled, else POSTPONED when one is postponed.
   - The game absent from the schedule: nothing.
4. **Write**, per event in its own transaction (`odds_lambda.jobs.fetch_mlb_results._apply`): the event first follows its game to the played slot (`odds_lambda.storage.event_kickoff.move_kickoff` to `odds_lambda.mlb_game_resolver.PlayedSlot.start_for` on the MLB Stats API's authority at the current time, which may be a past start, the step the live jobs and the `gamePk` backfill take, which makes a POSTPONED event, or a LIVE one moved to a start still ahead, SCHEDULED), then takes the status through `odds_lambda.storage.writers.OddsWriter.update_event_status`. A TBD start moves an event only when it is on another date. An event moved to a replay that has not started leaves the pending query until its new start passes. Other sources' snapshots on the event do not hold the move back, since the MLB Stats API ranks first for MLB.

A postponed slot's event normally reaches the played slot before this job sees it: live ingestion moves the event holding the `gamePk` whenever OddsPortal or Betfair lists the replay ([event matching](../storage/event-matching.md#live-mlb-events)). The job moves it only when no listing did. `odds_lambda.results_run.ResultsStats` counts FINAL (`events_updated`), CANCELLED, POSTPONED (including events already POSTPONED whose game still has no played slot) and moved events; every other pending event, those without a `gamePk` included, is unresolved. Each settled doubleheader game goes to the event holding its own `gamePk`, so a placeholder TBD start cannot swap the two games' scores.

## Run, alerts and booking

Both results jobs' `main` call `odds_lambda.results_run.run_daily_results_job`, which runs the settle inside `odds_core.alerts.job_alert_context`, then in `finally` books tomorrow 08:00 UTC through `odds_lambda.scheduling.helpers.self_schedule`, ignoring overnight windows. A scheduling failure sends `odds_core.alerts.send_scheduling_failure`, which never raises, so it cannot mask the original exception. The jobs have no pre-schedule, so a kill before `finally` skips the reschedule and ends the chain (#438). On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)). The job fails, with a critical alert and no heartbeat, when updating any event in the database raised, after the other events are settled (`odds_lambda.results_run.ResultsStats.errors`).

`odds_lambda.results_run.record_run_outcome`, called after each run, writes a `results_settled:<sport>` row to `odds_core.models.AlertHistory` when the run settled events FINAL. When a run settles none, it sends `results_stalled:<sport>` if unresolved events that kicked off after the latest such row are still pending more than 24 h on, so unmatched events from before the job's last settling run stay quiet, as does a sport with no settling run on record yet. Both keys are per sport, not per job. `odds_lambda.health_monitor.HealthMonitor.check_unsettled_events` covers single stuck EPL and MLB events, reminding every 24 h ([health monitor](../operations/health-monitor.md)).

## Verifying MLB scores

`odds events repair-scores [--apply]` (`odds_cli.commands.events.repair_scores_command`) checks every FINAL `baseball_mlb` event against the game its `gamePk` names. `odds_lambda.storage.mlb_score_repair.repair_scores` reads the linescore schedule over the events' MLB dates, padded a day each side (`odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_results`), and fetches each game not Final in it by `gamePk` (`odds_lambda.mlb_stats_fetcher.MlbStatsFetcher.fetch_game`). `odds_lambda.storage.mlb_score_repair.check_scores` compares each event's score with its game's Final score (`odds_lambda.mlb_game_resolver.final_scores`): equal is correct, different (or no stored score) a mismatch. An event without a `gamePk`, or whose game is not Final, is unmatched and never changed.

Without `--apply` it prints a summary line (`FINAL baseball_mlb events: N; correct: …; mismatch: …; unmatched: …`), then one line per mismatch (`  MISMATCH <event id> <start> <away> @ <home>: stored h-a, MLB h-a (game <gamePk>)`) and per unmatched event (`  UNMATCHED <event id> …: <reason>`), and writes nothing. With `--apply` `odds_lambda.storage.mlb_score_repair.repair_scores` writes `home_score` and `away_score` of the mismatches only, in one transaction, bypassing the status guard that a FINAL event cannot be settled again; a second run reports no mismatches. It corrects events settled before MLB was settled by `gamePk`, such as a doubleheader whose games took each other's score, and is re-run after the `gamePk` backfill ([Assign MLB events their gamePk](../storage/howto-assign-mlb-game-pks.md)).

## Heartbeats

The results jobs' heartbeat expectation is 26 h: `heartbeat:fetch-oddsportal-results-epl` and `heartbeat:fetch-mlb-results`. `update-status` writes `heartbeat:update-status` but has no default expectation, and `fetch-scores` is expected every 13 h per sport (the 12 h idle interval plus 1 h; no overnight pause) ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [Event status lifecycle](event-status-lifecycle.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Odds API ingestion service](odds-ingestion-service.md)
