---
type: reference
title: Results and status jobs
description: "update-status, fetch-scores, fetch-oddsportal-results and odds fetch scores: cadence, flow, heartbeats, failure handling."
sources:
  - odds_lambda.jobs.update_status.main
  - odds_lambda.jobs.update_status._update_event_statuses
  - odds_lambda.jobs.update_status._status_decision
  - odds_lambda.jobs.fetch_scores.main
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_lambda.jobs.fetch_scores.SCORES_WINDOW
  - odds_lambda.jobs.fetch_oddsportal_results.main
  - odds_lambda.jobs.fetch_oddsportal_results.process_results
  - odds_lambda.jobs.fetch_oddsportal_results.get_pending_events
  - odds_lambda.jobs.fetch_oddsportal_results.SCHEDULED_LOOKBACK
  - odds_lambda.jobs.fetch_oddsportal_results._raise_if_feed_parsed_nothing
  - odds_lambda.jobs.fetch_oddsportal_results.ResultsStats
  - odds_lambda.jobs.fetch_oddsportal_results.record_run_outcome
  - odds_lambda.storage.event_merge.find_match_id
  - odds_cli.commands.fetch.fetch_results
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

Three jobs and one command move events to LIVE, FINAL and CANCELLED ([event-status-lifecycle](event-status-lifecycle.md)). All three jobs are in the default local `bootstrap_jobs` and run on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)); their AWS rules are dormant.

## update-status

`odds_lambda.jobs.update_status.main(ctx)` flips SCHEDULED events whose `commence_time` has passed to LIVE. It is a global job (not per-sport): one schedule `update-status` (on AWS the always-created rule `odds-update-status`). It runs for `data_collection.sports` when no sport is given.

1. `odds_lambda.jobs.update_status._status_decision` calls `odds_lambda.scheduling.decision.decide_backward_resilient` with a 4-hour window, 1 h active and 6 h idle intervals, counting only `{EventStatus.SCHEDULED}`.
2. Pre-schedule under `update-status`, skip if idle.
3. Inside `job_alert_context("update-status")`, `odds_lambda.jobs.update_status._update_event_statuses` reads SCHEDULED events with `commence_time` in the last four hours and calls `update_event_status(LIVE)` for each that has started.
4. Re-decide and post-schedule.

`odds_lambda.jobs.update_status._update_event_statuses` does not filter by sport while the gate does, so when a per-sport schedule fires it flips events of every sport. Only events that kicked off within the last 4 hours are eligible, so if the job was down longer, older SCHEDULED events can only reach FINAL through the scores or results jobs.

## fetch-scores

`odds_lambda.jobs.fetch_scores.main(ctx)` is the Odds API `/scores` job. It is per sport, and its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist limits its schedules to EPL to save quota, so no scheduled job fetches MLB scores from the Odds API; MLB events reach FINAL through the OddsPortal results job.

1. Gate with `odds_lambda.scheduling.decision.decide_backward_resilient` using `odds_lambda.jobs.fetch_scores.SCORES_WINDOW` (3 days), `odds_lambda.jobs.fetch_scores.SCORES_ACTIVE_INTERVAL` 6 h and `odds_lambda.jobs.fetch_scores.SCORES_IDLE_INTERVAL` 12 h: it runs while any event that kicked off in the last three days is not FINAL. Any non-FINAL event counts (`odds_lambda.scheduling.decision.decide_backward`), including POSTPONED and CANCELLED ones.
2. Pre-schedule, then skip if idle.
3. `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`: for each sport call `TheOddsAPIClient.get_scores(sport, days_from=3)`, and for every `completed` entry with parseable scores call `update_event_status(FINAL, home_score, away_score)`, committing per sport.
4. Re-decide and post-schedule.

Per-event failures are logged and skipped. A whole-job failure re-raises through `odds_core.alerts.job_alert_context`, which sends the `job_failure:fetch-scores-<suffix>` critical once per outage. Each `/scores` call costs 2 units because the client sends `daysFrom` (#441; [odds-api-client](odds-api-client.md#cost-per-call)). Every run after the pre-schedule, gated ones included, writes `heartbeat:fetch-scores-<suffix>`.

`odds fetch scores` repeats the same loop inline with `--sport` (default `basketball_nba`) and `--days` (default 3); it does not call the job and does not gate or reschedule. The update loop is duplicated in `odds_cli.commands.fetch._fetch_scores` and `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`, so a change to result parsing needs both (#463). The job loops sports and alerts; the CLI takes one sport.

## fetch-oddsportal-results

`odds_lambda.jobs.fetch_oddsportal_results.main(ctx)` reads finished matches from OddsPortal's results page over HTTP, with no browser, and sets the matching events FINAL with scores. It stores no odds. League config is `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)); only its `listing_path` and `sport_key` are used. `odds_lambda.jobs.fetch_oddsportal_results.process_results` uses three session scopes so no connection is held across the fetch (Neon closes idle connections):

1. **Read**: `odds_lambda.jobs.fetch_oddsportal_results.get_pending_events` returns plain `odds_lambda.jobs.fetch_oddsportal_results.PendingEvent` dataclasses for events of the sport with `commence_time` in the past, and within `odds_lambda.jobs.fetch_oddsportal_results.SCHEDULED_LOOKBACK` (14 days) unless the lookback is lifted, that are SCHEDULED, LIVE, POSTPONED, or FINAL without both scores. None means exit.
2. **Fetch**: `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` loads `<listing_path>results/` for the archive feed URL in its RSC payload (`odds_lambda.oddsportal_client.parse_archive_path`; a page without one raises `odds_lambda.oddsportal_client.OddsPortalError`), then pages through `<feed>/page/N/`, newest first, until a page holds a kick-off older than the oldest pending event minus 2 h or the pages run out. A page that fails after the first keeps the earlier pages and the run settles from them; a failing first page raises. `odds_lambda.oddsportal_client.parse_results_page` keeps rows with status 3 (finished) and digit scores as matches and rows with status 5 ("Canceled") as `odds_lambda.oddsportal_client.CancelledMatch`; scheduled, in-play and unscored rows are dropped. The pages are the current season's, so a pending event from an earlier season stays pending.
3. **Write**: each finished match has its teams normalised with `odds_core.team.normalize_team_name` and is looked up with `odds_lambda.storage.event_merge.find_match_id`, the read-only finder that the other loaders use (same teams, within `odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW` of 2 h, canonical event of any twins). It applies only when that event is pending and the match is the nearest result of those teams to the event's kick-off, so two games of a series a few hours apart never take each other's score. `update_event_status(FINAL, scores)` sets it, or `update_event_status(CANCELLED)` for a cancelled match whose event can still become CANCELLED (a scoreless FINAL event cannot); the nearest-result rule counts finished and cancelled matches together. Commit per event with rollback on error.

An event with no finished or cancelled match stays pending and is counted in `events_unresolved`; that count does not alert on its own ([event-matching](../storage/event-matching.md)). `odds_lambda.jobs.fetch_oddsportal_results.record_run_outcome`, called by `main` after each run, writes a `results_settled:<sport>` row to `odds_core.models.AlertHistory` when the run settled events. When a run settles none, it sends `results_stalled:<sport>` if events that kicked off after the latest such row are still pending more than 24 h on, so unmatched events from before the job's last settling run stay quiet, as does a job with no settling run on record yet. `odds_lambda.health_monitor.HealthMonitor.check_unsettled_events` covers single stuck league events ([health monitor](../operations/health-monitor.md)). `odds_lambda.jobs.fetch_oddsportal_results._raise_if_feed_parsed_nothing` raises when pending events older than 24 h exist, the feed returned rows and none parsed as finished or cancelled, because that means the parser cannot read the page. `odds fetch results --sport S [--lookback-days N]` (`odds_cli.commands.fetch.fetch_results`) runs the same settle with a chosen lookback, or with every pending event when the option is omitted, for backfills. `odds_lambda.jobs.fetch_oddsportal_results.ResultsStats.events_cancelled` counts the cancellations. The job fails, with a critical alert and no heartbeat, when updating any event in the database raised, after the other events are settled (`odds_lambda.jobs.fetch_oddsportal_results.ResultsStats.errors`). Rerunning is idempotent because settled events leave the pending query. No postponed row has been observed on the EPL or MLB results pages, so a postponed match stays pending until ESPN marks it POSTPONED ([event-status-lifecycle](event-status-lifecycle.md#who-sets-what)).

`main` runs `odds_lambda.jobs.fetch_oddsportal_results.process_results` inside `odds_core.alerts.job_alert_context`, then in `finally` books tomorrow 08:00 UTC through `odds_lambda.scheduling.helpers.self_schedule`, ignoring overnight windows. A scheduling failure sends `odds_core.alerts.send_critical` (wrapped so it cannot mask the original exception). It has no pre-schedule, so a kill before `finally` (the scraper Lambda timeout is 900 s) skips the reschedule and ends the chain (#438). With `ctx.sport=None` it processes `odds_lambda.jobs.fetch_oddsportal_results.DEFAULT_SPORT_KEY` (`soccer_epl`) but reschedules under the bare job name, which matches no per-sport rule (#470). On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).

## Heartbeats

The results job's heartbeat expectation is 26 h per sport (`heartbeat:fetch-oddsportal-results-<suffix>`). `update-status` writes `heartbeat:update-status` but has no default expectation, and `fetch-scores` is expected every 13 h per sport (the 12 h idle interval plus 1 h; no overnight pause) ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [Event status lifecycle](event-status-lifecycle.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Odds API ingestion service](odds-ingestion-service.md)
