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
  - odds_lambda.jobs.fetch_oddsportal_results._match_record_to_event
  - odds_lambda.jobs.fetch_oddsportal_results._db_market_for_spec
  - odds_lambda.scheduling.decision.decide_backward
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_cli.commands.fetch.fetch_scores
status: stable
---

# Results and status jobs

Three jobs and one command move events to LIVE and FINAL ([event-status-lifecycle](event-status-lifecycle.md)). All three jobs are in the default local `bootstrap_jobs` and run on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)); their AWS rules are dormant.

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

Per-event failures are logged and skipped. A whole-job failure sends `odds_core.alerts.send_critical` and re-raises. Each `/scores` call costs 2 units because the client sends `daysFrom` (#441; [odds-api-client](odds-api-client.md#cost-per-call)). The job is not wrapped in `odds_core.alerts.job_alert_context`, so it writes no heartbeat (#438), and its failure alert is not rate-limited (#442).

`odds fetch scores` repeats the same loop inline with `--sport` (default `basketball_nba`) and `--days` (default 3); it does not call the job and does not gate or reschedule. The update loop is duplicated in `odds_cli.commands.fetch._fetch_scores` and `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`, so a change to result parsing needs both (#463). The job loops sports and alerts; the CLI takes one sport.

## fetch-oddsportal-results

`odds_lambda.jobs.fetch_oddsportal_results.main(ctx)` scrapes OddsPortal's historic results page (one page, the spec's primary market) for past matches, sets events FINAL with scores and stores one closing snapshot per event. League config is `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)); only its `sport` and `league` names are used, since the live job's `listing_path` is HTTP-only. `odds_lambda.jobs.fetch_oddsportal_results.process_results` uses three session scopes so no connection is held across the browser run (Neon closes idle connections):

1. **Read**: `odds_lambda.jobs.fetch_oddsportal_results.get_pending_events` returns plain `odds_lambda.jobs.fetch_oddsportal_results.PendingEvent` dataclasses for SCHEDULED or LIVE events of the sport with `commence_time` in the past. None means exit.
2. **Scrape**: `run_harvester_historic(spec)` with no open session. The call does not take the Playwright semaphore.
3. **Write**: for each record, `odds_lambda.jobs.fetch_oddsportal_results._match_record_to_event` matches on exact home and away team strings and a commence-time delta of at most 24 h; `update_event_status(FINAL, scores)`; `build_raw_data(... use_opening=False)` and `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` tagged `api_request_id="oddsportal_closing"`; commit per record with rollback on error. Records with no digit scores are skipped.

An unmatched scrape increments `events_not_matched` and nothing alerts ([event-matching](../storage/event-matching.md)). Rerunning is idempotent because finished events leave the pending query. `odds_lambda.jobs.fetch_oddsportal_results._db_market_for_spec` maps `home_away` to `h2h` and `1x2` to `1x2`, and raises for other primary markets.

`main` runs `odds_lambda.jobs.fetch_oddsportal_results.process_results` inside `odds_core.alerts.job_alert_context`, then in `finally` books tomorrow 08:00 UTC through `odds_lambda.scheduling.helpers.self_schedule`, ignoring overnight windows. A scheduling failure sends `odds_core.alerts.send_critical` (wrapped so it cannot mask the original exception). It has no pre-schedule, so a kill before `finally` (the scraper Lambda timeout is 900 s) skips the reschedule and ends the chain (#438). With `ctx.sport=None` it processes `odds_lambda.jobs.fetch_oddsportal_results.DEFAULT_SPORT_KEY` (`soccer_epl`) but reschedules under the bare job name, which matches no per-sport rule (#470).

## Heartbeats

The results job's heartbeat expectation is 26 h per sport (`heartbeat:fetch-oddsportal-results-<suffix>`). `update-status` writes `heartbeat:update-status` but has no default expectation, and `fetch-scores` writes none ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Related

- [Event status lifecycle](event-status-lifecycle.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Odds API ingestion service](odds-ingestion-service.md)
