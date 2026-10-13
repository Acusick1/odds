---
type: process
title: Event status lifecycle and results jobs
description: "The five event statuses, which job or command sets each (update-status to LIVE; fetch-scores, fetch-oddsportal-results and CLI to FINAL), each job's cadence and failure modes, and what each status gates."
sources:
  - odds_core.models.EventStatus
  - odds_lambda.jobs.update_status.main
  - odds_lambda.jobs.update_status._update_event_statuses
  - odds_lambda.jobs.fetch_scores.main
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_lambda.jobs.fetch_oddsportal_results.main
  - odds_lambda.jobs.fetch_oddsportal_results.process_results
  - odds_lambda.storage.writers.OddsWriter.update_event_status
  - odds_lambda.scheduling.decision.decide_backward_resilient
  - odds_cli.commands.fetch.fetch_scores
status: stable
---

# Event status lifecycle and results jobs

`odds_core.models.EventStatus` has five values: `scheduled`, `live`, `final`, `cancelled`, `postponed`. New [events](../storage/events.md) default to SCHEDULED. All transitions go through `odds_lambda.storage.writers.OddsWriter.update_event_status`, which stamps `updated_at` and, only for FINAL with both scores, sets `home_score`, `away_score` and `completed_at` ([odds-storage](../storage/odds-storage.md)). Three jobs and two commands write status; only LIVE and FINAL are ever written.

## Who sets what

| Transition | Setter | Rule |
|---|---|---|
| (new) → SCHEDULED | event sync, OddsWriter, `odds discover games` | default status |
| SCHEDULED → LIVE | `odds_lambda.jobs.update_status._update_event_statuses` | events with `commence_time` in the last 4 hours (`odds_lambda.jobs.update_status.STATUS_WINDOW`) and already past; `reader.get_events_by_date_range(status=SCHEDULED)` |
| → FINAL | `odds_lambda.jobs.fetch_scores._fetch_and_update_scores` | Odds API `/scores` (`days_from=3`), only when `completed` and both scores parse |
| → FINAL | `odds_lambda.jobs.fetch_oddsportal_results.process_results` | OddsPortal results for SCHEDULED/LIVE events already past kickoff (`odds_lambda.jobs.fetch_oddsportal_results.get_pending_events`) |
| → FINAL | `odds fetch scores` (`odds_cli.commands.fetch.fetch_scores`), `odds backfill scores` | manual; `backfill scores` sets FINAL from NBA game logs ([howto-load-historical-odds](howto-load-historical-odds.md)) |

`update-status` and `fetch-scores` self-schedule via `odds_lambda.scheduling.decision.decide_backward_resilient` (`odds_lambda.scheduling.decision`); the status job counts only SCHEDULED events as needing work, the scores job any non-FINAL event in its 3-day window. `fetch-oddsportal-results` reschedules for 08:00 UTC next day in a `finally` block. See [self-scheduling](../scheduling/self-scheduling.md).

## Where each job runs

`update-status` and `fetch-scores` run only on the scheduler Lambda (not in the local default bootstrap); the OddsPortal results job runs on the scraper Lambda or in the local default bootstrap. Deployed state: `odds scheduler list-jobs --backend aws` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)).

## update-status

`odds_lambda.jobs.update_status.main(ctx)` flips SCHEDULED events whose `commence_time` has passed to LIVE. It is a global job (not per-sport): one rule `odds-update-status`, always created by Terraform. It runs for `data_collection.sports` when no sport is given.

1. `odds_lambda.jobs.update_status._status_decision` calls `odds_lambda.scheduling.decision.decide_backward_resilient` with a 4-hour window, 1 h active and 6 h idle intervals, counting only `{EventStatus.SCHEDULED}`.
2. Pre-schedule under `update-status`, skip if idle.
3. Inside `job_alert_context("update-status")`, `odds_lambda.jobs.update_status._update_event_statuses` reads SCHEDULED events with `commence_time` in the last four hours and calls `update_event_status(LIVE)` for each that has started.
4. Re-decide and post-schedule.

Because the fetch-odds gate keys off SCHEDULED events, flipping to LIVE removes a started game from pre-game proximity decisions.

## fetch-scores

`odds_lambda.jobs.fetch_scores.main(ctx)` is the Odds API `/scores` job, a per-sport rule on the scheduler Lambda.

1. Gate with `odds_lambda.scheduling.decision.decide_backward_resilient` using `odds_lambda.jobs.fetch_scores.SCORES_WINDOW` (3 days), `odds_lambda.jobs.fetch_scores.SCORES_ACTIVE_INTERVAL` 6 h and `odds_lambda.jobs.fetch_scores.SCORES_IDLE_INTERVAL` 12 h: it runs while any event that kicked off in the last three days is not FINAL.
2. Pre-schedule, then skip if idle.
3. `odds_lambda.jobs.fetch_scores._fetch_and_update_scores`: for each sport call `TheOddsAPIClient.get_scores(sport, days_from=3)`, and for every `completed` entry with parseable scores call `update_event_status(FINAL, home_score, away_score)`, committing per sport ([odds-api-client-and-quota](odds-api-client-and-quota.md)).
4. Re-decide and post-schedule.

Per-event failures are logged and skipped. A whole-job failure sends `odds_core.alerts.send_critical` and re-raises. `odds fetch scores` repeats the same loop inline with `--sport` (default `basketball_nba`) and `--days` (default 3); it does not call the job and does not gate or reschedule.

## fetch-oddsportal-results

`odds_lambda.jobs.fetch_oddsportal_results.main(ctx)` scrapes OddsPortal's historic results page (one page, the spec's primary market) for past matches, sets events FINAL with scores and stores one closing snapshot per event. League config is `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` from [oddsportal](oddsportal.md). `odds_lambda.jobs.fetch_oddsportal_results.process_results` uses three session scopes so no connection is held across the browser run (Neon closes idle connections):

1. **Read**: `odds_lambda.jobs.fetch_oddsportal_results.get_pending_events` returns plain `odds_lambda.jobs.fetch_oddsportal_results.PendingEvent` dataclasses for SCHEDULED or LIVE events of the sport with `commence_time` in the past. None means exit.
2. **Scrape**: `run_harvester_historic(spec)` with no open session.
3. **Write**: for each record, `odds_lambda.jobs.fetch_oddsportal_results._match_record_to_event` matches on exact home and away team strings and a commence-time delta of at most 24 h; `update_event_status(FINAL, scores)`; `build_raw_data(... use_opening=False)` and `store_odds_snapshot` tagged `api_request_id="oddsportal_closing"`; commit per record with rollback on error. Records with no digit scores are skipped.

Rerunning is idempotent because finished events leave the pending query. `odds_lambda.jobs.fetch_oddsportal_results._db_market_for_spec` maps `home_away` to `h2h` and `1x2` to `1x2`, and raises for other primary markets.

`odds_lambda.jobs.fetch_oddsportal_results.main` runs `odds_lambda.jobs.fetch_oddsportal_results.process_results` inside `odds_core.alerts.job_alert_context`, then in `finally` books tomorrow 08:00 UTC through `odds_lambda.scheduling.helpers.self_schedule`, ignoring overnight windows. A scheduling failure sends `odds_core.alerts.send_critical` (wrapped so it cannot mask the original exception).

## What each status gates

- Schedule-driven readers filter on SCHEDULED: `odds_lambda.scheduling.helpers.get_next_kickoff` in `odds_lambda.scheduling.helpers`, `odds_lambda.jobs.score_predictions`, and the MCP tools in `odds_mcp.server` (`odds_mcp.server.get_upcoming_fixtures` returns SCHEDULED only, so it depends on update-status running on AWS; [mcp-market-tools](../agent/mcp-market-tools.md)).
- Training, standings and paper-trade settlement read only FINAL events (`odds_lambda.paper_trading`). Paper trades settle only on FINAL with both scores, so stalled results jobs leave bets unsettled ([match-briefs-and-paper-trades](../agent/match-briefs-and-paper-trades.md#settlement)).
- A status flip is not what stops pre-game fetching: the fetch job's gating is tier-based ([fetch-tier](../storage/fetch-tier.md)), not status-based (no status check in `odds_lambda.jobs.fetch_odds`).

## Heartbeats

The results job's heartbeat expectation is 26 h per sport (`heartbeat:fetch-oddsportal-results-<suffix>`). `update-status` writes `heartbeat:update-status` but has no default expectation, and `fetch-scores` is not wrapped in `odds_core.alerts.job_alert_context` so it writes none ([alerts-and-health](../operations/alerts-and-health.md#heartbeats)).

## Hazards

- **CANCELLED and POSTPONED are never set.** `odds_core.models.EventStatus` defines both, but no code in `packages` assigns either value. A cancelled game stays SCHEDULED/LIVE until a results job finalises it or it ages out of the 4-hour/3-day windows; `odds_lambda.scheduling.decision.decide_backward` counts it as needing an update, keeping `fetch-scores` on its 6 h cadence for three days.
- **Re-discovery resets status.** `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` (used by `odds discover games` and `odds_lambda.event_sync.EventSyncService`, i.e. every fetch-odds run) overwrites every non-key column, including `odds_core.models.Event.status` and scores. Re-discovering a date range resets FINAL events to SCHEDULED with null scores; `upsert_event` does not. Whether `/events` still returns started games, which would make event sync reset LIVE events, is not verified ([events](../storage/events.md#hazards)).
- **update-status doesn't stop fetching.** `odds_lambda.jobs.update_status._update_event_statuses` only flips the status; nothing in it stops later odds fetches for a started game.
- **update-status scope and age limit.** `odds_lambda.jobs.update_status._update_event_statuses` does not filter by sport while the gate does; when a per-sport schedule fires it flips events of every sport. Only events that kicked off within the last 4 hours are eligible, so if the job was down longer, older SCHEDULED events can only reach FINAL via the scores or results jobs.
- **Stale docstring.** The `update_status` function docstring cites a nonexistent scheduler jobs module.
- **Results job has no pre-schedule.** If the process is killed before `finally` (scraper Lambda timeout 900 s), the chain breaks until something re-invokes the job.
- **Results job bare-name reschedule.** `ctx.sport=None` processes `odds_lambda.jobs.fetch_oddsportal_results.DEFAULT_SPORT_KEY` (`soccer_epl`) but reschedules under the bare job name, which matches no per-sport rule.
- **Exact-string matching.** Results matching is exact team-string equality; an unmatched scrape increments `events_not_matched` and nothing alerts ([events](../storage/events.md#matchers)). The OddsHarvester call here does not take the Playwright semaphore.
- **`fetch-scores` quirks.** It sends `daysFrom`, so each `/scores` call costs 2 units ([odds-api-client-and-quota](odds-api-client-and-quota.md)); no heartbeat and an unrate-limited failure alert; the scoring loop is duplicated in the job and the CLI, so a change to result parsing needs both.

## Related

- [Events table and matching](../storage/events.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Odds API ingestion](odds-api-ingestion.md)
- [Jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md)
