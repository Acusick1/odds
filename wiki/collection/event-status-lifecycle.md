---
type: explanation
title: Event status lifecycle
description: The five statuses, which job or command sets each, and what each status gates downstream.
sources:
  - odds_core.models.EventStatus
  - odds_lambda.storage.writers.OddsWriter.update_event_status
  - odds_lambda.jobs.update_status._update_event_statuses
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_lambda.jobs.fetch_oddsportal_results.process_results
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.paper_trading
status: stable
---

# Event status lifecycle

`odds_core.models.EventStatus` has five values: `scheduled`, `live`, `final`, `cancelled`, `postponed`. New [events](../storage/events.md) default to SCHEDULED. All transitions go through `odds_lambda.storage.writers.OddsWriter.update_event_status`, which stamps `updated_at` and, only for FINAL with both scores, sets `home_score`, `away_score` and `completed_at` ([odds-writer](../storage/odds-writer.md)). Three jobs and two commands write status; only LIVE and FINAL are ever written. The jobs themselves are described on [results-jobs](results-jobs.md).

## Who sets what

| Transition | Setter | Rule |
|---|---|---|
| (new) → SCHEDULED | event sync (`odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`), `odds_lambda.storage.writers.OddsWriter.find_or_create_event`, `odds discover games` | default status |
| SCHEDULED → LIVE | `odds_lambda.jobs.update_status._update_event_statuses` | events already past `commence_time` within `odds_lambda.jobs.update_status.STATUS_WINDOW`; `reader.get_events_by_date_range(status=SCHEDULED)` |
| → FINAL | `odds_lambda.jobs.fetch_scores._fetch_and_update_scores` | Odds API `/scores` ([results-jobs](results-jobs.md#fetch-scores)), only when `completed` and both scores parse |
| → FINAL | `odds_lambda.jobs.fetch_oddsportal_results.process_results` | OddsPortal results for SCHEDULED/LIVE events already past kickoff (`odds_lambda.jobs.fetch_oddsportal_results.get_pending_events`) |
| → FINAL | `odds fetch scores` (`odds_cli.commands.fetch.fetch_scores`), `odds backfill scores` | manual; `backfill scores` sets FINAL from NBA game logs ([odds-api-backfill](odds-api-backfill.md)) |

No code sets `cancelled` or `postponed`. A cancelled game stays SCHEDULED or LIVE until a results job finalises it or it ages out of the status and scores windows ([results-jobs](results-jobs.md)) (#440).

Re-discovery can move a status backwards: `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` overwrites status and scores, so a re-discovered FINAL event returns to SCHEDULED with null scores ([events](../storage/events.md), #440). Whether `/events` still returns started games, which would make event sync reset LIVE events, is not verified.

`update-status` and `fetch-scores` self-schedule with a backward decision. The status job counts only SCHEDULED events as needing work, the scores job any non-FINAL event in its window ([self-scheduling](../scheduling/self-scheduling.md)).

## What each status gates

- Schedule-driven readers filter on SCHEDULED: `odds_lambda.scheduling.helpers.get_next_kickoff`, `odds_lambda.jobs.score_predictions` (future `commence_time` only), and three of the MCP tools in `odds_mcp.server`: `odds_mcp.server.get_upcoming_fixtures` and `odds_mcp.server.get_slate_briefs` list SCHEDULED events only, and `odds_mcp.server.paper_bet` rejects an event that is not SCHEDULED. The other MCP tools look an event up by id with no status filter. The three depend on update-status running on the mini-PC ([mcp-market-tools](../agent/mcp-market-tools.md)).
- Training, standings and paper-trade settlement read only FINAL events (`odds_lambda.paper_trading`). Paper trades settle only on FINAL with both scores, so stalled results jobs leave bets unsettled ([paper-trading](../agent/paper-trading.md#settlement)).
- Flipping to LIVE removes a started game from pre-game proximity decisions, because the fetch-odds gate keys off SCHEDULED events.
- A status flip is not what stops pre-game fetching: the fetch job's gating is tier-based ([fetch-tier](../storage/fetch-tier.md)), and `odds_lambda.jobs.fetch_odds` has no status check. `odds_lambda.jobs.update_status._update_event_statuses` only flips the status; nothing in it stops later odds fetches for a started game.

## Related

- [Results and status jobs](results-jobs.md)
- [Events table](../storage/events.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Job deployment matrix](../scheduling/job-deployment-matrix.md)
