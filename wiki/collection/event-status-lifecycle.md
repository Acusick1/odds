---
type: explanation
title: Event status lifecycle
description: The six statuses, which job or command sets each, and what each status gates downstream.
sources:
  - odds_core.models.EventStatus
  - odds_lambda.storage.writers.OddsWriter.update_event_status
  - odds_lambda.jobs.update_status._update_event_statuses
  - odds_lambda.jobs.fetch_scores._fetch_and_update_scores
  - odds_lambda.jobs.fetch_oddsportal_results.process_results
  - odds_lambda.jobs.fetch_mlb_results.process_results
  - odds_lambda.jobs.fetch_espn_fixtures._mark_postponed_events
  - odds_lambda.storage.espn_fixture_reader.EspnFixtureReader.get_postponed_home_fixtures
  - odds_lambda.storage.writers.OddsWriter.bulk_upsert_events
  - odds_core.models.EventStatus.can_become
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.paper_trading
  - scripts/ingest_football_data_uk.py
status: stable
---

# Event status lifecycle

`odds_core.models.EventStatus` has six values: `scheduled`, `live`, `final`, `cancelled`, `postponed`, `unmatched`. Unmatched marks an MLB event with no game in MLB's schedule; no code sets it. New [events](../storage/events.md) default to SCHEDULED. Every transition but one goes through `odds_lambda.storage.writers.OddsWriter.update_event_status` (the FDUK loader sets FINAL on the event directly), which stamps `updated_at`, refuses FINAL without both scores, refuses a move that `odds_core.models.EventStatus.can_become` does not allow, and for FINAL sets `home_score`, `away_score` and `completed_at` ([odds-writer](../storage/odds-writer.md)). Five jobs, the FDUK loader and two commands write status. The jobs themselves are described on [results-jobs](results-jobs.md).

## Allowed moves

`odds_core.models.EventStatus.can_become` is the one definition. A status may always be rewritten to itself.

| From | May become |
|---|---|
| SCHEDULED | LIVE, FINAL, CANCELLED, POSTPONED, UNMATCHED |
| LIVE | SCHEDULED, FINAL, CANCELLED, POSTPONED, UNMATCHED |
| POSTPONED | SCHEDULED, FINAL, CANCELLED, UNMATCHED |
| CANCELLED | FINAL |
| FINAL | nothing else |
| UNMATCHED | SCHEDULED |

A result is the strongest evidence, so every status but FINAL and UNMATCHED can reach FINAL, and FINAL never leaves it. An UNMATCHED event has no game to take a result from, and leaves only for SCHEDULED.

## Who sets what

| Transition | Setter | Rule |
|---|---|---|
| (new) → SCHEDULED | event sync (`odds_lambda.storage.writers.OddsWriter.bulk_upsert_events`), `odds_lambda.storage.writers.OddsWriter.find_or_create_event` (only when no event matches, by `gamePk` or teams for MLB), `odds discover games` | default status |
| SCHEDULED → LIVE | `odds_lambda.jobs.update_status._update_event_statuses` | events already past `commence_time` within `odds_lambda.jobs.update_status.STATUS_WINDOW`; `reader.get_events_by_date_range(status=SCHEDULED)` |
| → FINAL | `odds_lambda.jobs.fetch_scores._fetch_and_update_scores` | Odds API `/scores` ([results-jobs](results-jobs.md#fetch-scores)), only when `completed` and both scores parse |
| → FINAL | `odds_lambda.jobs.fetch_oddsportal_results.process_results` | EPL: OddsPortal finished matches for events already past kickoff that are SCHEDULED, LIVE, POSTPONED, or FINAL without scores (`odds_lambda.results_run.get_pending_events`); a run that settles nothing alerts ([results-jobs](results-jobs.md#fetch-oddsportal-results)) |
| → CANCELLED | `odds_lambda.jobs.fetch_oddsportal_results.process_results` | EPL: OddsPortal results row with status 5 ("Canceled") matched to a pending SCHEDULED or LIVE event ([results-jobs](results-jobs.md#fetch-oddsportal-results)) |
| → FINAL, CANCELLED, POSTPONED | `odds_lambda.jobs.fetch_mlb_results.process_results` | MLB: the state of the game a pending event's `gamePk` names in the MLB Stats API schedule; a run that settles nothing alerts ([results-jobs](results-jobs.md#fetch-mlb-results)) |
| → POSTPONED | `odds_lambda.jobs.fetch_espn_fixtures._mark_postponed_events` | a Premier League `espn_fixtures` row with status "Postponed" at the event's kick-off (within the event-matching window), for a SCHEDULED or LIVE event ([EPL context data](epl-context-data.md#espn-fixtures)) |
| POSTPONED → SCHEDULED | `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` | the re-discovered event's `commence_time` differs from the stored one |
| POSTPONED → SCHEDULED | `odds_lambda.storage.event_kickoff.move_kickoff` ([Kick-off authority](../storage/kickoff-authority.md)), from the live OddsPortal and Betfair jobs, `fetch-mlb-results` and `odds events assign-game-pks` / `set-game-pk` | an MLB event moves to its game's scheduled start ([Event matching](../storage/event-matching.md#live-mlb-events)) |
| LIVE → SCHEDULED | `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` | the re-discovered event's `commence_time` is after the clock passed to the call (a delayed kick-off) |
| → FINAL | `scripts/ingest_football_data_uk.py` | an existing event with no scores, whatever its status, gets FDUK's full-time score and FINAL ([historical-ingest-scripts](historical-ingest-scripts.md)) |
| → FINAL | `odds fetch scores` (`odds_cli.commands.fetch.fetch_scores`), `odds backfill scores` | manual; `backfill scores` sets FINAL from NBA game logs ([odds-api-backfill](odds-api-backfill.md)) |

Re-discovery changes a status in exactly two cases, both back to SCHEDULED: a POSTPONED event whose `commence_time` differs, and a LIVE event whose incoming `commence_time` is in the future. Every other status, and every score and `completed_at`, is left alone: `odds_lambda.storage.writers.OddsWriter.bulk_upsert_events` ([events](../storage/events.md#upsert-verbs)).

Two kinds of orphan have no source signal and stay SCHEDULED or LIVE: a fixture whose kick-off moved by more than the matching window leaves its old `op_live_` event beside a new one (`op_live_ASTVIL_LIV_2026-05-17T1130` in development), and an EPL match that neither OddsPortal nor ESPN reports as cancelled or postponed. They age out of the status and scores windows ([results-jobs](results-jobs.md)). ESPN's "Canceled" description and OddsPortal's postponed status id have not been observed in collected data, so neither is mapped.

`update-status` and `fetch-scores` self-schedule with a backward decision. The status job counts only SCHEDULED events as needing work, the scores job any non-FINAL event in its window ([self-scheduling](../scheduling/self-scheduling.md)).

## What each status gates

- Schedule-driven readers filter on SCHEDULED: `odds_lambda.scheduling.helpers.get_next_kickoff`, and three of the MCP tools in `odds_mcp.server`: `odds_mcp.server.get_upcoming_fixtures` and `odds_mcp.server.get_slate_briefs` list SCHEDULED events only, and `odds_mcp.server.paper_bet` rejects an event that is not SCHEDULED, before any candidate check (the run-candidates job filters the same way). The other MCP tools look an event up by id with no status filter. The three depend on update-status running on the production host ([mcp-market-tools](../agent/mcp-market-tools.md)).
- Training, standings and paper-trade settlement read only FINAL events (`odds_lambda.paper_trading`). Paper trades settle on FINAL, with both scores present (on every [run-candidates](../evaluation/run-candidates-job.md#one-run) run among others), (graded by the market's outcomes, `h2h` or `1x2`, with profit from the trade's decimal price) and are voided on CANCELLED (a POSTPONED event's trades stay open), so stalled results jobs leave bets unsettled ([paper-trading](../agent/paper-trading.md#settlement)).
- Flipping to LIVE removes a started game from pre-game proximity decisions, because the fetch-odds gate keys off SCHEDULED events.
- A status flip is not what stops pre-game fetching: the fetch job's gating is tier-based ([fetch-tier](../storage/fetch-tier.md)), and `odds_lambda.jobs.fetch_odds.main`, which runs for one sport, has no status check at all (a gated run still writes its heartbeat; [flow](fetch-odds-job.md#flow)). `odds_lambda.jobs.update_status._update_event_statuses` only flips the status; nothing in it stops later odds fetches for a started game.

## Related

- [Results and status jobs](results-jobs.md)
- [Events table](../storage/events.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Job deployment matrix](../scheduling/job-deployment-matrix.md)
