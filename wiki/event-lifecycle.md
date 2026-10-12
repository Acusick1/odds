---
type: concept
title: Event lifecycle
summary: The five EventStatus values, which job or command sets each (only LIVE and FINAL are ever written), and what each status gates.
sources:
  - packages/odds-core/odds_core/models.py::EventStatus
  - packages/odds-lambda/odds_lambda/jobs/update_status.py
  - packages/odds-lambda/odds_lambda/jobs/fetch_scores.py
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py
  - packages/odds-lambda/odds_lambda/storage/writers.py::update_event_status
  - docs/ARCHITECTURE.md#event-lifecycle
  - docs/ARCHITECTURE.md#update-status-job
  - docs/ARCHITECTURE.md#fetch-scores-job
  - docs/ARCHITECTURE.md#oddsportal-results-job
verified_at: 7887039
---

# Event lifecycle

`EventStatus` (`packages/odds-core/odds_core/models.py::EventStatus`) has five values: `scheduled`, `live`, `final`, `cancelled`, `postponed`. New [events](table-events.md) default to SCHEDULED. All transitions go through `OddsWriter.update_event_status` (`packages/odds-lambda/odds_lambda/storage/writers.py::update_event_status`), which stamps `updated_at` and, only for FINAL with both scores, sets `home_score`, `away_score` and `completed_at`.

## Who sets what

| Transition | Setter | Rule |
|---|---|---|
| (new) → SCHEDULED | event sync, [OddsWriter](odds-writer.md), `odds discover games` | default status |
| SCHEDULED → LIVE | `packages/odds-lambda/odds_lambda/jobs/update_status.py::_update_event_statuses` | events with `commence_time` in the last 4 hours (`STATUS_WINDOW`) and already past; `reader.get_events_by_date_range(status=SCHEDULED)` |
| → FINAL | `packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::_fetch_and_update_scores` | Odds API `/scores` (`days_from=3`), only when `completed` and both scores parse |
| → FINAL | `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::process_results` | OddsPortal results for SCHEDULED/LIVE events already past kickoff (`get_pending_events`) |
| → FINAL | `odds fetch scores` (`packages/odds-cli/odds_cli/commands/fetch.py`), `odds backfill scores` | manual; see [Historical backfill](howto-historical-backfill.md) |

[update-status](job-update-status.md) and [fetch-scores](job-fetch-scores.md) self-schedule via `decide_backward_resilient` in `packages/odds-lambda/odds_lambda/scheduling/decision.py`; the status job counts only SCHEDULED events as needing work, the scores job counts any non-FINAL event in its 3-day window. [fetch-oddsportal-results](job-fetch-oddsportal-results.md) reschedules for 08:00 UTC next day in a `finally` block, so a failed scrape does not break the chain. See [Self-scheduling](self-scheduling.md).

## What each status stops

- Schedule-driven readers filter on SCHEDULED: `get_next_kickoff` in `packages/odds-lambda/odds_lambda/scheduling/helpers.py`, `packages/odds-lambda/odds_lambda/jobs/score_predictions.py`, and the MCP tools in `packages/odds-mcp/odds_mcp/server.py`.
- Training, standings and paper-trade settlement read only FINAL events (e.g. `packages/odds-lambda/odds_lambda/paper_trading.py`).
- A status flip is not what stops pre-game fetching: the fetch job's gating is tier-based ([fetch tiers](fetch-tier.md)), not status-based (no status check in `packages/odds-lambda/odds_lambda/jobs/fetch_odds.py`).

## Hazards

- Docs say CANCELLED and POSTPONED mean "no further data collection". No code in `packages` assigns either value; they are enum members only. A cancelled game stays SCHEDULED/LIVE until a results job finalises it or it ages out of the 4-hour/3-day windows.
- `docs/ARCHITECTURE.md#update-status-job` says the job "prevents fetching pre-game odds for started games". Code only flips the status; see above.
- `update_status` ignores SCHEDULED events older than 4 hours, so those can only reach FINAL via the scores or results jobs.
- `OddsWriter.bulk_upsert_events` (used by `odds discover games` and by `packages/odds-lambda/odds_lambda/event_sync.py::EventSyncService`) overwrites every non-key column, including `status` and scores, from the incoming event. Re-discovering a date range resets FINAL events to SCHEDULED with null scores. Whether `/events` still returns started games, which would make event sync reset LIVE events, is not verified.
- Whether the jobs are enabled in AWS cannot be verified from the repo; check `odds scheduler list-jobs --backend aws`.

## Related

- [Events table](table-events.md)
- [Job deployment matrix](job-deployment-matrix.md)
- [Odds ingestion flow](odds-ingestion-flow.md)
