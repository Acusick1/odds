---
type: component
title: fetch-scores job
summary: The Odds API scores job that marks events FINAL, its backward-looking cadence, and the `odds fetch scores` CLI copy of it.
sources:
  - docs/ARCHITECTURE.md#fetch-scores-job
  - docs/CLI.md#fetch-scores
  - packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::main
  - packages/odds-cli/odds_cli/commands/fetch.py::fetch_scores
verified_at: 7887039
---

# fetch-scores job

`packages/odds-lambda/odds_lambda/jobs/fetch_scores.py::main(ctx)` updates results for finished games from The Odds API `/scores` endpoint. On AWS it is a per-sport rule on the scheduler Lambda; locally it is not bootstrapped by default ([deployment matrix](job-deployment-matrix.md)). OddsPortal results are a second path: [fetch-oddsportal-results](job-fetch-oddsportal-results.md).

## Flow

1. Gate with `decide_backward_resilient` using `SCORES_WINDOW` (3 days), `SCORES_ACTIVE_INTERVAL` 6 h and `SCORES_IDLE_INTERVAL` 12 h. It runs while any event that kicked off in the last three days is not FINAL ([decision engine](scheduling-decision-engine.md)).
2. Pre-schedule, then skip if idle ([self-scheduling](self-scheduling.md)).
3. `_fetch_and_update_scores`: for each sport call `TheOddsAPIClient.get_scores(sport, days_from=3)`, and for every `completed` entry with parseable scores call `OddsWriter.update_event_status(FINAL, home_score, away_score)`, committing per sport ([odds API client](odds-api-client.md), [odds writer](odds-writer.md), [event lifecycle](event-lifecycle.md)).
4. Re-decide and post-schedule.

Per-event failures are logged and skipped. A whole-job failure sends `send_critical` and re-raises.

## CLI copy

`odds fetch scores` (`packages/odds-cli/odds_cli/commands/fetch.py::fetch_scores`) repeats the same fetch-and-update loop inline with `--sport` and `--days` (default 3). It does not call the job and does not gate or reschedule.

## Hazards

- `odds fetch scores --sport` defaults to `basketball_nba`, not the active EPL sport ([CLI overview](cli-overview.md)).
- Not wrapped in `job_alert_context`: no heartbeat row, unrate-limited failure alert ([alerts module](alerts-module.md)).
- Sending `daysFrom` makes each `/scores` call cost 2 units ([odds API quota](odds-api-quota.md)).
- `decide_backward` counts postponed and cancelled events as needing an update, so they keep the job on the 6 h cadence for three days.
- The scoring loop is duplicated in the job and the CLI, so a change to result parsing needs both.

## Related

- [update-status job](job-update-status.md), [odds ingestion flow](odds-ingestion-flow.md).
