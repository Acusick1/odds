---
type: process
title: Reporting surfaces and the daily digest
description: What the system reports and through which channel (Discord, odds status, MCP, traces), and the opt-in, local-only EPL daily-digest job with its single-model, home-side convention.
sources:
  - packages/odds-cli/odds_cli/commands/status.py
  - packages/odds-cli/odds_cli/commands/scheduler.py::health
  - packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py::send_digest
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py::build_digest_embed
status: stable
---

# Reporting surfaces and the daily digest

"Reporting" is everything the system emits about itself (operations) or its outputs (predictions, paper trades, agent briefs). It has three modes.

| Mode | Producers | Output | Backing tables |
|---|---|---|---|
| Event-driven | jobs via `packages/odds-core/odds_core/alerts.py::job_alert_context`, `packages/odds-core/odds_core/alerts.py::send_job_warning`, quota alerts | Discord | `alert_history` (rate limit, heartbeats) |
| Poll-driven | check-health, the daily digest | Discord summary | reads `fetch_logs`, `odds_snapshots`, `data_quality_logs`, `predictions`, `alert_history` |
| On-demand | `odds status`, `odds scheduler`, MCP tools | terminal tables or JSON | reads everything |

Delivery, rate limiting, heartbeats and check-health are in [alerts, heartbeats and the health monitor](alerts-and-health.md); `fetch_logs` is described with [Odds API ingestion](../collection/odds-api-ingestion.md); scoring output in [publishing and scoring](../modelling/model-publishing-and-scoring.md). Predictions exist only where scoring runs (a local scheduler), one row per snapshot, with `model_version` an S3 ETag.

## CLI status

`packages/odds-cli/odds_cli/commands/status.py` prints Rich tables only; no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, odds and snapshot totals, quota and 24 h success rate from `OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota plus the last 10 `packages/odds-core/odds_core/models.py::FetchLog` rows. It shows nothing when the newest row is Betfair (null quota) ([quota](../collection/odds-api-client-and-quota.md)).
- `odds status events --days N --team T` (default 7 days): recent events with status and scores.
- `odds scheduler list-jobs` and `packages/odds-cli/odds_cli/commands/scheduler.py::health` report the scheduler backend; `odds scheduler health` calls the backend's `health_check()` and prints a green or red overall status, the backend name, the passed and failed checks and any details, exiting 1 when unhealthy or on error ([local scheduler](../scheduling/local-scheduler.md), [scheduler backends](../scheduling/scheduler-backends.md)); `list-jobs` catches `packages/odds-lambda/odds_lambda/scheduling/exceptions.py::BackendUnavailableError` for Railway.

## MCP and traces

The agent reads state through MCP tools in `packages/odds-mcp/odds_mcp/server.py`: `packages/odds-mcp/odds_mcp/server.py::get_scrape_status`, `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs`, `packages/odds-mcp/odds_mcp/server.py::get_predictions`, `packages/odds-mcp/odds_mcp/server.py::get_portfolio`, `packages/odds-mcp/odds_mcp/server.py::settle_bets`, `packages/odds-mcp/odds_mcp/server.py::save_match_brief`, `packages/odds-mcp/odds_mcp/server.py::get_slate_briefs`, `packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup` ([MCP server](../agent/mcp-server.md), [match briefs and paper trades](../agent/match-briefs-and-paper-trades.md)). `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` matches jobs by sport suffix (NBA matches nothing) and reads only the local APScheduler store. Agent results surface only through MCP `packages/odds-mcp/odds_mcp/server.py::get_portfolio` and `odds paper`; the digest does not read paper trades. Agent run output is a JSONL trace under `logs/agent_runs/`, viewed with `experiments/scripts/show_agent_summaries.py` ([betting agent](../agent/betting-agent.md)).

## The daily-digest job

`packages/odds-lambda/odds_lambda/jobs/daily_digest.py` posts one Discord embed per sport with model predictions. It is opt-in and local-only: no EventBridge rule exists, it is commented out of the default `bootstrap_jobs`, and when listed in `SCHEDULER_BOOTSTRAP_JOBS` it runs from `_JOB_CRON_MAP` at `0 8 * * *` for EPL only. `odds scheduler smoke --no-side-effects` drops it ([jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs), [local scheduler](../scheduling/local-scheduler.md)). Its heartbeat key `daily-digest-epl` is expected every 26 h.

`main(ctx)` runs `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::send_digest` inside `job_alert_context("daily-digest-<suffix>")`, with `ctx.lookback_hours` (24) and `ctx.lookahead_hours` (48).

- Completed: `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::_get_completed_events_with_predictions` selects FINAL events with `completed_at >= now - lookback`, joined to the prediction with the highest `snapshot_id` per event for `model_name`, ordered by kickoff.
- Upcoming: `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::_get_upcoming_events_with_predictions` selects SCHEDULED events kicking off in `(now, now + lookahead]` with the same join, ordered by `|predicted_clv|` descending.
- `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::build_digest_embed` makes a green embed titled `<EPL|NBA|MLB> Daily Digest` with fields "Post-Match Results" (tick or cross, score, side and CLV, footer `X events | Y/X correct side`) and "Upcoming Predictions". Each field is truncated at `MAX_FIELD_CHARS` (1024).
- Delivery is `alert_manager.send_embed`. If both lists are empty nothing is sent and the log says `daily_digest_empty`.

`model_name` defaults to `settings.model.name`, falling back to `"epl-clv-home"`.

## Hazards

- **Single-model, home-side digest.** `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::_value_side` treats positive predicted CLV as "Home" and negative as "Away"; `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::_result_hit` counts a draw as a miss for both sides. There is no draw side and no multi-model digest.
- **The digest never reads paper trades or P&L**; those exist only on the MCP and CLI side.
- **`sent` can be wrong.** `sent` in the returned stats is set to 1 whenever an embed was built, even if alerts are disabled or the webhook post failed.
- **EPL-only by allowlist.** `_SPORT_DISPLAY_NAMES` is hard-coded for EPL, NBA, MLB, and the cron allowlist limits the job to EPL, matching the EPL-only heartbeat expectation.
- `packages/odds-lambda/odds_lambda/jobs/daily_digest.py::_get_upcoming_events_with_predictions` also selects `OddsSnapshot.snapshot_time`, which the embed never uses.
- Discord is the only alert channel and failure to deliver is only logged.
- `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` and `packages/odds-mcp/odds_mcp/server.py::get_scrape_status` open the local APScheduler store (`packages/odds-lambda/odds_lambda/scheduling/backends/local.py::build_scheduler`) whatever `SCHEDULER_BACKEND` says, so on AWS they do not show EventBridge rules.
- `odds status events` has no sport filter and `odds status show` mixes all fetch sources, so Betfair rows can distort the success rate and quota.
- Sport naming differs by surface: compound job names (`daily-digest-epl`), sport keys (`soccer_epl`) and display names (`EPL`) each have their own translation table ([jobs](../scheduling/jobs-and-deployment-matrix.md)).

## Related

- [Alerts and health](alerts-and-health.md), [Jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md), [Packages and CLI](../foundations/packages-and-cli.md)
