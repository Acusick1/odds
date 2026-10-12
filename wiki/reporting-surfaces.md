---
type: process
title: Reporting surfaces
summary: "Event-driven, polled and on-demand reporting: what the system reports about itself and its outputs, through Discord, CLI status, MCP and traces."
sources:
  - docs/REPORTING.md#1-system-map
  - docs/REPORTING.md#5-cli-status-surfaces
  - docs/REPORTING.md#9-known-limitations
  - docs/REPORTING.md#10-reference-where-each-piece-lives
  - docs/CLI.md#status--monitoring
  - packages/odds-cli/odds_cli/commands/status.py
  - packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs
verified_at: 7887039
---

# Reporting surfaces

"Reporting" is everything the system emits about itself (operations) or its outputs (predictions, paper trades, agent briefs). It has three modes.

## Modes

| Mode | Producers | Output | Backing tables |
|---|---|---|---|
| Event-driven | jobs via `job_alert_context`, `send_job_warning`, quota alerts | Discord | `alert_history` (rate limit, heartbeats) |
| Poll-driven | [check-health](health-monitor.md), [daily-digest](job-daily-digest.md) | Discord summary | reads `fetch_logs`, `odds_snapshots`, `data_quality_logs`, `predictions`, `alert_history` |
| On-demand | `odds status`, `odds scheduler`, MCP tools | terminal tables or JSON | reads everything |

Delivery, rate limiting and heartbeats are in [alerts module](alerts-module.md); storage in [table alert_history](table-alert-history.md) and [table fetch_logs](table-fetch-logs.md); scoring output in [inline scoring](job-score-predictions.md).

## CLI status (`packages/odds-cli/odds_cli/commands/status.py`)

Rich-printed tables only; no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, odds and snapshot totals, quota and 24 h success rate from `OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota plus the last 10 `FetchLog` rows ([odds API quota](odds-api-quota.md)).
- `odds status events --days N --team T` (default 7 days): recent events with status and scores.
- `odds scheduler list-jobs` and `health` report the scheduler backend ([local scheduler](local-scheduler.md), [scheduler backends](scheduler-backends.md)); `list-jobs` catches `BackendUnavailableError` for Railway.

## MCP and traces

The agent reads state through MCP tools in `packages/odds-mcp/odds_mcp/server.py`: `get_scrape_status`, `get_scheduled_jobs`, `get_predictions`, `get_portfolio`, `settle_bets`, `save_match_brief`, `get_slate_briefs`, `schedule_next_wakeup`. Details: [MCP scrape and schedule tools](mcp-scrape-and-schedule-tools.md), [paper trading](paper-trading.md), [match briefs](match-briefs.md). Agent run output is a JSONL trace file, described in [agent run traces](agent-run-traces.md) and [agent run job](job-agent-run.md). The digest does not read paper trades, so bet performance is visible only through MCP and `odds paper`.

## Hazards

- Discord is the only alert channel and failure to deliver is only logged ([alerts module](alerts-module.md)).
- `get_scheduled_jobs` and `get_scrape_status` open the local APScheduler store (`build_scheduler`) whatever `SCHEDULER_BACKEND` says, so on AWS they do not show EventBridge rules.
- `HealthMetrics` never leaves the job log ([health monitor](health-monitor.md)).
- `odds status events` has no sport filter and `odds status show` mixes all fetch sources ([table fetch_logs](table-fetch-logs.md)).
- Sport naming differs by surface: compound job names (`daily-digest-epl`), sport keys (`soccer_epl`) and display names (`EPL`) each have their own translation table ([job registry](job-registry.md)).
- The component locations the doc lists in `#10-reference-where-each-piece-lives` are on the pages linked above, not repeated here.

## Related

- [Job deployment matrix](job-deployment-matrix.md), [CLI overview](cli-overview.md).
