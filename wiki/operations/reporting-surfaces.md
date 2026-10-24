---
type: explanation
title: Reporting surfaces
description: What the system reports about itself and its outputs, through which channel, and what each surface cannot show.
sources:
  - odds_cli.commands.status
  - odds_cli.commands.scheduler.health
  - odds_mcp.server.get_scheduled_jobs
  - odds_mcp.server.get_scrape_status
  - odds_mcp.server.get_portfolio
  - experiments/scripts/show_agent_summaries.py
status: stable
---

# Reporting surfaces

"Reporting" is everything the system emits about itself (operations) or its outputs (paper trades, agent briefs). It has three modes.

| Mode | Producers | Output | Backing tables |
|---|---|---|---|
| Event-driven | jobs via [`odds_core.alerts.job_alert_context`](alerts.md#delivery) (`odds_core.alerts.send_job_failure`, once per outage), `odds_core.alerts.send_job_warning`, and the health monitor's alerts (quota included) | Discord | `alert_history` (outage and reminder state, heartbeats) |
| Poll-driven | check-health | Discord alerts | reads `fetch_logs`, `odds_snapshots`, `data_quality_logs`, `alert_history` |
| On-demand | `odds status`, `odds scheduler`, MCP tools | terminal tables or JSON | reads everything |

Delivery, rate limiting and heartbeats are on [alerts](alerts.md) and [alerting-and-heartbeats](alerting-and-heartbeats.md); the check-health job is [health-monitor](health-monitor.md); `fetch_logs` (Odds API and Betfair rows, no source column) is described with [the ingestion service](../collection/odds-ingestion-service.md#fetch-logs).

## What each surface cannot show

- **Discord** is the only alert channel, and delivery failure is only logged (#457; [alerts](alerts.md#delivery)).
- **`odds status`** prints Rich tables only, with no machine-readable output, and its `show` view mixes fetch sources in its last-fetch line ([odds status](health-monitor.md#odds-status)).
- **`odds scheduler list-jobs` and `odds scheduler health`** report the scheduler backend; `odds scheduler health` prints the backend's `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check` result ([output](health-monitor.md#odds-status)). `list-jobs` catches `odds_lambda.scheduling.exceptions.BackendUnavailableError` for Railway ([local-scheduler](../scheduling/local-scheduler.md), [scheduler-backends](../scheduling/scheduler-backends.md)).
- **MCP tools** let the agent read state: `odds_mcp.server.get_scrape_status`, `odds_mcp.server.get_scheduled_jobs`, `odds_mcp.server.get_portfolio`, `odds_mcp.server.settle_bets`, `odds_mcp.server.save_match_brief`, `odds_mcp.server.get_slate_briefs`, `odds_mcp.server.schedule_next_wakeup` ([mcp-server](../agent/mcp-server.md), [match-briefs](../agent/match-briefs.md), [paper-trading](../agent/paper-trading.md)). `odds_mcp.server.get_scheduled_jobs` and `odds_mcp.server.get_scrape_status` read the local APScheduler store whatever `SCHEDULER_BACKEND` says, so on AWS they do not show EventBridge rules (#449; [mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).
- **Agent results** surface only through MCP `odds_mcp.server.get_portfolio` (open trades with their candidate and commit, and each candidate's checkpoint summary, or the registry's error), `odds paper` and `odds candidates report` (#384). Agent run output is a JSONL trace under `logs/agent_runs/`, viewed with `experiments/scripts/show_agent_summaries.py` ([agent-run-job](../agent/agent-run-traces.md)).
- **Sport naming** differs per surface: compound job names (`fetch-oddsportal-epl`), sport keys (`soccer_epl`) and display names (`EPL`) each have their own translation (#384; [sport-and-market-keys](../collection/sport-and-market-keys.md#sport-keys)).

## Related

- [How failures are detected](alerting-and-heartbeats.md)
- [odds status](health-monitor.md#odds-status)
