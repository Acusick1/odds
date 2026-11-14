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
  - odds_web.app
  - experiments/scripts/show_agent_summaries.py
status: stable
---

# Reporting surfaces

"Reporting" is everything the system emits about itself (operations) or its outputs (paper trades, agent briefs). It has three modes.

| Mode | Producers | Output | Backing tables |
|---|---|---|---|
| Event-driven | jobs via [`odds_core.alerts.job_alert_context`](alerts.md#delivery) (`odds_core.alerts.send_job_failure`, once per outage), `odds_core.alerts.send_job_warning`, and the health monitor's alerts (quota included) | Discord | `alert_history` (outage and reminder state, heartbeats) |
| Poll-driven | check-health | Discord alerts | reads `fetch_logs`, `odds_snapshots`, `data_quality_logs`, `alert_history` |
| On-demand | `odds status`, `odds scheduler`, MCP tools, the [web UI](odds-web.md) | terminal tables, JSON or a page | reads everything |

Delivery, rate limiting and heartbeats are on [alerts](alerts.md) and [alerting-and-heartbeats](alerting-and-heartbeats.md); the check-health job is [health-monitor](health-monitor.md); `fetch_logs` (Odds API and Betfair rows, no source column) is described with [the ingestion service](../collection/odds-ingestion-service.md#fetch-logs).

## What each surface cannot show

- **Discord** is the only alert channel, and delivery failure is only logged (#457; [alerts](alerts.md#delivery)).
- **`odds status`** prints Rich tables only (events by integer id), with no machine-readable output, one quota row per Odds API key (balance over the key's plan size, and in `status quota` its burn projection), and its `show` view (event, price-row and snapshot counts) mixes fetch sources in its last-fetch line ([odds status](health-monitor.md#odds-status)).
- **`odds scheduler list-jobs` and `odds scheduler health`** report the scheduler backend; `odds scheduler health` prints the backend's `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check` result ([output](health-monitor.md#odds-status)). See [local scheduler](../scheduling/local-scheduler.md) and [scheduler backends](../scheduling/scheduler-backends.md).
- **MCP tools** let the agent read state: `odds_mcp.server.get_scrape_status`, `odds_mcp.server.get_scheduled_jobs`, `odds_mcp.server.get_portfolio`, `odds_mcp.server.settle_bets`, `odds_mcp.server.save_match_brief` (which stamps sharp prices read from `prices`, one per outcome, from `odds_core.bookmakers.SHARP_BOOKMAKERS`), `odds_mcp.server.get_slate_briefs`, `odds_mcp.server.schedule_next_wakeup` ([mcp-server](../agent/mcp-server.md), [match-briefs](../agent/match-briefs.md), [paper-trading](../agent/paper-trading.md)). `odds_mcp.server.get_scheduled_jobs` and `odds_mcp.server.get_scrape_status` read the local APScheduler store whatever `SCHEDULER_BACKEND` says, so on AWS they do not show EventBridge rules ([mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).
- **The web UI** shows the candidates' forward tests, each with its registry status beside its verdict, the stored [backtest](../evaluation/candidate-backtest.md#stored-runs) of each beside it (a candidate page's development and holdout runs, the portfolio's backtest estimate and a list of killed ideas), on the portfolio page and a page per candidate version (an earlier version that recorded decisions only included), with no next checkpoint for a version that takes no more bets, built on the same report data (`odds_analytics.candidate_report.candidate_report`, cached) as `odds candidates report` and `odds_mcp.server.get_portfolio`, with each registered version's [book replays](../evaluation/book-replays.md) for information on its page, decision-window and close coverage and every job, source and Odds API key on the data-health page, and on every page a one-row status strip, summarising collection per sport, the next candidate run and the quota and naming any job or source that is late, stale or unknown (a job whose next run passed unserved is late), judged as the health monitor judges them; a section whose read fails says whether the database was unavailable or odds-web itself failed. It is read-only, reachable over the tailnet, and alerts on nothing ([odds-web](odds-web.md)).
- **Agent results** surface only through MCP `odds_mcp.server.get_portfolio` (open trades with a bankroll, with their candidate and commit, and each candidate's checkpoint summary, or the registry's error), `odds paper`, `odds candidates report` (#384) and the web UI. Agent run output is a JSONL trace under `logs/agent_runs/`, viewed one session per `result` entry with `experiments/scripts/show_agent_summaries.py` ([agent run traces](../agent/agent-run-traces.md#viewer)).
- **Sport naming** differs per surface: compound job names (`fetch-oddsportal-epl`), sport keys (`soccer_epl`) and display names (`EPL`) each have their own translation (#384; [sport-and-market-keys](../collection/sport-and-market-keys.md#sport-keys)).

## Related

- [How failures are detected](alerting-and-heartbeats.md)
- [odds status](health-monitor.md#odds-status)
