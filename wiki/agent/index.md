---
title: Betting agent
description: "The per-sport LLM agent: runtime and wake-up scheduling, its prompts and staking rules, the odds-mcp server and its tools, match briefs, paper trading, and the agent's data sources."
---

# Betting agent

An LLM agent per sport (EPL, MLB) runs as a `claude` subprocess over the odds-mcp tool server and records its decisions as match briefs and paper trades. Both agents run by hand and are paused; the pages here cover the code around the agent, while the prompt files it follows are the authority on its behaviour.

<!-- pages:start -->
### Reference

- [Agent prompts](agent-prompts.md) — The prompt files under packages/odds-mcp/agents/, the wake-up steps, edge types, conviction tiers and caps, and which of those rules code enforces.
- [Agent run traces](agent-run-traces.md) — Where each agent run's stream-json trace is written, how files are named, limited and pruned, which messages reach structlog, and what the summary viewer reads.
- [Agent runtime and the agent-run job](agent-run-job.md) — The claude subprocess, the agent-run job flow and cadence, agent_wakeups, odds agent run and the database role.
- [Match briefs](match-briefs.md) — The match_briefs table, the sharp-price stamp, and the save, get and slate MCP tools.
- [MCP odds, prediction and retail-edge tools](mcp-market-tools.md) — get_upcoming_fixtures, get_current_odds, get_odds_history, get_predictions and find_retail_edges, with inputs, anchor-relative price resolution, ranking and dispersion statistics, and the limits of their prices.
- [MCP sport context tools](mcp-sport-context-tools.md) — EPL get_team_context (form, fixtures, standings from ESPN fixtures) and MLB get_probable_pitchers (live read-through with cached fallback), and their data dependencies.
- [odds-mcp server](mcp-server.md) — How the FastMCP server is launched and mounted, the 17-tool catalogue and owning pages, sport sub-servers, shared vocabulary, and the scrape and scheduling tools (refresh_scrape, get_scrape_status, get_scheduled_jobs, schedule_next_wakeup).
- [Paper trading](paper-trading.md) — The paper_trades table, the paper-trading service, MCP and CLI entry points, portfolio maths and settlement.

### Explanation

- [Agent data sources and rejected sources](agent-data-sources.md) — Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why every evaluated source was rejected.
- [The betting agent](betting-agent.md) — What the per-sport agent is, its three channels, why prompts are authoritative, the role of the CLV model, the EPL matchday timeline, and its evaluation status.
<!-- pages:end -->
