---
title: Betting agent
description: "The per-sport LLM agent: runtime and wake-up scheduling, the wake-up workflow and staking rules, the odds-mcp server and its tools, match briefs and paper trades, and the agent's data sources."
---

# Betting agent

An LLM agent per sport (EPL, MLB) runs as a `claude` subprocess over the odds-mcp tool server and records its decisions as match briefs and paper trades. Both agents are in interactive evaluation; the pages here cover the code around the agent, while the prompt files it follows are the authority on its behaviour.

<!-- pages:start -->
### Concepts

- [Agent data sources and rejected sources](agent-data-sources.md) — Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why each evaluated source was rejected.

### Processes

- [Wake-up workflow and staking rules](agent-wakeup-workflow.md) — The steps of each wake-up (orient, settle, triage, read, research, brief, decide), the EPL matchday timeline, the 0-3% conviction tiers and four edge types, and what code does not enforce.

### Components

- [Betting agent runtime and the agent-run job](betting-agent.md) — How each agent wake-up runs as a claude subprocess over odds-mcp, how the opt-in, local-only agent-run job schedules wake-ups and honours agent_wakeups, the agent's database role, and the JSONL run traces.
- [Match briefs and paper trades](match-briefs-and-paper-trades.md) — The agent's two persistent records: append-only match_briefs with their sharp-price stamp and the save/get/slate tools, and paper_trades with place, settle and portfolio logic via MCP and odds paper.
- [MCP odds, prediction and retail-edge tools](mcp-market-tools.md) — get_upcoming_fixtures, get_current_odds, get_odds_history, get_predictions and find_retail_edges, with inputs, anchor-relative price resolution, ranking and dispersion statistics, and why their prices are not co-temporal or devigged.
- [MCP sport context tools](mcp-sport-context-tools.md) — EPL get_team_context (form, fixtures, standings from ESPN fixtures) and MLB get_probable_pitchers (live read-through with cached fallback), and their data dependencies.
- [odds-mcp server and scheduling tools](mcp-server.md) — How the FastMCP server is launched and mounted, the 17-tool catalogue and owning pages, shared vocabulary, and the scrape and scheduling tools (refresh_scrape, get_scrape_status, get_scheduled_jobs, schedule_next_wakeup) and what they do not guarantee.

### Decisions

- [Pivot to information edge](decision-information-edge.md) — Why the agent's breadth of information synthesis, not the CLV model, is treated as the primary edge, the evidence recorded for it, and what it changed.
<!-- pages:end -->
