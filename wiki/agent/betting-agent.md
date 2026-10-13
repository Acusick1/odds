---
type: explanation
title: The betting agent
description: What the per-sport agent is, its three channels, why prompts are authoritative, the role of the CLV model, the EPL matchday timeline, and its evaluation status.
sources:
  - odds_mcp.server.mcp
  - odds_mcp.server.find_retail_edges
  - odds_lambda.jobs.agent_run.CADENCE
  - odds_core.models.Event
  - odds_core.models.OddsSnapshot
  - odds_core.paper_trade_models.PaperTrade
  - odds_core.match_brief_models.MatchBrief
  - odds_core.agent_wakeup_models.AgentWakeup
  - packages/odds-mcp/agents/common.md
  - packages/odds-mcp/agents/epl/base.md
status: stable
---

# The betting agent

An LLM agent per sport (EPL and MLB) synthesises odds, sharp-market prices and unstructured match context and records bets as paper trades. Both agents are in interactive evaluation: the prompt calls itself a "first-draft workflow" (`packages/odds-mcp/agents/common.md`, #290). Nothing in code enforces that they are not autonomous `(inferred)`. The thesis is breadth and speed of information synthesis ([ADR-0005](../../docs/adr/0005-information-edge.md)). The thesis has no measured result: results exist only as paper trades, read through `odds_mcp.server.get_portfolio` and `odds paper`, and the [daily digest](../operations/daily-digest.md) does not read them ([paper trading](paper-trading.md)). The runtime is described in [Agent runtime and the agent-run job](agent-run-job.md).

## Architecture

The runtime is a `claude` subprocess; everything it touches goes through one of three channels:

| Channel | Used for |
|---|---|
| `odds` MCP server (`odds_mcp.server.mcp`) | fixtures, odds, retail edges, briefs, paper bets, portfolio |
| Web search | press conferences, injury news |
| Playwright MCP (not in `.mcp.json`, so supplied by the user's Claude configuration) | club sites, lineup pages, BBC and ESPN |

The MCP server reads and writes PostgreSQL: `events`, `odds_snapshots`, `paper_trades`, `match_briefs`, `predictions` and `agent_wakeups` (`odds_core.models.Event`, `odds_core.models.OddsSnapshot`, `odds_core.paper_trade_models.PaperTrade`, `odds_core.match_brief_models.MatchBrief`, `odds_core.agent_wakeup_models.AgentWakeup`; predictions are written by [scoring](../modelling/prediction-scoring.md)). See [odds-mcp server](mcp-server.md) for the tools.

## Prompts

The agent's behaviour is defined by prompt files in `packages/odds-mcp/agents/`: a common prompt (strategy and the wake-up workflow) plus one base prompt per sport (`epl` and `mlb` subdirectories). Those files are authoritative; the wiki summarises them in [Agent prompts](agent-prompts.md) and cannot verify them against code.

## Role of the CLV model

A supplementary signal: weak, unprofitable alone, and its strongest feature is the sharp-retail spread the agent sees directly via `odds_mcp.server.find_retail_edges`. The prompt treats models as supplementary sanity checks (`packages/odds-mcp/agents/common.md`) ([CLV model findings](../evaluation/clv-model-findings.md), [sharp benchmark](../modelling/sharp-benchmark.md), [ADR-0005](../../docs/adr/0005-information-edge.md)). Code and comments written before the pivot may still treat the CLV model as the edge ([project overview](../foundations/project-overview.md)).

## EPL matchday timeline

A Saturday 3pm kickoff:

| Window | Market activity | Agent action |
|---|---|---|
| Mon-Thu | Lines open, early sharp money | Light research, pre-market read |
| Fri evening | Press conferences, market firms up | Research, build briefs |
| Sat morning | Team news leaks, odds move | Update briefs, early bets if edge found |
| Sat ~14:00 | Confirmed lineups (about KO-60) | Deep research, final decisions |
| Sat 14:00-15:00 | Final adjustments, late money | Bets placed, if any |
| Sat evening | Matches complete | Settle bets, report P&L |

The table is `(inferred)` from the wake-up cadence in `odds_lambda.jobs.agent_run.CADENCE` (24 h, 12 h, 4 h, then 1 h inside 3 hours, see [Cadence](../scheduling/decision-engine.md#cadence)) and the prompt's KO-60 lineup note (`packages/odds-mcp/agents/epl/base.md`); it is not a schedule in code. The cadence is what puts a wake-up inside the lineup window. Lineups come from web search because every structured lineup API was rejected ([ADR-0006](../../docs/adr/0006-lineups-from-web-research.md), [rejected sources](agent-data-sources.md#rejected-sources)). The timeline is an EPL example. MLB has its own overnight suppression window (05:00-14:00 UTC) and sport file, and no MLB timeline is recorded ([overnight windows](../scheduling/decision-engine.md#overnight-windows)).

## Design principles

Recorded without a decision record, so the reasons are `(inferred)` (the sharp-price and breadth points match the Thesis in `packages/odds-mcp/agents/common.md`): sharp price is a strong reference, not gospel; breadth of information is the edge; MCP is the tool interface so the runtime can be swapped; tools return structured data; the database is shared state; local-first, never on AWS.

## Consequences of the pivot

- Effort moved to agent tooling: [odds-mcp server](mcp-server.md), [match briefs](match-briefs.md), [paper trading](paper-trading.md), [agent data sources](agent-data-sources.md).
- Do not cite the thesis as evidence that the agent is profitable.

## Related

- [Agent runtime and the agent-run job](agent-run-job.md)
- [Agent prompts](agent-prompts.md)
- [Agent data sources and rejected sources](agent-data-sources.md)
- [Match briefs](match-briefs.md) and [paper trading](paper-trading.md), what the agent writes
- [About the system and its pivots](../foundations/project-overview.md)
