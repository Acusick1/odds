---
status: proposed
---

# Treat the agent's information synthesis as the primary edge

## Context

The CLV model's signal is weak but real: cross-validated R² is about 0.058 on EPL (#361) and about 0.036 on NBA. Flat betting on NBA at bet365's roughly 4.5% vig loses at every threshold that yields a meaningful bet count, and line shopping does not close the gap (`experiments/results/exp7_backtest_sim/FINDINGS.md`, #170; `experiments/results/exp7b_line_shopping/FINDINGS.md`, #171; learning curve in `experiments/results/exp6_learning_curve/FINDINGS.md`). The model's strongest feature, the sharp-retail spread, is visible to an agent directly through `odds_mcp.server.find_retail_edges`.

## Decision

The hypothesised edge is breadth and speed of information synthesis by an LLM agent that watches odds, exchange prices, lineups, press conferences and fixtures together (2026-04-12, #290 for issue #287; the Thesis in `packages/odds-mcp/agents/common.md`; `AGENTS.md`). The CLV model stays a supplementary signal: it is scored inline and exposed through `get_predictions`, and the agent treats it as a sanity check, never a trading signal. Structural biases and information gaps both count as edge types.

## Consequences

- That the model's weakness caused the pivot is `(inferred)` in the prior wiki text (f2a0862): the sources state the thesis and the evidence side by side but do not link them. The discussion behind #290 is not recorded.
- Effort moves to agent tooling: the MCP server, match briefs, paper trades and data sources.
- The thesis has no measured result. Results exist only as paper trades, read through `get_portfolio` and `odds paper`. 
- Code written before the pivot may still treat the model as the edge, and NBA-era defaults remain in live code (#465).
- Reversing the decision would need the model to beat the vig on its own, which the recorded simulation did not show.

## Evidence

- PRs and issues: #170, #171, #287, #290, #361, #465.
- Results: `experiments/results/exp6_learning_curve/FINDINGS.md`, `experiments/results/exp7_backtest_sim/FINDINGS.md`, `experiments/results/exp7b_line_shopping/FINDINGS.md`.
- Prior text: `wiki/agent/decision-information-edge.md` at f2a0862; `AGENTS.md` at `9b3ae09^`; `packages/odds-mcp/agents/common.md`.
- Wiki: [The betting agent](../../wiki/agent/betting-agent.md), [About the system and its pivots](../../wiki/foundations/project-overview.md), [CLV model findings](../../wiki/evaluation/clv-model-findings.md), [Betting simulation](../../wiki/evaluation/exp-betting-simulation.md), [Agent data sources](../../wiki/agent/agent-data-sources.md).
