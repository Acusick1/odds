---
type: decision
title: Pivot to information edge
description: Why the agent's breadth of information synthesis, not the CLV model, is treated as the primary edge, the evidence recorded for it, and what it changed.
sources:
  - odds_lambda.jobs.score_predictions.score_events
  - packages/odds-mcp/agents/common.md
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/results/exp7_backtest_sim/FINDINGS.md
  - experiments/results/exp7b_line_shopping/FINDINGS.md
status: stable
---

# Pivot to information edge

**Status:** in effect. No ADR records it. The pivot dates to #290 (2026-04-12, "Write agent system prompt for information-edge workflow", issue #287); no document records the discussion.

## Decision

The hypothesised edge is **breadth and speed of information synthesis** by an LLM agent that watches bookmaker odds, exchange prices, lineups, press conferences and fixture context together (`AGENTS.md`; the Thesis in `packages/odds-mcp/agents/common.md`, #290). The XGBoost CLV model remains a supplementary signal: it keeps being scored inline (`odds_lambda.jobs.score_predictions.score_events`) and exposed through `odds_mcp.server.get_predictions`, and the agent treats it as a sanity check, never a trading signal (`packages/odds-mcp/agents/common.md`, [betting agent](betting-agent.md), [publishing and scoring](../modelling/model-publishing-and-scoring.md)).

Structural biases (liability shading, accumulator distortion, public money) and information gaps both count as edge types, and bets execute on whichever venue prices best against estimated fair value ([edge types](agent-wakeup-workflow.md#edge-types)).

## Rationale

Recorded evidence on the model side ([CLV model findings](../evaluation/clv-model-findings.md); `experiments/results/exp6_learning_curve/FINDINGS.md`, `experiments/results/exp7_backtest_sim/FINDINGS.md`, `experiments/results/exp7b_line_shopping/FINDINGS.md`; EPL figure #361):

- The signal is weak but real: CV R² about 0.058 on EPL and about 0.036 on NBA.
- It is not strong enough to bet alone: flat betting on NBA at bet365's roughly 4.5% vig loses at every threshold that yields a meaningful bet count, and line shopping does not close the gap ([betting simulation](../evaluation/clv-model-findings.md#betting-simulation-exp-7-7b)).
- Its strongest feature, the sharp-retail spread, is directly visible to the agent through `odds_mcp.server.find_retail_edges` ([sharp benchmark](../modelling/sharp-benchmark.md), [retail edges tool](mcp-market-tools.md#retail-edges)).

That the model's weakness caused the pivot is `(inferred)`: the docs give the thesis and the evidence side by side but do not link them. The design principle "breadth of information is the edge" is itself `(inferred)` ([betting agent](betting-agent.md#design-principles)).

## Consequences

- Effort moved to agent tooling: [odds-mcp server](mcp-server.md), [match briefs and paper trades](match-briefs-and-paper-trades.md), [agent data sources](agent-data-sources.md).
- Code from the model-edge era disagrees with the current framing in places; NBA-era defaults still sit in live code ([project overview](../foundations/project-overview.md)).
- The thesis is untested: both agents are in interactive evaluation, and results exist only in paper trades. They surface through MCP `odds_mcp.server.get_portfolio` and `odds paper`; the [daily digest](../operations/reporting-and-daily-digest.md) does not read them ([paper trades](match-briefs-and-paper-trades.md#paper-trades)).

## Hazards

- The thesis has no measured result yet; do not cite it as evidence that the agent is profitable.
- Code and comments written before the pivot may still treat the CLV model as the edge; when they disagree, the framing here and in `AGENTS.md` is current.
- Reversing the decision would mean showing the model beats the vig on its own, which the recorded simulation did not.

## Related

- [Agent data sources and rejected sources](agent-data-sources.md) — what was evaluated for the agent
- [Project overview](../foundations/project-overview.md)
