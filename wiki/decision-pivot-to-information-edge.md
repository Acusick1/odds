---
type: decision
title: Pivot to information edge
summary: "Why the agent's breadth of information synthesis, not the CLV model, is treated as the primary edge."
sources:
  - docs/BETTING_AGENT.md
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - docs/MODELING.md#what-the-model-predicts
  - docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
verified_at: 7887039
---

# Pivot to information edge

**Status:** in effect. No ADR exists (`docs/adr/` is absent), so the decision is reconstructed from the docs below. The audit dates the pivot to 2026-04-12 ([audit T1](../docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots)); no document records the discussion.

## Decision

The hypothesised edge is **breadth and speed of information synthesis** by an LLM agent that watches bookmaker odds, exchange prices, lineups, press conferences and fixture context together ([AGENTS.md](../AGENTS.md#strategic-goal), [BETTING_AGENT.md](../docs/BETTING_AGENT.md)). The XGBoost CLV model remains a supplementary signal: it keeps being scored inline and exposed through `get_predictions`, and the agent treats it as a sanity check, never a trading signal ([Betting agent](betting-agent.md), [Inline CLV scoring](job-score-predictions.md)).

Structural biases (liability shading, accumulator distortion, public money) and information gaps both count as edge types, and bets execute on whichever venue prices best against estimated fair value ([Conviction tiers](conviction-tiers.md)).

## Rationale

Recorded evidence on the model side ([MODELING.md](../docs/MODELING.md#what-the-model-predicts)):

- The signal is weak but real: CV R² about 0.058 on EPL and about 0.036 on NBA.
- It is not strong enough to bet alone: flat betting on NBA at bet365's roughly 4.5% vig loses at every threshold that yields a meaningful bet count, even with line shopping.
- Its strongest feature, the sharp-retail spread, is directly visible to the agent through `find_retail_edges` ([Retail-sharp divergence](retail-sharp-divergence.md)).

That the model's weakness caused the pivot is `(inferred)`: the docs give the thesis and the evidence side by side but do not link them. The design principle "breadth of information is the edge" is itself marked `(inferred)` in the doc. Details: [CLV model findings](clv-model-findings.md).

## Consequences

- Effort moved to agent tooling: [odds-mcp server](mcp-server.md), [Match briefs](match-briefs.md), [Paper trading](paper-trading.md), [Agent data sources](agent-data-sources.md).
- Docs and code from the model-edge era disagree with the current framing in places ([Project pivots](project-pivots.md)).
- The thesis is untested: both agents are in interactive evaluation and results live only in paper trades ([Paper trading](paper-trading.md)).

## Related

- [Project overview](project-overview.md)
- [Rejected data sources](decision-rejected-data-sources.md) — what was evaluated for the agent
