---
type: decision
title: Pivot to information edge
description: Why the agent's breadth of information synthesis, not the CLV model, is treated as the primary edge, the evidence recorded for it, and what it changed.
sources:
  - docs/BETTING_AGENT.md
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - docs/MODELING.md#what-the-model-predicts
  - docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
  - AGENTS.md
verified: "7887039"
status: stable
---

# Pivot to information edge

**Status:** in effect. No ADR exists (`docs/adr/` is absent), so the decision is reconstructed from the docs cited above. The audit dates the pivot to 2026-04-12 (`docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots`); no document records the discussion.

## Decision

The hypothesised edge is **breadth and speed of information synthesis** by an LLM agent that watches bookmaker odds, exchange prices, lineups, press conferences and fixture context together (`AGENTS.md`, `docs/BETTING_AGENT.md`). The XGBoost CLV model remains a supplementary signal: it keeps being scored inline (`packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events`) and exposed through `get_predictions`, and the agent treats it as a sanity check, never a trading signal ([betting agent](betting-agent.md), [publishing and scoring](../modelling/model-publishing-and-scoring.md)).

Structural biases (liability shading, accumulator distortion, public money) and information gaps both count as edge types, and bets execute on whichever venue prices best against estimated fair value ([edge types](agent-wakeup-workflow.md#edge-types)).

## Rationale

Recorded evidence on the model side (`docs/MODELING.md#what-the-model-predicts`):

- The signal is weak but real: CV R² about 0.058 on EPL and about 0.036 on NBA.
- It is not strong enough to bet alone: flat betting on NBA at bet365's roughly 4.5% vig loses at every threshold that yields a meaningful bet count, and line shopping does not close the gap ([betting simulation](../evaluation/clv-model-findings.md#betting-simulation-exp-7-7b)).
- Its strongest feature, the sharp-retail spread, is directly visible to the agent through `find_retail_edges` ([sharp benchmark](../modelling/sharp-benchmark.md), [retail edges tool](mcp-market-tools.md#find_retail_edges)).

That the model's weakness caused the pivot is `(inferred)`: the docs give the thesis and the evidence side by side but do not link them. The design principle "breadth of information is the edge" is itself marked `(inferred)` in the doc. Details: [CLV model findings](../evaluation/clv-model-findings.md).

## Consequences

- Effort moved to agent tooling: [odds-mcp server](mcp-server.md), [match briefs and paper trades](match-briefs-and-paper-trades.md), [agent data sources](agent-data-sources.md).
- Docs and code from the model-edge era disagree with the current framing in places; NBA-era defaults still sit in live code ([project overview](../foundations/project-overview.md)).
- The thesis is untested: both agents are in interactive evaluation, and results exist only in paper trades. They surface through MCP `get_portfolio` and `odds paper`; the [daily digest](../operations/reporting-and-daily-digest.md) does not read them ([paper trades](match-briefs-and-paper-trades.md#paper-trades)).

## Hazards

- The thesis has no measured result yet; do not cite it as evidence that the agent is profitable.
- Pages and docs written before the pivot may still describe the CLV model as the edge; when they disagree, the agent-side framing here and in `AGENTS.md` is current.
- Reversing the decision would mean showing the model beats the vig on its own, which the recorded simulation did not.

## Related

- [Agent data sources and rejected sources](agent-data-sources.md) — what was evaluated for the agent
- [Project overview](../foundations/project-overview.md)
