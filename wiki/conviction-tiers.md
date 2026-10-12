---
type: concept
title: Conviction tiers and edge types
summary: "Stakes of 0-3% of bankroll by conviction and the four edge bases every bet must name, plus what code does and does not enforce."
sources:
  - docs/BETTING_AGENT.md#conviction-tiers
  - docs/BETTING_AGENT.md#design-principles
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
verified_at: 7887039
---

# Conviction tiers and edge types

Staking rules for the [betting agent](betting-agent.md), stated in [BETTING_AGENT.md](../docs/BETTING_AGENT.md#conviction-tiers) and defined in full in the prompt file `packages/odds-mcp/agents/common.md` (the criteria per tier live there).

## Edge types

The agent does not bet on qualitative team assessment. Every bet needs one of four specific bases:

1. an information gap;
2. retail or cross-venue dispersion ([find_retail_edges](mcp-find-retail-edges.md));
3. a structural bias (bookmaker liability management, accumulator distortion, public money on popular teams);
4. fundamental value.

The strategic goal frames these as valid edge types alongside information gaps ([AGENTS.md](../AGENTS.md#strategic-goal), [Pivot to information edge](decision-pivot-to-information-edge.md)).

## Stake tiers

| Tier | Stake (share of bankroll) |
|---|---|
| 0 (default, no identifiable edge) | 0% |
| 1 | 1% |
| 2 | 2% |
| 3 | 3% |

Higher conviction maps to a higher tier; the criteria per tier are in the prompt file.

## What code enforces

`place_trade` in `packages/odds-lambda/odds_lambda/paper_trading.py` checks only that the stake is positive, does not exceed the bankroll, and that the selection is `home`, `draw` or `away`. Tier sizes, the four edge bases and the requirement to state a basis are prompt-level: `PaperTrade.reasoning` and `confidence` are free-form optional fields ([Paper trading](paper-trading.md)).

## Hazards

- A stake above 3% of bankroll, or a bet with no stated edge, is accepted by the tool; only the prompt prevents it.
- `bankroll` for the 1-3% sizing is whatever `paper_bet` computes from the portfolio summary (initial 1000 plus settled P&L minus open stakes), unless the agent passes its own value.

## Related

- [Matchday wake-up](matchday-wakeup.md) — where the bet decision happens
- [Betting agent](betting-agent.md)
