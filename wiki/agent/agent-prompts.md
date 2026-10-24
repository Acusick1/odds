---
type: reference
title: Agent prompts
description: The prompt files under packages/odds-mcp/agents/, the wake-up steps, edge types, conviction tiers and caps, and which of those rules code enforces.
sources:
  - odds_lambda.jobs.agent_run.CADENCE
  - odds_lambda.paper_trading.place_trade
  - odds_core.paper_trade_models.PaperTrade
  - odds_mcp.server.get_upcoming_fixtures
  - odds_mcp.server.get_slate_briefs
  - odds_mcp.server.save_match_brief
  - odds_mcp.server.paper_bet
  - odds_mcp.server.settle_bets
  - odds_mcp.server.schedule_next_wakeup
  - odds_mcp.server.find_retail_edges
  - packages/odds-mcp/agents/common.md
  - packages/odds-mcp/agents/epl/base.md
status: stable
---

# Agent prompts

Every [agent](betting-agent.md) wake-up follows the same steps, defined in the prompt file `packages/odds-mcp/agents/common.md`; staking rules sit in the same file, and each sport has a base prompt (`packages/odds-mcp/agents/epl/base.md`, `packages/odds-mcp/agents/mlb/base.md`). This page summarises those files; the prompts are authoritative and not verifiable against code. The tool-to-step mapping below comes from the prompt; nothing in code ties a tool to a step.

## Steps

1. **Orient.** Check the date, weekday and UTC time (`date -u`), then load upcoming fixtures with `odds_mcp.server.get_upcoming_fixtures`.
2. **Settle.** Settle finished bets (`odds_mcp.server.settle_bets`, see [settlement](paper-trading.md#settlement)).
3. **Triage.** Read the latest brief per upcoming match (`odds_mcp.server.get_slate_briefs`) and choose what to research ([match briefs](match-briefs.md)). Briefs are append-only and do not store the market.
4. **Pre-market read**, for new fixtures only.
5. **Research.** Web search and Playwright; depth scales with kickoff proximity ([data sources](agent-data-sources.md)). Lineups come from BBC Sport at about KO-60, not an API ([ADR-0006](../../docs/adr/0006-lineups-from-web-research.md)).
6. **Brief.** Append a brief per researched match with a decision of `bet`, `skip` or `watching` (`odds_mcp.server.save_match_brief`).
7. **Decide.** Place bets when there is an edge (`odds_mcp.server.paper_bet`); bets can be placed at any time if the edge is clear. Stakes follow the conviction tiers below.

The agent may ask for its next wake-up with `odds_mcp.server.schedule_next_wakeup` ([scheduling tools](mcp-server.md#scrape-and-scheduling-tools)); otherwise the job's cadence applies. The cadence (`odds_lambda.jobs.agent_run.CADENCE`, [cadence table](../scheduling/decision-engine.md#cadence)) is 1 h inside 3 h to kickoff and 4, 12 and 24 h further out, and a requested wake-up overrides it ([wake-up requests](agent-run-job.md#wake-up-requests)).

## Edge types

The agent does not bet on qualitative team assessment. Every bet needs one of four specific bases:

1. an information gap;
2. retail or cross-venue dispersion among bettable books (`odds_mcp.server.find_retail_edges`, whose rows give each retail price as both an American quote and the source decimal; see [retail edges](mcp-market-tools.md#retail-edges));
3. a structural bias (bookmaker liability management, accumulator distortion, public money on popular teams);
4. fundamental value.

The prompt's Thesis frames structural biases as valid edge types alongside information gaps (`packages/odds-mcp/agents/common.md`).

## Conviction tiers

| Tier | Stake (share of bankroll) |
|---|---|
| 0 (default, no identifiable edge) | 0% |
| 1 | 1% |
| 2 | 2% |
| 3 | 3% |

The criteria per tier are in the prompt file. The prompt also caps exposure at 10% of bankroll across all open bets on one matchday and says never to exceed 3% on a single bet.

## What code enforces

`odds_lambda.paper_trading.place_trade` checks only that the decimal price exceeds 1.0, the stake is positive, does not exceed the bankroll, that the market is `h2h` or `1x2` and that the selection is `home`, `draw` or `away` (`draw` on `h2h` only for a three-way sport such as EPL, not MLB). A candidate trade is also checked against the registry, including that the event's sport is the candidate's and that a three-way sport uses `1x2`. Tier sizes, the 3% single-bet cap, the 10% matchday cap, the four edge bases and the requirement to state a basis are prompt-level: `odds_core.paper_trade_models.PaperTrade.reasoning` and `confidence` are free-form optional fields, so the tool accepts a stake above 3% of bankroll, a matchday over the 10% cap, or a bet with no stated edge. The price is the decimal price as quoted (`price_decimal`), and the optional candidate is checked against the registry ([paper trading](paper-trading.md#service-functions)). The `bankroll` that `odds_mcp.server.paper_bet` uses for sizing is computed from the portfolio summary unless the agent passes its own ([paper trading](paper-trading.md)).

## Related

- [The betting agent](betting-agent.md)
- [odds-mcp server](mcp-server.md), the tools
- [Fetch tier](../storage/fetch-tier.md), the proximity bands behind the cadence
