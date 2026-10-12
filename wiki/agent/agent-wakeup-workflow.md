---
type: process
title: Wake-up workflow and staking rules
description: The steps of each wake-up (orient, settle, triage, read, research, brief, decide), the EPL matchday timeline, the 0-3% conviction tiers and four edge types, and what code does not enforce.
sources:
  - docs/BETTING_AGENT.md#matchday-workflow
  - docs/BETTING_AGENT.md#the-timeline-for-an-epl-match-saturday-3pm-ko-example
  - docs/BETTING_AGENT.md#wake-up-workflow
  - docs/BETTING_AGENT.md#conviction-tiers
  - docs/BETTING_AGENT.md#design-principles
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-mcp/odds_mcp/server.py::get_slate_briefs
  - packages/odds-mcp/odds_mcp/server.py::save_match_brief
  - packages/odds-mcp/odds_mcp/server.py::paper_bet
  - packages/odds-mcp/odds_mcp/server.py::settle_bets
verified: "7887039"
status: stable
---

# Wake-up workflow and staking rules

Every [agent](betting-agent.md) wake-up follows the same steps, defined in the prompt file `packages/odds-mcp/agents/common.md`; staking rules sit in the same file. This page summarises them from `docs/BETTING_AGENT.md#wake-up-workflow`. The prompt file is authoritative and not verifiable against code.

## Steps

1. **Orient.** Named in the doc without further detail.
2. **Settle.** Settle finished bets (`settle_bets`, see [paper trades](match-briefs-and-paper-trades.md#settlement)).
3. **Triage.** Read the latest brief per upcoming match (`get_slate_briefs`) and choose what to research ([match briefs](match-briefs-and-paper-trades.md#match-briefs)). Briefs are append-only and do not store the market.
4. **Pre-market read**, for new fixtures only.
5. **Research.** Web search and Playwright; depth scales with kickoff proximity ([data sources](agent-data-sources.md)). Lineups come from BBC Sport at about KO-60, not an API.
6. **Brief.** Append a brief per researched match with a decision of `bet`, `skip` or `watching` (`save_match_brief`).
7. **Decide.** Place bets when there is an edge (`paper_bet`); bets can be placed at any time if the edge is clear. Stakes follow the conviction tiers below.

The agent may ask for its next wake-up with `schedule_next_wakeup` ([scheduling tools](mcp-server.md#scrape-and-scheduling-tools)); otherwise the job's cadence applies. The cadence is 1 h inside 3 h to kickoff and 4, 12 and 24 h further out, and a requested wake-up overrides it ([agent-run job](betting-agent.md#cadence)).

## EPL timeline (Saturday 3pm kickoff)

| Window | Market activity | Agent action |
|---|---|---|
| Mon-Thu | Lines open, early sharp money | Light research, pre-market read |
| Fri evening | Press conferences, market firms up | Research, build briefs |
| Sat morning | Team news leaks, odds move | Update briefs, early bets if edge found |
| Sat ~14:00 | Confirmed lineups (about KO-60) | Deep research, final decisions |
| Sat 14:00-15:00 | Final adjustments, late money | Bets placed, if any |
| Sat evening | Matches complete | Settle bets, report P&L |

The wake-up cadence in `CADENCE` (24h, 12h, 4h, then 1h inside 3 hours) is what puts a wake-up inside the lineup window; the table is the doc's example, not a schedule in code. Lineups come from web search because every structured lineup API was rejected ([rejected sources](agent-data-sources.md#rejected-sources)).

## Edge types

The agent does not bet on qualitative team assessment. Every bet needs one of four specific bases:

1. an information gap;
2. retail or cross-venue dispersion ([`find_retail_edges`](mcp-market-tools.md#find_retail_edges));
3. a structural bias (bookmaker liability management, accumulator distortion, public money on popular teams);
4. fundamental value.

The strategic goal frames structural biases as valid edge types alongside information gaps ([pivot to information edge](decision-information-edge.md)).

## Conviction tiers

| Tier | Stake (share of bankroll) |
|---|---|
| 0 (default, no identifiable edge) | 0% |
| 1 | 1% |
| 2 | 2% |
| 3 | 3% |

Higher conviction maps to a higher tier; the criteria per tier are in the prompt file.

## What code enforces

`place_trade` in `packages/odds-lambda/odds_lambda/paper_trading.py` checks only that the stake is positive, does not exceed the bankroll, and that the selection is `home`, `draw` or `away`. Settlement is h2h-only. Tier sizes, the four edge bases and the requirement to state a basis are prompt-level: `PaperTrade.reasoning` and `confidence` are free-form optional fields. The `bankroll` that `paper_bet` uses for sizing defaults to the portfolio's current bankroll ([paper trades](match-briefs-and-paper-trades.md#paper-trades)).

## Hazards

- A stake above 3% of bankroll, or a bet with no stated edge, is accepted by the tool; only the prompt prevents it.
- `bankroll` for the 1-3% sizing is whatever `paper_bet` computes from the portfolio summary (initial 1000 plus settled P&L minus open stakes), unless the agent passes its own value.
- The timeline is an EPL example. MLB has its own overnight suppression window (05:00-14:00 UTC) and sport file, and the doc gives no MLB timeline ([self-scheduling](../scheduling/self-scheduling.md)).
- The tool-to-step mapping above is the doc's description of the toolset; nothing in code ties a tool to a step.

## Related

- [odds-mcp server](mcp-server.md) — the tools
- [Fetch tier](../storage/fetch-tier.md) — proximity bands behind the cadence
- [Betting agent](betting-agent.md)
