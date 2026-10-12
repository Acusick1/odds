---
type: process
title: Matchday wake-up workflow
summary: "The steps of each agent wake-up (orient, settle, triage, pre-market read, research, brief, decide) and the EPL matchday timeline."
sources:
  - docs/BETTING_AGENT.md#matchday-workflow
  - docs/BETTING_AGENT.md#the-timeline-for-an-epl-match-saturday-3pm-ko-example
  - docs/BETTING_AGENT.md#wake-up-workflow
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE
  - packages/odds-mcp/odds_mcp/server.py::get_slate_briefs
  - packages/odds-mcp/odds_mcp/server.py::save_match_brief
  - packages/odds-mcp/odds_mcp/server.py::paper_bet
  - packages/odds-mcp/odds_mcp/server.py::settle_bets
verified_at: 7887039
---

# Matchday wake-up workflow

Every [agent](betting-agent.md) wake-up follows the same steps, defined in the prompt file `packages/odds-mcp/agents/common.md`. This page summarises them from [BETTING_AGENT.md](../docs/BETTING_AGENT.md#wake-up-workflow); the prompt file is authoritative and not verifiable against code.

## Steps

1. **Orient.** Named in the doc without further detail.
2. **Settle.** Settle finished bets (`settle_bets`, see [Paper trading](paper-trading.md)).
3. **Triage.** Read the latest brief per upcoming match (`get_slate_briefs`) and choose what to research ([Match briefs](match-briefs.md)).
4. **Pre-market read**, for new fixtures only.
5. **Research.** Web search and Playwright; depth scales with kickoff proximity ([Agent data sources](agent-data-sources.md)).
6. **Brief.** Append a brief per researched match with a decision of `bet`, `skip` or `watching` (`save_match_brief`).
7. **Decide.** Place bets when there is an edge (`paper_bet`); bets can be placed at any time if the edge is clear. Stakes follow [Conviction tiers](conviction-tiers.md).

The agent may ask for its next wake-up with `schedule_next_wakeup` ([scheduling tools](mcp-scrape-and-schedule-tools.md)); otherwise the job's cadence applies ([agent-run job](job-agent-run.md)).

## EPL timeline (Saturday 3pm kickoff)

| Window | Market activity | Agent action |
|---|---|---|
| Mon-Thu | Lines open, early sharp money | Light research, pre-market read |
| Fri evening | Press conferences, market firms up | Research, build briefs |
| Sat morning | Team news leaks, odds move | Update briefs, early bets if edge found |
| Sat ~14:00 | Confirmed lineups (about KO-60) | Deep research, final decisions |
| Sat 14:00-15:00 | Final adjustments, late money | Bets placed, if any |
| Sat evening | Matches complete | Settle bets, report P&L |

The wake-up cadence in `CADENCE` (24h, 12h, 4h, then 1h inside 3 hours) is what puts a wake-up inside the lineup window; the table above is the doc's example, not a schedule in code. Lineups come from web search, not an API ([Rejected data sources](decision-rejected-data-sources.md)).

## Hazards

- The timeline is an EPL example. MLB has its own overnight suppression window (05:00-14:00 UTC) and sport file, and the doc gives no MLB timeline ([Scheduling decision engine](scheduling-decision-engine.md)).
- Tool-to-step mapping above is the doc's description of the toolset; nothing in code ties a tool to a step.

## Related

- [odds-mcp server](mcp-server.md) — the tools
- [Fetch tier](fetch-tier.md) — proximity bands behind the cadence
