---
type: explanation
title: Goal and strategy
description: What the project is for, what counts as a candidate, how a candidate is scored, promoted or killed, and which parts of the workflow exist today.
sources:
  - odds_core.paper_trade_models.PaperTrade
  - odds_analytics.retail_ev
status: stable
---

# Goal and strategy

## Goal

The goal is profit from betting at UK retail books on EPL and MLB. No approach is privileged: agents, models, mechanical rules or anything else are tried if they might get there, and several are tried at once.

## Candidates

A candidate is any rule that decides bets: a mechanical rule on prices, a model, or an LLM agent.

- **Registry.** A registered candidate is one YAML file in `candidates/` ([Candidate registry](../evaluation/candidate-registry.md)), holding its definition and its status: active, promoted or killed. Status changes by pull request, and no database table copies it.
- **Version.** Each candidate has a whole-number version, bumped whenever its rule, parameters or model changes. For an agent, a change to `packages/odds-mcp/agents/common.md`, its sport's `base.md` or the model passed by `odds_lambda.jobs.agent_run._run_claude_agent` is a new version. A change to `packages/odds-mcp/agents/<sport>/observations.md` is not: that file is the agent's own running playbook, and it is gitignored, so a version and git SHA do not pin it `(inferred)`.
- **Tracking.** An idea starts as a GitHub issue labelled `candidate`. Once registered, its file's status is its stage.

No candidate is running and no mechanical rule exists; mechanical rules are the current focus. The repo also holds:

- **The EPL and MLB betting agents** synthesise prices, exchange data and match news ([The betting agent](../agent/betting-agent.md)). Neither is registered or being run.
- **The EPL retail-EV model** predicts a retail bet's EV against the close ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)). Its development results are level with the exchange-price baseline ([CLV model findings](../evaluation/clv-model-findings.md)), and it places no bets.

The suspended bet365 CLV model is not a candidate. Its predictions are an input the agents may read ([prediction scoring](../modelling/prediction-scoring.md)).

## How a candidate is judged

- **Development data generates and kills ideas, and never promotes one.** The EPL 2024–26 range has been used for evaluation several times ([EPL CLV re-baseline](../evaluation/exp-epl-clv-rebaseline.md)). Choosing the best of many ideas on data already seen finds edges that are not there `(inferred)`.
- **A candidate is judged on a live forward test.** It paper-trades from a start date at a fixed version. Any number of candidates run in parallel, and none waits on another.
- **The score is EV at the price taken against the Betfair Exchange close:** `p_close × decimal_odds − 1`, with the close as defined for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target). Betfair Exchange is the close because its closing prices exist from 2024-25 on, where FDUK Pinnacle closing prices end 2026-01-08 ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md), "Considered options"). That close is defined for EPL only.
- **EV beats realised P&L as the score because of variance** `(inferred)`. A bet's P&L is a win or a loss of the whole stake, while its EV against the close varies only with how far the price moves before kick-off, so EV shows an edge in far fewer bets.
- **P&L is reported beside EV.** Positive EV with P&L below it by more than variance explains means the close is the wrong benchmark for that market `(inferred)`.
- **Promote or kill at fixed checkpoints.** A candidate is scored on its first 100, 200, … settled bets, never on a running total, so the verdict does not depend on when it is read. The score is mean EV per unit staked, with an event-clustered bootstrap CI.
  - **Promote** when the 99% CI's lower bound is above zero. The stricter interval allows roughly for repeated looks and for many candidates running at once `(inferred)`.
  - **Kill** when the 95% CI's upper bound is at or below zero, or after 500 bets without promotion.
- **Real money needs stake capacity too.** UK retail books restrict accounts that win, so a promoted candidate is also judged on how much it can stake `(inferred)`. The gate has no rule until a candidate is promoted.

## What exists today

Most of the workflow's tooling does not exist (#608). The [candidate registry](../evaluation/candidate-registry.md) does, with no candidate in it. Missing:

- `odds_core.paper_trade_models.PaperTrade` has no field for the candidate or its version, and stores no close (#604). MLB has no close defined.
- Nothing runs a mechanical candidate; only the agent can place paper trades, through `odds_mcp.server.paper_bet` (#609).

## Related

- [About the system and its pivots](project-overview.md)
- [Paper trading](../agent/paper-trading.md)
- [Retail-EV evaluation](../evaluation/retail-ev-evaluation.md)
