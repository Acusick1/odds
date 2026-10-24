---
type: explanation
title: Goal and strategy
description: What the project is for, what counts as a candidate, how a candidate is scored, promoted or killed, and what scoring exists today.
sources:
  - odds_core.paper_trade_models.PaperTrade
  - odds_analytics.retail_ev
status: stable
---

# Goal and strategy

## Goal

The goal is profit from betting at UK retail books on EPL and MLB. No approach is privileged: agents, models, mechanical rules or anything else are tried if they might get there, and several are tried at once.

## Candidates

A candidate is any rule that decides bets. Each one has a version, and a change of rule is a new version. The candidates today:

- **The EPL and MLB betting agents.** They synthesise prices, exchange data and match news, and are run by hand ([The betting agent](../agent/betting-agent.md)).
- **The EPL retail-EV model.** It predicts a retail bet's EV against the close ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)). Its development results are level with the exchange-price baseline ([CLV model findings](../evaluation/clv-model-findings.md)), and it places no bets.

The suspended bet365 CLV model is not a candidate. Its predictions are an input the agents may read ([prediction scoring](../modelling/prediction-scoring.md)).

## How a candidate is judged

- **Development data generates ideas and never promotes.** The EPL 2024–26 range has been used for evaluation several times ([EPL CLV re-baseline](../evaluation/exp-epl-clv-rebaseline.md)). Choosing the best of many ideas on data already seen finds edges that are not there `(inferred)`.
- **A candidate is judged on a live forward test.** It paper-trades from a start date at a fixed version. Any number of candidates run in parallel, and none waits on another.
- **The score is EV at the price taken against the Betfair Exchange close:** `p_close × decimal_odds − 1`, with the close as defined for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target). Betfair Exchange is the close because its closing prices exist from 2024-25 on, where FDUK Pinnacle closing prices end 2026-01-08 ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md), "Considered options"). That close is defined for EPL only.
- **EV beats realised P&L as the score because of variance** `(inferred)`. A bet's P&L is a win or a loss of the whole stake, while its EV against the close varies only with how far the price moves before kick-off, so EV shows an edge in far fewer bets.
- **P&L is reported beside EV.** Positive EV with P&L below it by more than variance explains means the close is the wrong benchmark for that market `(inferred)`.
- **Promote or kill.** A candidate is promoted when the event-clustered bootstrap 95% CI on its mean EV is above zero, and killed when the CI is at or below zero.
- **Real money needs stake capacity too.** UK retail books restrict accounts that win, so a promoted candidate is also judged on how much it can stake `(inferred)`.

## What scoring exists today

`odds_core.paper_trade_models.PaperTrade` has no field for the candidate or its version, and stores no close, so no candidate is scored yet (#604). MLB has no close defined (#604).

## Related

- [About the system and its pivots](project-overview.md)
- [Paper trading](../agent/paper-trading.md)
- [Retail-EV evaluation](../evaluation/retail-ev-evaluation.md)
