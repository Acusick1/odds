---
type: explanation
title: Goal and strategy
description: What the project is for, what counts as a candidate, how a candidate is scored, promoted or killed, and which parts of the workflow exist today.
sources:
  - odds_core.paper_trade_models.PaperTrade
  - odds_analytics.retail_ev
  - odds_core.betfair_close.event_close
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

Mechanical rules are the current focus. One candidate is registered: `exchange-gap-epl` (`candidates/exchange-gap-epl.yaml`, idea #626) backs the longest UK retail price when the Betfair Exchange fair probability says it is too long, at 60 and 30 minutes before kick-off (`minutes_before_kickoff: [60, 30]`), betting each outcome of a match at most once ([decision times](../evaluation/candidate-rules.md#decision-times)), from one snapshot no older than 10 minutes that carries both prices (`max_input_age_minutes`). It was registered at KO-60 by #626's pre-registered fallback, because its development backtest had too few bets to judge. Its KO-30 time comes from development screens that suggested more bets at similar EV ([#692](https://github.com/Acusick1/odds/pull/692)). Its live forward test, from the `start_date` in its file, is the verdict. The repo also holds:

- **The EPL and MLB betting agents** synthesise prices, exchange data and match news ([The betting agent](../agent/betting-agent.md)). Neither is registered or being run.
- **The EPL retail-EV model** predicts a retail bet's EV against the close ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)). Its development results are in [Price-only models on the retail-EV target](../evaluation/exp-epl-retail-ev-price-only.md), and it places no bets. Its one re-test on dense live data is a draft backtest of the [retail-EV model rule](../evaluation/candidate-rules.md#the-retail-ev-model-rule) against `exchange-gap-epl` on the same matches ([#657](https://github.com/Acusick1/odds/issues/657)).

## How a candidate is judged

- **Development data generates and kills ideas, and never promotes one.** The EPL 2024–26 development data has been examined many times ([price-only runs](../evaluation/exp-epl-retail-ev-price-only.md)). Choosing the best of many ideas on data already seen finds edges that are not there `(inferred)`.
- **A candidate is judged on a live forward test.** It paper-trades from a start date at a fixed version. Any number of candidates run in parallel, and none waits on another.
- **The score is EV at the price taken against the Betfair Exchange close:** `p_close × decimal_odds − 1`, with the close as defined for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target) (`odds_core.betfair_close.event_close`, over the stored Betfair Exchange prices). Betfair Exchange is the close because its closing prices exist from 2024-25 on, where FDUK Pinnacle closing prices end 2026-01-08 ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md), "Considered options"). That close is defined for EPL only.
- **EV beats realised P&L as the score because of variance** `(inferred)`. A bet's P&L is a win or a loss of the whole stake, while its EV against the close varies only with how far the price moves before kick-off, so EV shows an edge in far fewer bets.
- **P&L is reported beside EV.** Positive EV with P&L below it by more than variance explains means the close is the wrong benchmark for that market `(inferred)`.
- **Promote or kill at fixed checkpoints.** A candidate is scored on its first 100, 200, … bets, never on a running total, so the verdict does not depend on when it is read. Bets are counted in placement order and only bets with a close count. The counted bets stop at a trade that is still open, or has no close and kicked off less than 7 days ago, because the FDUK closing price can still arrive; a trade with no close after 7 days is skipped. The score is mean EV per unit staked, with an event-clustered bootstrap CI. The first promote or kill verdict stands, whatever later bets show.
  - **Promote** when the lower bound of the CI at confidence 1 − 0.01/k is above zero. k is the number of versions in the candidate's family, the mechanical candidates with the same `rule` and `sport`, with a recorded decision at or before the placement of the checkpoint's nth bet. Each checkpoint is judged at its own k, so a variant that first decides later never reverses a verdict already reached. A single version gives the 99% interval; three give about 99.7%. This is a Bonferroni correction that holds the family-wise false-promote rate at 1%, so variants of one rule that bet on the same matches cannot buy extra chances ([decision](https://github.com/Acusick1/odds/issues/652)). k can grow while a candidate runs, which only tightens the bar of its later checkpoints. A draft backtest counts itself as one more version, at one k for the whole replay. An agent candidate's family is itself alone.
  - **A version counts toward k from its first recorded decision.** A version with no decision has taken no look, so a registered variant that has not yet decided, or was superseded before it did, does not count. Killed versions count. The registry `start_date` plays no part, because it can be back-dated and a decision cannot.
  - **Kill** when the 95% CI's upper bound is at or below zero, or after 500 bets without promotion. The kill bar is per candidate and does not depend on k: a wrong kill costs one missed idea, and loosening it would spend scarce bets on weak variants.
- **Real money needs stake capacity too.** UK retail books restrict accounts that win, so a promoted candidate is also judged on how much it can stake `(inferred)`. The gate has no rule until a candidate is promoted.

## What exists today

The workflow for a mechanical rule exists end to end, from idea to verdict ([From idea to verdict](../evaluation/howto-take-an-idea-to-a-verdict.md), #608):

- The [candidate registry](../evaluation/candidate-registry.md), with no candidate in it yet.
- The candidate, its version and the git commit on each paper trade (`odds_core.paper_trade_models.PaperTrade`, with the decision that placed it), with the candidate checked against the registry when the bet is placed through `odds_mcp.server.paper_bet` ([paper trading](../agent/paper-trading.md#candidate-fields)).
- The close, computed from stored snapshots when a report runs, so a trade stores none. MLB has no close defined, so its trades are reported without EV.
- `odds candidates report`, which gives the checkpoint verdict ([Candidate registry](../evaluation/candidate-registry.md#odds-candidates-report)).
- The [mechanical rule interface](../evaluation/candidate-rules.md) and its first rule, the exchange gap, with the market state at decision time and a code hash that each mechanical candidate records.
- The [run-candidates job](../evaluation/run-candidates-job.md), which places each active mechanical candidate's flat-stake paper trades at its decision times and records every decision.

- The [replay backtest](../evaluation/candidate-backtest.md), which scores a rule on stored data with the report's code, and the [parity check](../evaluation/candidate-parity.md), which compares the runner's decisions with their replay.

Missing:

- The stake-capacity gate, which has no rule until a candidate is promoted.
- A close for MLB, so MLB candidates are scored on P&L only.

## Related

- [Take an idea to a verdict](../evaluation/howto-take-an-idea-to-a-verdict.md)
- [About the system and its pivots](project-overview.md)
- [Paper trading](../agent/paper-trading.md)
- [Retail-EV evaluation](../evaluation/retail-ev-evaluation.md)
