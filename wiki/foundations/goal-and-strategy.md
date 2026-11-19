---
type: explanation
title: Goal and strategy
description: What the project is for, how a candidate is scored, promoted or killed, and which parts of the workflow exist today.
sources:
  - odds_core.bet_models.Bet
  - odds_analytics.retail_ev
  - odds_core.close.event_close
status: stable
---

# Goal and strategy

## Goal

The goal is profit from betting at UK retail books on EPL and MLB. No approach is privileged: agents, models, rules or anything else are tried if they might get there, and several are tried at once.

## Candidates

- **Registry.** A registered [candidate](glossary.md#candidate) is one YAML file in `candidates/` ([Candidate registry](../evaluation/candidate-registry.md)), holding its definition and its status: active, promoted or killed. Status changes by pull request, and no database table copies it.
- **Version.** Each candidate has a [version](glossary.md#version); a collection fix that changes the prices its earlier decisions saw bumps it, so a series decided on the faulty data is not judged with the fixed one. For an agent, a change to `packages/odds-mcp/agents/common.md`, its league's `base.md` or the model passed by `odds_lambda.jobs.agent_run.run_claude_agent` is a new version. A change to `packages/odds-mcp/agents/<sport>/observations.md` is not: that file is the agent's own running playbook, and it is gitignored, so a version and git SHA do not pin it `(inferred)`.
- **Tracking.** An idea starts as a GitHub issue labelled `candidate`. Once registered, its file's status is its stage.

Rule candidates are the current focus. Nine are registered, all on the `uk_retail` [book set](glossary.md#book-set) ([ADR-0025](../../docs/adr/0025-book-set-is-part-of-a-candidate.md)): three in EPL alone and six pooled across six football leagues. Each records a hash of its rule's code (`odds_lambda.candidate_rules.pinning.rule_sha256`), and a code change that leaves every bet the same is a rehash rather than a new version, so its forward test continues ([rule pinning](../evaluation/candidate-rules.md#rule-pinning), [rehash ledger](../evaluation/candidate-registry.md#rehash-ledger)). Each rule is described on [candidate rules](../evaluation/candidate-rules.md#rules); the registered parameters are in each `candidates/<name>.yaml`.

- `exchange-gap-epl` (`candidates/exchange-gap-epl.yaml`, idea #626) runs the [exchange gap](../evaluation/candidate-rules.md#the-gap-rules) on `1x2` at 60 and 30 minutes before kick-off. It was registered at KO-60 by #626's pre-registered fallback, because its development backtest had too few bets to judge; its KO-30 time comes from development screens that suggested more bets at similar EV ([#692](https://github.com/Acusick1/odds/pull/692)).
- `polymarket-gap-epl` (`candidates/polymarket-gap-epl.yaml`, idea #655) runs the [Polymarket gap](../evaluation/candidate-rules.md#the-gap-rules) on `1x2` at the same times with a 1% threshold.
- `exchange-gap-epl-totals` (`candidates/exchange-gap-epl-totals.yaml`, [#708](https://github.com/Acusick1/odds/issues/708)) runs `exchange-gap-epl`'s rule and params on the totals market, from version 3's start on 14 November 2026. It is its own [family](glossary.md#family), and [development data](glossary.md#development-data) has no totals close to score it against.

The other six are the survivors of ideation round 1 ([#817](https://github.com/Acusick1/odds/issues/817)), each pooled across the EPL, La Liga, Serie A, the Bundesliga, Ligue 1 and the Championship ([ADR-0022](../../docs/adr/0022-pooled-league-candidates.md)). Each runs the cell its development screen selected and passed the one-shot sealed-holdout gate ([ADR-0021](../../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)); neither result promotes it ([From idea to verdict](../evaluation/howto-take-an-idea-to-a-verdict.md)):

- `exchange-steam-continuation` ([#771](https://github.com/Acusick1/odds/issues/771)) and `team-news-latency` ([#832](https://github.com/Acusick1/odds/issues/832)) run the [exchange-steam rule](../evaluation/candidate-rules.md#the-exchange-steam-rule).
- `single-book-outlier` ([#833](https://github.com/Acusick1/odds/issues/833)) runs a [retail-consensus rule](../evaluation/candidate-rules.md#the-retail-consensus-rules).
- `heavy-favourite-best-price` ([#835](https://github.com/Acusick1/odds/issues/835)) and `crowded-kickoff-gap` ([#840](https://github.com/Acusick1/odds/issues/840)) run [round-1 football rules](../evaluation/candidate-rules.md#the-football-rules-of-round-1).
- `exchange-gap-early` ([#842](https://github.com/Acusick1/odds/issues/842)) takes the exchange gap 6 hours before kick-off, in one family with `exchange-gap-epl`. Its sealed holdout read retail snapshots up to 60 minutes old; its registered version reads one no older than 10 minutes, as its issue fixed.

The three EPL candidates' forward tests start on 2026-11-14 and the six pooled ones' on 2026-11-13 (`start_date`), and each is judged by its [forward test](glossary.md#forward-test) alone.

The repo also holds:

- **The EPL and MLB betting agents** synthesise prices, exchange data and event news ([The betting agent](../agent/betting-agent.md)). Neither is registered or being run.
- **The EPL retail-EV model** predicts a retail bet's EV against the close (`odds_analytics.retail_ev`, which reads the price rows of stored snapshots, Polymarket's through `odds_core.book_price.polymarket_price`; [retail-EV target](../modelling/clv-target.md#the-retail-ev-target)). Its development results are in [Price-only models on the retail-EV target](../evaluation/exp-epl-retail-ev-price-only.md), and it places no bets. Its one re-test on dense live data is a draft backtest of the [retail-EV model rule](../evaluation/candidate-rules.md#the-retail-ev-model-rule) against `exchange-gap-epl` on the same events ([#657](https://github.com/Acusick1/odds/issues/657)).

## How a candidate is judged

- **Development data generates and kills ideas, and never promotes one.** The EPL 2024–26 development data has been examined many times ([price-only runs](../evaluation/exp-epl-retail-ev-price-only.md)). Choosing the best of many ideas on data already seen finds edges that are not there `(inferred)`.
- **A candidate is judged on its forward test.** Any number of candidates run in parallel, and none waits on another.
- **A candidate may pool several leagues into one verdict.** It lists the leagues it bets in (`leagues`), and its bets in all of them form one series in placement order, judged by the rules below as if from one league. A per-league breakdown is reported beside the verdict for information and never changes it. Adding or dropping a league is a new version, and every listed league must have a close for the candidate's market at registration ([ADR-0022](../../docs/adr/0022-pooled-league-candidates.md)).
- **The score is [retail EV](glossary.md#retail-ev):** `p_close × decimal_odds − 1`, with the [close](glossary.md#close) as computed for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target) (`odds_core.close.event_close`, over the stored Betfair Exchange prices; [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md), [ADR-0019](../../docs/adr/0019-nba-h2h-close-betfair-with-polymarket-fallback.md)). MLB has none. No other sharp price scores a candidate ([sharp references](../modelling/sharp-benchmark.md#which-reference-is-used-for-what)).
- **Retail EV beats realised P&L as the score because of variance** `(inferred)`. A bet's P&L is a win or a loss of the whole stake, while its retail EV varies only with how far the price moves before kick-off, so EV shows an edge in far fewer bets.
- **P&L is reported beside retail EV.** Positive EV with P&L below it by more than variance explains means the close is the wrong benchmark for that market `(inferred)`.
- **Promote or kill at fixed [checkpoints](glossary.md#checkpoint).** A candidate is scored on its first 100, 200, … bets, never on a running total, so the verdict does not depend on when it is read. Bets are counted in placement order and only bets with a close count. The counted bets stop at a bet that is still open, or has no close and kicked off less than 7 days ago, because the FDUK close can still arrive; a bet with no close after 7 days is skipped. The score is mean EV per unit staked, with an event-clustered bootstrap CI. The first promote or kill verdict stands, whatever later bets show.
  - **Promote** when the lower bound of the CI at confidence 1 − 0.01/k is above zero. k is the number of versions in the candidate's [family](glossary.md#family) with a recorded decision at or before the placement of the checkpoint's nth bet. A family's `rule` and market are fixed for a name at every version: another rule or market is registered under a new name. Each version counts by the league set it listed when it decided, so an earlier version is not counted under its successor's leagues. It is counted per version and is not transitive: a pooled {EPL, Serie A} version shares its family with an EPL-only version and with a Serie A-only one, but those two do not share one with each other ([ADR-0022](../../docs/adr/0022-pooled-league-candidates.md)). Each checkpoint is judged at its own k, so a variant that first decides later never reverses a verdict already reached. A single version gives the 99% interval; three give about 99.7%. This is a Bonferroni correction that holds the family-wise false-promote rate at 1%, so variants of one rule that bet on the same events cannot buy extra chances ([decision](https://github.com/Acusick1/odds/issues/652)). k can grow while a candidate runs, which only tightens the bar of its later checkpoints. A draft backtest counts itself as one more version, at one k for the whole replay. An agent candidate's family is itself alone.
  - **A version counts toward k from its first recorded decision.** A version with no decision has taken no look, so a registered variant that has not yet decided, or was superseded before it did, does not count. Killed versions count. The registry `start_date` plays no part, because it can be back-dated and a decision cannot.
  - **Kill** when the 95% CI's upper bound is at or below zero, or after 500 bets without promotion. The kill bar is per candidate and does not depend on k: a wrong kill costs one missed idea, and loosening it would spend scarce bets on weak variants.
- **Real money needs stake capacity too.** UK retail books restrict accounts that win, so a promoted candidate is also judged on how much it can stake `(inferred)`. The gate has no rule until a candidate is promoted.

## What exists today

The workflow for a rule candidate exists end to end, from idea to verdict ([From idea to verdict](../evaluation/howto-take-an-idea-to-a-verdict.md), #608):

- The [candidate registry](../evaluation/candidate-registry.md), holding the registered candidates above.
- The candidate, its version and the git commit on each bet (`odds_core.bet_models.Bet`, [bets](../agent/bets.md#candidate-fields)).
- The close, computed from stored price rows when a report runs, so a bet stores none.
- `odds candidates report`, which gives the checkpoint verdict ([Candidate registry](../evaluation/candidate-registry.md#odds-candidates-report)).
- The [rule interface](../evaluation/candidate-rules.md) and its rules, with the hash each rule candidate records.
- The [run-candidates job](../evaluation/run-candidates-job.md), which places each active rule candidate's flat-stake bets at its decision times and records every decision.
- The [replay backtest](../evaluation/candidate-backtest.md), which scores a rule on stored data with the report's code, and the [parity check](../evaluation/candidate-parity.md), which compares the runner's decisions with their replay.

Missing:

- The stake-capacity gate, which has no rule until a candidate is promoted.
- A close for MLB: until it exists, no rule candidate can register in MLB, and an MLB agent's bets are reported on P&L only.

## Related

- [Take an idea to a verdict](../evaluation/howto-take-an-idea-to-a-verdict.md)
- [About the system and its pivots](project-overview.md)
- [Bets](../agent/bets.md)
