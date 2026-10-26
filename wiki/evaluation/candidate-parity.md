---
type: reference
title: Parity check
description: What odds candidates parity replays, how it rebuilds each decision from its recorded input snapshots, how it reports data differences from the current database, and its exit code.
sources:
  - odds_analytics.candidate_parity
  - odds_cli.commands.candidates.parity
  - odds_core.paper_trade_models.CandidateDecision
mentions:
  - odds_lambda.candidate_rules.rules.apply_rule
  - odds_lambda.candidate_rules.rules.run_rule
  - odds_lambda.candidate_rules.market_state.load_market_state
status: stable
---

# Parity check

`odds candidates parity <name> [--version N]` checks that the replay of each recorded decision reproduces the paper trades the [run-candidates job](run-candidates-job.md#the-decision-record) placed, so the [replay backtest](candidate-backtest.md) and the forward test are known to run one rule on one market state. The task is on [From idea to verdict](howto-take-an-idea-to-a-verdict.md).

## Command

`odds_cli.commands.candidates.parity` finds the candidate in the registry, which `odds_cli.commands.candidates._load_registry` validates, so a rule whose code changed without a new version fails there. `--version` defaults to the registry's version. `odds_analytics.candidate_parity.check_parity` raises `ValueError` for another version, because the rule and params of an earlier version are only in git history, and for a [model rule](candidate-rules.md#model-rules), which the live runner never runs and so has no decisions.

The output is the counts of decisions checked, matched, mismatched and data differences, then each mismatched decision with its live and replayed bets and each data difference with both replays. The exit code is 1 when any decision is a mismatch, and 0 otherwise, including when no decision was checked, which prints a "no decisions to check" warning.

Run it on the production host: `odds copy from-prod` copies no trades or decisions, so a development database holds none.

## Replay of a decision

`odds_analytics.candidate_parity.check_parity` takes every `odds_core.paper_trade_models.CandidateDecision` of the candidate version that is not missed (a missed decision, late or without inputs, placed nothing and recorded no snapshot), in the order the live runs took them (`known_at`, then id; a kick-off moved earlier can give a later decision an earlier decision time), and makes two comparisons. Both apply [once per selection](candidate-rules.md#once-per-selection) by passing `bet_selections` to the replay of a match's later decision.

**Parity, from the recorded inputs.** `odds_lambda.candidate_rules.market_state.load_market_state` loads the quotes of the decision's `input_snapshot_ids` (the one snapshot it decided from) by id, stamped within the params' maximum input age before the decision time, whatever event they now belong to, and builds the state from exactly those snapshots at the decision's recorded decision time, with the params' maximum input age, by the same rules as the live run ([market state](candidate-rules.md#market-state)), `odds_lambda.candidate_rules.rules.apply_rule` runs the rule with the decision's `kickoff_at_decision` in place of the event's current kick-off, and the bets are compared with the decision's paper trades as a sorted set of outcome, bookmaker, price and stake. `bet_selections` are the selections of the paper trades the match's earlier decisions placed, each with the line `None` because a paper trade records none, which is what the live run saw, so a mismatch at an earlier decision does not spread to a later one. A zero-bet decision matches a zero-bet replay. Any difference is a `mismatch`: a bug in the rule or the state, or a recorded snapshot missing from the database (listed on the mismatch). Nothing in this comparison depends on the database's current content beyond the recorded snapshots' prices, so a later twin merge or a commit lag cannot change it. A decision recorded before `input_snapshot_ids` existed has none, and a replay from none places no bet.

**Data differences, from the database now.** `odds_lambda.candidate_rules.rules.run_rule` replays the decision with `known_at` the decision's and the params' maximum input age, loading the snapshots from the database as it is now ([known_at](candidate-rules.md#known_at-and-recorded-inputs)). Its `bet_selections` are the selections its own replays of the match's earlier decisions bet, as the backtest chains them. When its bets differ from the recorded-input replay, the decision is listed as a data difference: snapshots stored after the run but stamped before the decision, a twin merged since, or rows committed after the run built its row objects (`odds_core.models.OddsSnapshot.created_at` is set when the row is built, not committed). It is informational and never affects the exit code, and it says the backtest, which loads from the database, would not reproduce the live bets.

| Status | Meaning |
|---|---|
| `match` | The recorded-input replay reproduces the live bets |
| `mismatch` | It does not |
