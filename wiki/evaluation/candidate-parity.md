---
type: reference
title: Parity check
description: What odds candidates parity replays, how it rebuilds each decision from its recorded input snapshots and events, how it reports data differences from the current database, its exit code, and the backfill that fills live bets' estimated EV from the same replay.
sources:
  - odds_analytics.candidate_parity
  - odds_cli.commands.candidates.parity
  - odds_core.bet_models.CandidateDecision
  - odds_analytics.estimate_backfill
  - odds_cli.commands.candidates.backfill_estimates
mentions:
  - odds_lambda.candidate_rules.rules.apply_rule
  - odds_lambda.candidate_rules.rules.run_rule
  - odds_lambda.candidate_rules.market_state.load_market_state
  - odds_lambda.candidate_rules.market_state.build_market_state
status: stable
---

# Parity check

`odds candidates parity <name> [--version N]` checks that the replay of each recorded decision reproduces the [bets](../foundations/glossary.md#bet) the [run-candidates job](run-candidates-job.md#the-decision-record) recorded for it, so the [replay backtest](candidate-backtest.md) and the [forward test](../foundations/glossary.md#forward-test) are known to run one rule on one market state. The task is on [From idea to verdict](howto-take-an-idea-to-a-verdict.md).

## Command

`odds_cli.commands.candidates.parity` finds the rule candidate in the registry, which `odds_cli.commands.candidates._load_registry` validates (`odds_lambda.candidate_rules.pinning.validate_candidate_rules`), so a rule whose code changed without a new version fails there. `--version` defaults to the registry's version. `odds_analytics.candidate_parity.check_parity` raises `ValueError` for another version, because the rule and params of an earlier version are only in git history, and for a [model rule](candidate-rules.md#model-rules), which the live runner never runs and so has no decisions.

The output is the counts of decisions checked, matched, mismatched and data differences, then each mismatched decision with its live and replayed bets and each data difference with both replays. The exit code is 1 when any decision is a mismatch, and 0 otherwise, including when no decision was checked, which prints a "no decisions to check" warning.

Run it on the production host: `odds copy from-prod` copies no bets or decisions, so a development database holds none.

## Replay of a decision

`odds_analytics.candidate_parity.check_parity` takes every `odds_core.bet_models.CandidateDecision` (keyed by the integer event id) of the candidate version that is not missed (the recorded `leagues` column is not read, so a database not yet migrated to hold it can be checked; a missed decision, late or without inputs, placed nothing and recorded no snapshot), in the order the live runs took them (`known_at`, then id; a kick-off moved earlier can give a later decision an earlier decision time), and makes two comparisons. Both apply [once per selection](candidate-rules.md#once-per-selection) by passing `bet_selections` to the replay of an event's later decision.

**Parity, from the recorded inputs.** `odds_analytics.candidate_parity.recorded_quotes` loads the prices of the decision's `input_snapshot_ids` (the retail snapshot it decided from, then any Polymarket snapshot its rule read, then each extra market's snapshots, then each lookback point's) by id, stamped within the params' maximum input age before the decision time (or, for a lookback point, before its own time, [lookback points](candidate-rules.md#lookback-points)), whatever event they now belong to. `odds_analytics.candidate_parity.recorded_intents` builds the state from exactly those prices with `odds_lambda.candidate_rules.market_state.build_market_state` at the decision's recorded decision time, with the params' maximum input age and the candidate's [book set](candidate-rules.md#book-set), by the same rules as the live run ([market state](candidate-rules.md#market-state)), and `odds_lambda.candidate_rules.rules.apply_rule` runs the rule with the decision's `kickoff_at_decision` in place of the event's current kick-off, and the bets are compared with the decision's bets as a sorted set of outcome, line, book, price and stake. `bet_selections` are the selections (outcome and line) of the bets the event's earlier decisions placed, which is what the live run saw, so a mismatch at an earlier decision does not spread to a later one. A zero-bet decision matches a zero-bet replay. Any difference is a `mismatch`: a bug in the rule or the state, or a recorded snapshot missing from the database (listed on the mismatch). Nothing in this comparison depends on the database's current content beyond the recorded snapshots' prices, so a later twin merge or a commit lag cannot change it. A decision recorded before `input_snapshot_ids` existed has none, and a replay from none places no bet. A rule that declares [fixture inputs](candidate-rules.md#fixture-context) gets the context the decision recorded (`odds_core.bet_models.CandidateDecision.input_fixture`, rebuilt by `odds_lambda.candidate_rules.FixtureContext.from_record`), not one read from the events again, because scores, kick-offs and ids change after a decision. A recorded event (`input_event_ids`) absent from the database makes the decision a mismatch and is listed (`odds_analytics.candidate_parity.DecisionParity.missing_event_ids`), as a missing snapshot is. The data comparison reads the context from the database as it is now, so a corrected score or a rescheduled event shows there as a data difference.

**Data differences, from the database now.** `odds_lambda.candidate_rules.rules.run_rule` replays the decision with the candidate's book set and `known_at` the decision's and the params' maximum input age, loading the snapshots from the database as it is now ([known_at](candidate-rules.md#known_at-and-recorded-inputs)). Its `bet_selections` are the selections its own replays of the event's earlier decisions bet, as the backtest chains them. When its bets differ from the recorded-input replay, the decision is listed as a data difference: snapshots stored after the run but stamped before the decision, a twin merged since, or rows committed after the run built its row objects (`odds_core.models.OddsSnapshot.created_at` is set when the row is built, not committed). It is informational and never affects the exit code, and it says the backtest, which loads from the database, would not reproduce the live bets.

| Status | Meaning |
|---|---|
| `match` | The recorded-input replay reproduces the live bets |
| `mismatch` | It does not |

## Estimate backfill

`odds candidates backfill-estimates [--apply]` fills `odds_core.bet_models.Bet.estimated_ev` of rule-candidate bets placed without one, from the replay of their decisions. `odds_analytics.estimate_backfill.plan_backfill` takes every decision-placed bet grouped by candidate version. Every version of a name the registry holds as a rule candidate whose rule is not a [model rule](candidate-rules.md#model-rules) is replayed decision by decision under the registry entry's rule, params and [book set](candidate-rules.md#book-set): the registry holds only a name's current version, and a name's rule never changes across versions, so an earlier version is replayed with the current entry, in the order and with the once-per-selection chaining of the parity comparison above (`odds_analytics.candidate_parity.recorded_quotes` and `odds_analytics.candidate_parity.recorded_intents`, `bet_selections` from the bets of the event's earlier decisions). Each recorded bet takes the `odds_lambda.candidate_rules.BetIntent.estimated_ev` of the replayed bet with its outcome, line, book and price (`odds_analytics.estimate_backfill.BetMatch`, prices compared at `odds_analytics.estimate_backfill.PRICE_DECIMALS` decimal places, as parity compares them; `odds_analytics.estimate_backfill.BetEstimate`). A recorded bet with no such replayed bet, including one of an earlier version whose params or book set differ from the current entry's where that changes the bet, is unmatched and keeps none. The bets of a name the registry does not hold, or whose rule is a model rule, are counted as skipped.

`odds_cli.commands.candidates.backfill_estimates` prints, per version, the bets, how many matched and how many carry an estimate, the unmatched ones with their decision and event, the bets whose stored estimate differs from the replay's (`odds_analytics.estimate_backfill.VersionBackfill.mismatched`), the count to write, and each skipped version with its count of bets. Without `--apply` it writes nothing. With it, `odds_analytics.estimate_backfill.apply_backfill` writes the estimate of each matched bet that has none stored (`odds_analytics.estimate_backfill.VersionBackfill.to_write`) and commits. A stored estimate is never overwritten, so a rerun writes nothing. It writes to whatever `DATABASE_URL` names, so on the production host back the database up first ([back up production](../foundations/howto-back-up-production-database.md)).
