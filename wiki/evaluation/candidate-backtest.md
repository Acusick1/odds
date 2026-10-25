---
type: reference
title: Replay backtest
description: What odds candidates backtest replays, the decision state and timing it uses, how a model rule is tuned and refitted during the replay, the comparison candidate, how bets are scored, the coverage it reports and its JSON output.
sources:
  - odds_analytics.candidate_backtest
  - odds_analytics.candidate_model.ModelReplay
  - odds_analytics.candidate_model.refit_blocks
  - odds_cli.commands.candidates.backtest
  - odds_lambda.candidate_rules.rules.run_rule
  - odds_lambda.storage.event_merge.canonical_events
mentions:
  - odds_core.candidates.load_candidate_file
status: stable
---

# Replay backtest

`odds candidates backtest <name-or-path> --from YYYY-MM-DD --to YYYY-MM-DD` replays a [mechanical candidate's](candidate-registry.md) rule over completed events and scores the bets with the code the forward-test report uses. Development data can kill an idea and never promotes one ([goal and strategy](../foundations/goal-and-strategy.md#how-a-candidate-is-judged)); the task is on [From idea to verdict](howto-take-an-idea-to-a-verdict.md).

## Command

`odds_cli.commands.candidates.backtest` resolves its argument with `odds_cli.commands.candidates._resolve`: a path to an existing file loads that file, and any other argument is a registered candidate name. `--from` and `--to` are UTC days, both inclusive. `--json <path>` also writes the result as JSON. `--compare <name-or-path>` also replays a second candidate, resolved the same way, on the events the first decided ([comparison](#comparison)). An unknown candidate, an agent candidate, a bad date, `--to` before `--from` or an invalid rule or params prints the error and exits 1.

A file loaded by path may be an unregistered draft. `odds_core.candidates.load_candidate_file` parses it with the registry's model, but does not require the file name to match `name` and does not require `odds_core.candidates.MechanicalCandidate.rule_sha256`. `odds_analytics.candidate_backtest.hash_status` reports whether the recorded hash is `match`, `mismatch` or `absent` against the rule's current hash ([rule pinning](candidate-rules.md#rule-pinning)), and the command prints a warning unless it matches. The registry rejects a mechanical file without the hash.

## Replay

`odds_analytics.candidate_backtest.replay` runs these steps.

1. **Events.** `odds_analytics.candidate_backtest.final_events` selects the candidate's sport's `FINAL` events with both scores and a kick-off within the days, ordered by kick-off and id. `odds_lambda.storage.event_merge.canonical_events` then keeps one event per match, the one `odds_lambda.storage.event_merge.find_match_id` names, which is the same helper the [run-candidates job](run-candidates-job.md#one-run) uses, so the replay decides the events the job decides. A match whose canonical event lacks scores or is not `FINAL` drops out.
2. **Decision times.** `odds_lambda.candidate_rules.rules.decision_times` of the event's kick-off: each entry of the params' `minutes_before_kickoff` with its time, decided earliest first. A completed event has only its final kick-off, so a rescheduled fixture is decided against its final time and not the one a live run saw ([parity check](candidate-parity.md) uses the recorded one).
3. **State.** At each decision time, `odds_lambda.candidate_rules.rules.run_rule` loads the state, within the params' maximum input age, with `known_at=None`, so a snapshot counts when it is stamped at or before the decision time, whenever it was stored: backfilled rows were created long after their snapshot times ([as-of rules](candidate-rules.md#market-state)). The live runner and the parity check decide through the same function. Only a snapshot carrying UK retail prices and a Betfair Exchange row together is an input (in practice live OddsPortal snapshots, and live Odds API ones that quote the exchange), so a range before that collection, or one whose snapshots fall outside the params' maximum input age, has no inputs and no bets. A decision without inputs is what the live runner records as missed; the replay counts it in coverage.
4. **Bets.** The rule's bets, less those on outcomes an earlier decision time of the event already bet, as the live runner drops them ([once per outcome](candidate-rules.md#once-per-outcome)). Each bet is placed at its decision time, and the bets are ordered by it, then by kick-off and rule order, as the report orders paper trades by placement. Each is settled from the final score with `odds_core.odds_math.determine_h2h_winner` and `odds_core.odds_math.calculate_profit_from_decimal`, as `odds_lambda.paper_trading.settle_trades` does.
5. **Close.** Each `1x2` bet's `p_close` comes from `odds_analytics.candidate_report.selection_close_probability` over the snapshots `odds_analytics.candidate_report.close_candidate_snapshots` selects. An `h2h` bet, every MLB bet, has none.

A [model rule](candidate-rules.md#model-rules) decides only part of the range, with its model refitted as the replay goes ([model rules](#model-rules)).

Events are processed in batches of `odds_analytics.candidate_backtest.EVENT_BATCH_SIZE`, with the session's loaded snapshots dropped after each batch: one query per decision reads only the maximum input age before its decision time, and the close snapshots are read once per batch for events that have a bet.

## Model rules

For a model rule, `odds_analytics.candidate_backtest.replay` takes the range's canonical events in kick-off order and leaves the first `training_matches` of them to training only, with any later match that kicks off at the same time as the last of them (`odds_analytics.candidate_model.training_prefix`): they are never decided. `odds_analytics.candidate_model.refit_blocks` splits the rest, the scored matches, into blocks of `refit_every_matches` in kick-off order; the last block may be partial. A range with no match after its training prefix scores nothing.

`odds_analytics.candidate_model.ModelReplay` first tunes the model's hyperparameters on the training matches that kicked off before the first scored decision time, the matches the first refit trains on, so every tuning label exists before any scored decision ([hyperparameters](candidate-rules.md#the-retail-ev-model-rule)); no scored match reaches the tuning, and a training match kicking off later is left out of it. It then refits the model before each block, at the block's earliest decision time, with those hyperparameters, on the [training rows](candidate-rules.md#the-retail-ev-model-rule) of every range match that kicked off before that time: the prefix and the earlier scored matches. A prefix match that kicks off after a scored match's first decision waits for a later block. Each match's rows are built once, from all of its snapshots, and only a match with a live Betfair close has rows. A tuning or a refit without any training row raises `ValueError`, which the command prints before exiting 1. The block's matches are then decided as in [replay](#replay), steps 2 to 5, with the rule bound to that fit, so the model never sees a match before it is scored.

`odds_analytics.candidate_backtest.BacktestResult.tuning` records the chosen hyperparameters, the trial count, the tuning matches, the folds used with each one's validation row count, and the mean fold weighted MSE of the model and of the exchange-at-t baseline (`odds_analytics.candidate_model.Tuning`). `odds_analytics.candidate_backtest.BacktestResult.refits` records each fit (`odds_analytics.candidate_model.Refit`): its time, the matches with training rows, the row count and the scored event ids. The command prints both as tables above the coverage.

## Comparison

`--compare` replays the second candidate over the same range with `only_events` set to the first candidate's `odds_analytics.candidate_backtest.BacktestResult.decided_event_ids`: a model rule's scored matches, or every range event for a rule without a model. Its k is its own family's. The command prints its tables after the first candidate's, then one table with both candidates' estimates over every bet with a close. The command refuses a model rule as the second candidate, because it trains on its own range. The re-test of the retail-EV model compares its draft, `experiments/configs/candidate_retail_ev_model_epl.yaml`, with `exchange-gap-epl` this way ([#657](https://github.com/Acusick1/odds/issues/657)).

## Scoring

`odds_analytics.candidate_backtest.score` passes the bets, as `odds_analytics.candidate_report.ScoredBet`, to `odds_analytics.candidate_report.summarize_candidate`, with the family size k the command computes: the type, EV, prefix, checkpoint and verdict code that [the report](candidate-registry.md#odds-candidates-report) applies to paper trades. All backtest bets are settled. The no-close grace (`odds_analytics.candidate_report.CLOSE_GRACE`) is measured from the run time, so a range ending within 7 days of today leaves its last bets pending. k is the family's versions with any recorded decision, from `odds_analytics.candidate_report.family_starts` and `odds_analytics.candidate_report.family_size_at`, plus the draft as one more. A registered candidate that already has decisions is not counted twice ([family size](candidate-registry.md#odds-candidates-report)). One k applies to every checkpoint, since a replay has no placement history to count versions against. The command reads the registry with `odds_core.candidates.load_candidates` alone and the recorded decisions from the database, so a registered rule whose code changed since its hash was recorded does not stop a draft backtest. The bootstrap uses the same seed, so a replay gives the same numbers every time.

The checkpoint is the verdict's: the first 100, 200, … 500 bets with a close, which stop at the first time the verdict code reports the promote bar met or a kill. `odds_analytics.candidate_backtest.BacktestResult.overall` is a further estimate over every bet with a close in the scoreable prefix, for a range with more bets than the verdict looks at.

## Coverage

`odds_analytics.candidate_backtest.Coverage` counts, for the replayed events and their decisions (one per event and decision time):

| Field | Counts |
|---|---|
| `events` | Canonical events decided: those in range, or for a model rule its scored matches (`odds_analytics.candidate_backtest.BacktestResult.range_events` counts the range, before any `only_events` filter) |
| `decisions` | Decisions: each event once per decision time |
| `with_input` | Decisions with a snapshot within the maximum input age at the decision time |
| `with_exchange` | Decisions with an exchange price at the decision time |
| `with_retail` | Decisions with at least one UK retail quote at that time |
| `with_bet` | Decisions that placed at least one bet |
| `bets`, `bets_with_close` | Bets, and those with a close |

The command prints the verdict as `meets_promote_bar` (with "development data, not a promotion"), `kill`, `continue` or `not_yet`, never `promote`, and `--json` writes the same strings (`odds_cli.commands.candidates._backtest_verdict`). It prints these beside the no-close and pending counts, k, the checkpoint and overall estimates with their 95% interval and promote interval (the promote level is 1 − 0.01/k), stake, P&L and ROI, and the coverage table's title names the decision times and the maximum input age the params set, so a draft that widened the age says so. A low `with_input`, `with_exchange` or `with_retail` count against `decisions` says the range lacks the data for the rule's decision times, which no verdict can repair.

## JSON output

`--json` writes `dataclasses.asdict` of `odds_analytics.candidate_backtest.BacktestResult`: the candidate's name, version, rule, params and hash status, the range, the coverage, the summary (including `odds_analytics.candidate_report.CandidateSummary.family_versions`), the overall estimate, the range's event count, a model rule's tuning, refits and `odds_analytics.candidate_backtest.BacktestResult.library_versions` (the XGBoost, Optuna and NumPy versions, from `odds_analytics.candidate_model.library_versions`), and the comparison candidate's result in the same form, or `null`. Dates and enums are written as strings.
