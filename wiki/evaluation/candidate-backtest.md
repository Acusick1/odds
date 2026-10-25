---
type: reference
title: Replay backtest
description: What odds candidates backtest replays, the decision state and timing it uses, how bets are scored, the coverage it reports and its JSON output.
sources:
  - odds_analytics.candidate_backtest
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

`odds_cli.commands.candidates.backtest` resolves its argument with `odds_cli.commands.candidates._resolve`: a path to an existing file loads that file, and any other argument is a registered candidate name. `--from` and `--to` are UTC days, both inclusive. `--json <path>` also writes the result as JSON. An unknown candidate, an agent candidate, a bad date, `--to` before `--from` or an invalid rule or params prints the error and exits 1.

A file loaded by path may be an unregistered draft. `odds_core.candidates.load_candidate_file` parses it with the registry's model, but does not require the file name to match `name` and does not require `odds_core.candidates.MechanicalCandidate.rule_sha256`. `odds_analytics.candidate_backtest.hash_status` reports whether the recorded hash is `match`, `mismatch` or `absent` against the rule's current hash ([rule pinning](candidate-rules.md#rule-pinning)), and the command prints a warning unless it matches. The registry rejects a mechanical file without the hash.

## Replay

`odds_analytics.candidate_backtest.replay` runs these steps.

1. **Events.** `odds_analytics.candidate_backtest.final_events` selects the candidate's sport's `FINAL` events with both scores and a kick-off within the days, ordered by kick-off and id. `odds_lambda.storage.event_merge.canonical_events` then keeps one event per match, the one `odds_lambda.storage.event_merge.find_match_id` names, which is the same helper the [run-candidates job](run-candidates-job.md#one-run) uses, so the replay decides the events the job decides. A match whose canonical event lacks scores or is not `FINAL` drops out.
2. **Decision time.** `odds_lambda.candidate_rules.rules.decision_time` of the event's kick-off. A completed event has only its final kick-off, so a rescheduled fixture is decided against its final time and not the one a live run saw ([parity check](candidate-parity.md) uses the recorded one).
3. **State.** `odds_lambda.candidate_rules.rules.run_rule` loads the state with `known_at=None`, so a snapshot counts when it is stamped at or before the decision time, whenever it was stored: backfilled rows were created long after their snapshot times ([as-of rules](candidate-rules.md#market-state)). The live runner and the parity check decide through the same function.
4. **Bets.** The rule's bets, in kick-off order and then rule order. Each is settled from the final score with `odds_core.odds_math.determine_h2h_winner` and `odds_core.odds_math.calculate_profit_from_decimal`, as `odds_lambda.paper_trading.settle_trades` does.
5. **Close.** Each `1x2` bet's `p_close` comes from `odds_analytics.candidate_report.selection_close_probability` over the snapshots `odds_analytics.candidate_report.close_candidate_snapshots` selects. An `h2h` bet, every MLB bet, has none.

Events are processed in batches of `odds_analytics.candidate_backtest.EVENT_BATCH_SIZE`, with the session's loaded snapshots dropped after each batch: one query per event reads only the `odds_lambda.candidate_rules.market_state.SNAPSHOT_LOOKBACK` before its decision time, and the close snapshots are read once per batch for events that have a bet.

## Scoring

`odds_analytics.candidate_backtest.score` passes the bets, as `odds_analytics.candidate_report.ScoredBet`, to `odds_analytics.candidate_report.summarize_candidate`, with the family size k the command computes: the type, EV, prefix, checkpoint and verdict code that [the report](candidate-registry.md#odds-candidates-report) applies to paper trades. All backtest bets are settled. The no-close grace (`odds_analytics.candidate_report.CLOSE_GRACE`) is measured from the run time, so a range ending within 7 days of today leaves its last bets pending. k is the family's versions with any recorded decision, from `odds_analytics.candidate_report.family_starts` and `odds_analytics.candidate_report.family_size_at`, plus the draft as one more. A registered candidate that already has decisions is not counted twice ([family size](candidate-registry.md#odds-candidates-report)). One k applies to every checkpoint, since a replay has no placement history to count versions against. The command reads the registry with `odds_core.candidates.load_candidates` alone and the recorded decisions from the database, so a registered rule whose code changed since its hash was recorded does not stop a draft backtest. The bootstrap uses the same seed, so a replay gives the same numbers every time.

The checkpoint is the verdict's: the first 100, 200, … 500 bets with a close, which stop at the first time the verdict code reports the promote bar met or a kill. `odds_analytics.candidate_backtest.BacktestResult.overall` is a further estimate over every bet with a close in the scoreable prefix, for a range with more bets than the verdict looks at.

## Coverage

`odds_analytics.candidate_backtest.Coverage` counts, for the replayed events:

| Field | Counts |
|---|---|
| `events` | Canonical events in range |
| `with_exchange` | Events with an exchange price at the decision time |
| `with_retail` | Events with at least one UK retail quote at that time |
| `with_bet` | Events where the rule took at least one bet |
| `bets`, `bets_with_close` | Bets, and those with a close |

The command prints the verdict as `meets_promote_bar` (with "development data, not a promotion"), `kill`, `continue` or `not_yet`, never `promote`, and `--json` writes the same strings (`odds_cli.commands.candidates._backtest_verdict`). It prints these beside the no-close and pending counts, k, the checkpoint and overall estimates with their 95% interval and promote interval (the promote level is 1 − 0.01/k), stake, P&L and ROI. A low `with_exchange` or `with_retail` count says the range lacks the data for the rule's decision time, which no verdict can repair.

## JSON output

`--json` writes `dataclasses.asdict` of `odds_analytics.candidate_backtest.BacktestResult`: the candidate's name, version, rule, params and hash status, the range, the coverage, the summary (including `odds_analytics.candidate_report.CandidateSummary.family_versions`) and the overall estimate. Dates and enums are written as strings.
