---
type: reference
title: Parity check
description: What odds candidates parity replays, which inputs it takes from each decision, how a difference is classified as a match, storage lag or mismatch, and its exit code.
sources:
  - odds_analytics.candidate_parity
  - odds_cli.commands.candidates.parity
  - odds_core.paper_trade_models.CandidateDecision
status: stable
---

# Parity check

`odds candidates parity <name> [--version N]` checks that the replay of each recorded decision reproduces the paper trades the [run-candidates job](run-candidates-job.md#the-decision-record) placed, so the [replay backtest](candidate-backtest.md) and the forward test are known to run one rule on one market state. The task is on [From idea to verdict](howto-take-an-idea-to-a-verdict.md).

## Command

`odds_cli.commands.candidates.parity` finds the candidate in the registry, which `odds_cli.commands.candidates._load_registry` validates, so a rule whose code changed without a new version fails there. `--version` defaults to the registry's version. `odds_analytics.candidate_parity.check_parity` raises `ValueError` for another version, because the rule and params of an earlier version are only in git history.

The output is the counts of decisions checked, matched, storage lag and mismatches, then each storage-lag and mismatched decision with its live and replayed bets. The exit code is 1 when any decision is a mismatch, and 0 otherwise.

## Replay of a decision

`odds_analytics.candidate_parity.check_parity` takes every `odds_core.paper_trade_models.CandidateDecision` of the candidate version that is not missed, and replays each through `odds_lambda.candidate_rules.rules.run_rule` with the inputs the decision recorded:

- the decision's `odds_core.paper_trade_models.CandidateDecision.decision_time`;
- its `kickoff_at_decision`, which replaces the event's current kick-off, since the event may have moved;
- its `known_at`, so only snapshots stored by then count ([known_at](candidate-rules.md#known_at)), and not the paper trade's `placed_at`, because a decision that placed nothing has no trade.

The replayed bets are compared with the decision's paper trades as a sorted set of outcome, bookmaker, price and stake. A zero-bet decision matches a zero-bet replay. Twins need no handling: decisions are recorded on canonical events.

## Classification

| Status | Meaning |
|---|---|
| `match` | The replay reproduces the live bets |
| `storage_lag` | It does not, but moving `known_at` back to just before a snapshot reproduces them |
| `mismatch` | Neither; a bug in the rule, the state or the recorded inputs |

`odds_core.models.OddsSnapshot.created_at` is set when the row object is built, not when it is committed. A snapshot built just before `known_at` and committed after it was invisible to the live run, yet a replay at `known_at` sees it. `odds_analytics.candidate_parity.check_parity` tries a replay at one microsecond before the `created_at` of each distinct snapshot stored in the `odds_analytics.candidate_parity.STORAGE_LAG_WINDOW` (5 minutes) before `known_at` and stamped at or before the decision time, newest first. The first that reproduces the live bets makes the decision `storage_lag`, with that `known_at` reported. A difference with no such snapshot is a mismatch, whatever its cause.
