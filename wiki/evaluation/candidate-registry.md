---
type: reference
title: Candidate registry
description: The candidate file format, the odds candidates list, rule-hash and report commands, which commands replay a candidate, how paper trades are scored and judged per candidate, and the version check that stops a candidate changing without a new version.
sources:
  - odds_core.candidates
  - odds_cli.commands.candidates
  - odds_analytics.candidate_report
  - odds_core.betfair_close.event_close
  - odds_core.paper_trade_models.PaperTrade
  - odds_mcp.server.get_portfolio
  - scripts/check_candidate_versions.py
  - odds_lambda.candidate_rules.validate_candidate_rules
status: stable
---

# Candidate registry

The registry is the `candidates/` directory: one `<name>.yaml` file per candidate, and the only definition of it. No database table copies it. What a candidate is and how it is judged is on [Goal and strategy](../foundations/goal-and-strategy.md#candidates).

## File format

`odds_core.candidates.load_candidates` reads every `*.yaml` file in a directory with `odds_core.candidates.load_candidate_file`, validates it against `odds_core.candidates.Candidate` and returns the candidates sorted by name. A missing directory gives an empty list. Every model rejects unknown fields.

| Field | Model | Meaning |
|---|---|---|
| `name`, `description`, `sport`, `version`, `issue`, `start_date`, status | `odds_core.candidates._CandidateBase` | Shared by both kinds. `name` matches the file name and the pattern `odds_core.candidates.NAME_PATTERN`. `sport` is a [sport key](../collection/sport-and-market-keys.md). `issue` is the idea's GitHub issue. `start_date` is the first day it may place bets. The status is an `odds_core.candidates.CandidateStatus` |
| `kind: mechanical`, `rule`, `odds_core.candidates.MechanicalCandidate.rule_sha256` and params | `odds_core.candidates.MechanicalCandidate` | `rule` names a rule function ([candidate rules](candidate-rules.md)). `odds_core.candidates.MechanicalCandidate.rule_sha256` is the hash of the rule's code ([rule pinning](candidate-rules.md#rule-pinning)), 64 lowercase hex characters, and required in the registry; a [draft](candidate-backtest.md#command) outside it may omit it. The parameters are a free JSON mapping here, validated against the rule's params model by `odds_lambda.candidate_rules.validate_candidate_rules` |
| `kind: agent` | `odds_core.candidates.AgentCandidate` | No further fields; the agent's prompts and model are its definition ([agent run](../agent/agent-run-job.md)) |

`odds_core.candidates.load_candidates` raises a `ValueError` naming the file when a file fails validation, its `name` differs from its file stem or a mechanical file lacks `odds_core.candidates.MechanicalCandidate.rule_sha256`, and when a sport has more than one agent candidate whose status is active or promoted. Agents for one sport share `packages/odds-mcp/agents/common.md` and the sport's `base.md`, so two live agents for one sport would run identical prompts. A killed agent does not count, so its replacement can be added.

A change of status goes through a pull request. The version is a strict positive integer (`"3"` is rejected), bumped whenever the rule, parameters or model change ([versioning](../foundations/goal-and-strategy.md#candidates)).

## `odds candidates list`

`odds_cli.commands.candidates.list_candidates` prints each candidate's name, kind, sport, version, status, start date and issue, optionally filtered by status. It reads the files in `odds_core.candidates.candidates_dir` and does not touch the database. It validates every mechanical candidate's rule and params with `odds_lambda.candidate_rules.validate_candidate_rules`. An invalid registry or rule prints the error in red and exits 1.

## `odds candidates rule-hash`

`odds_cli.commands.candidates.rule_hash` prints `odds_lambda.candidate_rules.rules.rule_sha256` for a rule name, the value a mechanical candidate records as `odds_core.candidates.MechanicalCandidate.rule_sha256`. An unknown rule name prints the known names and exits 1.

## `odds candidates report`

`odds_cli.commands.candidates.report` prints one row per candidate version: settled bets, bets with a close, bets without one, pending bets, stake, P&L, ROI and the family size k, the checkpoint the verdict rests on (n, mean EV, the 95% CI, the promote CI with its confidence level) and the verdict. It reads the database through `odds_analytics.candidate_report.candidate_report` and the registry files. Every registered candidate appears, with zero bets before it trades, and so does any (name, version) with trades whose version differs from the registry's (or whose candidate is absent from it), shown as "not current".

`odds_mcp.server.get_portfolio` returns the same summaries as `candidates`, from the same function. The trades behind them come from the agent and from the [run-candidates job](run-candidates-job.md). When the registry is invalid it returns `{"error": <message>}` there and the rest of the portfolio unchanged.

What is scored, in `odds_analytics.candidate_report`:

- **Bets.** `odds_analytics.candidate_report.ScoredBet` is what is scored, built from an `odds_core.paper_trade_models.PaperTrade` or from a [replayed](candidate-backtest.md) bet. `odds_analytics.candidate_report.load_scored_trades` takes every paper trade that carries a candidate, settled or not, in placement order (`placed_at`, then id), and drops void ones ([paper trading](../agent/paper-trading.md#candidate-fields)). Pushes count as settled bets.
- **Close.** `p_close` is the devigged probability of the trade's selection in `odds_core.betfair_close.event_close`, computed from the event's stored snapshots when the report runs, so nothing is recorded at settlement and the close can be re-derived. `odds_analytics.candidate_report.close_candidate_snapshots` loads only the snapshots that can be the close: those within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before kick-off, and FDUK closing snapshots. Only the `1x2` market has a close (`odds_analytics.candidate_report.selection_close_probability`): an `h2h` trade, which is every MLB trade, has none, and so does a `1x2` trade whose event has no close.
- **EV.** `odds_analytics.candidate_report.ScoredBet.ev` is `p_close × price_decimal − 1`.
- **Scoreable prefix.** `odds_analytics.candidate_report.scoreable_prefix` walks a version's trades in placement order and stops at the first trade that is unsettled, or has no close and kicked off less than `odds_analytics.candidate_report.CLOSE_GRACE` (7 days) before the report's `now`, since the FDUK closing price may still arrive. A trade with no close after that is final: it counts in P&L and in the no-close count, and is left out of the EV. Trades beyond the prefix are the pending count. The prefix only grows, and the bets with a close in it keep their order, so a checkpoint's bets never change once reached.
- **Checkpoint.** `odds_analytics.candidate_report.checkpoint_at` scores the first n bets with a close, the earliest, for n a multiple of `odds_analytics.candidate_report.CHECKPOINT_SIZE` (100). The score is the stake-weighted mean EV per unit staked, `Σ stake·EV / Σ stake`, with `odds_analytics.retail_ev_metrics.event_clustered_ratio` giving percentile CIs at the 95% kill level and the promote level (below) from `odds_analytics.candidate_report.DEFAULT_BOOTSTRAP_SAMPLES` resamples of whole events under a fixed seed, so the same trades always give the same verdict.
- **Family size k.** `odds_analytics.candidate_report.family_starts` lists when each version of the candidate's family, the mechanical candidates with its `rule` and `sport`, started, and `odds_analytics.candidate_report.family_size_at` counts those started by a time (at least 1). A registry entry holds its candidate's current version and starts at midnight UTC of its `start_date`, whatever its status. Earlier versions are not in the registry (a version bump overwrites the file), so each starts at its first `odds_core.paper_trade_models.CandidateDecision.decision_time`, the minimum per name and version that `odds_analytics.candidate_report.load_decision_starts` reads, for names whose registry entry is in the family. A version with no decision that is not current never started, which is the [definition of a started version](../foundations/goal-and-strategy.md#how-a-candidate-is-judged). Every version of a name takes the family of its registry entry; a name absent from the registry, and an agent candidate, has k = 1. A checkpoint's k is the size at the placement of its nth bet (`odds_analytics.candidate_report.ScoredBet.placed_at`), so later versions do not change an earlier checkpoint. `odds_analytics.candidate_report.CandidateSummary.family_versions` is the verdict checkpoint's k, or the size now before any checkpoint, and `odds_mcp.server.get_portfolio` returns it.
- **Verdict.** `odds_analytics.candidate_report.judge` evaluates checkpoints 100, 200, … in order, each at the k of its nth bet, and returns the first `odds_analytics.candidate_report.Verdict` that fires, with the checkpoint's n: `promote` when the lower bound of the promote CI is above zero, at confidence `odds_analytics.candidate_report.promote_level` (1 − `odds_analytics.candidate_report.PROMOTE_ERROR` / k, with `odds_analytics.candidate_report.promote_level`); `kill` when the 95% CI's upper bound is at or below zero, or at the `odds_analytics.candidate_report.KILL_AT_BETS` (500) checkpoint without promotion. Later bets do not reverse it. With neither it returns `continue` at the last checkpoint reached, and `not_yet` below 100 bets with a close. The rules and their rationale are on [Goal and strategy](../foundations/goal-and-strategy.md#how-a-candidate-is-judged). The verdict is advice: a status change is still a pull request.

## Replay commands

`odds_cli.commands.candidates.backtest` replays a rule over completed events and scores it with the functions above ([replay backtest](candidate-backtest.md)). `odds_cli.commands.candidates.parity` replays each recorded decision from its recorded input snapshots and compares the bets ([parity check](candidate-parity.md)). The steps that use them are on [From idea to verdict](howto-take-an-idea-to-a-verdict.md).

## Version check

`scripts/check_candidate_versions.py` compares the working tree's `candidates/*.yaml` with the same files at a base commit. It runs as the `candidate-versions` pre-commit hook ([code conventions](../foundations/code-conventions.md#mechanical-gates)), locally and in the `Lint` workflow. The base is `--base`, else the merge-base of `HEAD` with `origin/main`, else with `main`, else `HEAD`.

It fails when:

- a file that exists at the base is deleted. A candidate is killed by setting its status to killed. A rename reads as a deletion, so it fails: a renamed candidate is a new candidate;
- a file is not valid YAML (the error names the file);
- a file's `version` decreased;
- any key other than the status and `description` changed and the version did not increase (`UNVERSIONED_KEYS` in the script).

A new file always passes. The check compares YAML only. A mechanical rule's code is pinned by the `odds_core.candidates.MechanicalCandidate.rule_sha256` in the file ([rule pinning](candidate-rules.md#rule-pinning)), so a code change reaches this check as a change to that key. Changes to an agent's prompts or model are not enforced by it.
