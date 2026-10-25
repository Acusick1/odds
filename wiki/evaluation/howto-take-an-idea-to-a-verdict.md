---
type: howto
title: Take an idea to a verdict
description: The steps from a candidate idea issue through a development backtest, registration, the forward test and its checkpoints and the parity check, to promoting or killing the candidate.
sources:
  - odds_cli.commands.candidates
  - scripts/check_candidate_versions.py
status: stable
---

# Take an idea to a verdict

What a candidate is and when it is promoted or killed is on [Goal and strategy](../foundations/goal-and-strategy.md#how-a-candidate-is-judged). This page is the order of work for a mechanical rule; an agent candidate has no backtest step.

1. **Open an idea issue** labelled `candidate`.
   - State the rule, the reason it might earn EV and the pre-registered grid: the parameter values, the data range and how a cell is selected. Fix these before any backtest runs, as [#626](https://github.com/Acusick1/odds/issues/626) does.
   - A rule that is not in `odds_lambda.candidate_rules.rules.RULES` needs a rule function first ([candidate rules](candidate-rules.md#rule-interface)).
2. **Backtest a draft on development data.**
   - Write a candidate file outside `candidates/`, with the rule, params and sport. `odds_core.candidates.MechanicalCandidate.rule_sha256` may be missing.
   - Run `uv run odds candidates backtest <path> --from 2024-08-01 --to 2026-07-31`, with `--json <file>` to keep the result ([replay backtest](candidate-backtest.md)).
   - Read the coverage first: events with an exchange price and a retail quote at the decision time, and bets with a close. Then read the checkpoint and overall estimates.
   - Development data can kill an idea and never promotes one; the range has been examined many times.
3. **Register it** with a pull request that adds `candidates/<name>.yaml`.
   - Take the hash from `uv run odds candidates rule-hash <rule>`, and set `version: 1`, the issue, a `start_date` and `status: active` ([file format](candidate-registry.md#file-format)).
   - `uv run odds candidates list` validates the file, and the `candidate-versions` hook checks that a later change carries a new version ([version check](candidate-registry.md#version-check)). A rule's code change reaches the file as a new hash, so it needs a new version.
4. **Deploy.** The [run-candidates job](run-candidates-job.md) places the candidate's paper trades from its start date and records every decision. Nothing else is needed.
5. **Read the report.** `uv run odds candidates report` on the production host, or on a [local copy](../foundations/howto-copy-production-data.md) of its data ([report](candidate-registry.md#odds-candidates-report)).
   - The verdict is read only at 100, 200, … bets with a close. Below 100 it is `not_yet`; do not act on a running mean.
6. **Run the parity check** with `uv run odds candidates parity <name>` on the same data ([parity check](candidate-parity.md)).
   - A mismatch means the backtest and the forward test disagree, which is a bug to fix before trusting either. Storage lag is reported and does not fail.
7. **Promote or kill** by changing the status in a pull request, which needs no version bump.
   - Promote when the report says `promote`, then the stake-capacity gate before real money (not yet defined).
   - Kill when it says `kill`, or by choice. Record the result on the idea issue and close it. Never delete the file.
