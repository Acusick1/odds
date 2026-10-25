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
   - Write a draft file outside `candidates/` with every required field of `odds_core.candidates.MechanicalCandidate` except its hash: name, description, kind (`mechanical`), sport, version, issue, start date, status, rule and the rule's params, which include `minutes_before_kickoff`, the list of decision times, and `max_input_age_minutes` (10 for a version to register; a draft may widen it to reach older snapshots, [rule interface](candidate-rules.md#rule-interface)). A candidate with several decision times bets each outcome of a match at most once, at the first time it qualifies, and its verdict covers the set of times ([once per outcome](candidate-rules.md#once-per-outcome)).
   - For example:

     ```yaml
     name: back-longest
     description: Back the longest retail price when it beats the exchange fair price by 2%.
     kind: mechanical
     sport: soccer_epl
     version: 1
     issue: 626
     start_date: 2026-11-01
     status: active
     rule: exchange_gap
     params:
       threshold: 0.02
       minutes_before_kickoff: [60]
       max_input_age_minutes: 10
     ```

   - Run `uv run odds candidates backtest <path> --from 2024-08-01 --to 2026-07-31`, with `--json <file>` to keep the result ([replay backtest](candidate-backtest.md)).
   - Read the coverage first: decisions (one per event and decision time) with an input snapshot (UK retail and Betfair Exchange prices together) within the maximum input age, with an exchange price and a retail quote at the decision time, and bets with a close. Such snapshots start with live OddsPortal collection in March 2026. Then read k, the family's versions with a recorded decision plus this draft, and the checkpoint and overall estimates, whose promote interval widens as k grows.
   - Development data can kill an idea and never promotes one: the output says `meets_promote_bar`, not `promote`, and the range has been examined many times.
3. **Register it** with a pull request that adds `candidates/<name>.yaml`.
   - Take the hash from `uv run odds candidates rule-hash <rule>`, and set `version: 1`, the issue, a `start_date` and `status: active` ([file format](candidate-registry.md#file-format)).
   - `uv run odds candidates list` validates the file, and the `candidate-versions` hook checks that a later change carries a new version once the start date has arrived ([version check](candidate-registry.md#version-check)). A rule's code change reaches the file as a new hash, so it needs a new version.
4. **Deploy.** The [run-candidates job](run-candidates-job.md) places the candidate's paper trades from its start date and records every decision. Nothing else is needed.
5. **Read the report on the production host.** `odds copy from-prod` copies no trades or decisions, so a local database cannot show them ([report](candidate-registry.md#odds-candidates-report)).
   - `ssh` to the production host, `cd ~/odds`, then run `uv run odds candidates report`.
   - The verdict is read only at 100, 200, … bets with a close. Below 100 it is `not_yet`; do not act on a running mean.
6. **Run the parity check on the production host**: `ssh` to it, `cd ~/odds`, then `uv run odds candidates parity <name>` ([parity check](candidate-parity.md)).
   - It replays each decision from the snapshots it recorded. A mismatch means the rule or state code does not reproduce the live bets, which is a bug to fix before trusting either result; it exits 1.
   - A data difference (the database now yields other bets than the recorded inputs did) is informational: the backtest, which loads from the database, would not reproduce that decision.
   - "No decisions to check" means the candidate has not decided yet.
7. **Promote or kill** by changing the status in a pull request, which needs no version bump.
   - Promote when the report says `promote` (each checkpoint is judged at the k of its nth bet, the family's versions that have decided, so a later variant never reverses it), then the stake-capacity gate before real money (not yet defined).
   - Kill when it says `kill`, or by choice. Record the result on the idea issue and close it. Never delete the file.
