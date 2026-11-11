---
type: howto
title: Take an idea to a verdict
description: The steps from a candidate idea issue through a development backtest, the one-shot holdout run, registration, the forward test and its checkpoints and the parity check, to promoting or killing the candidate.
sources:
  - odds_cli.commands.candidates
  - odds_analytics.holdout
  - odds_analytics.holdout_run
  - scripts.check_candidate_versions.check
status: stable
---

# Take an idea to a verdict

What a candidate is and when it is promoted or killed is on [Goal and strategy](../foundations/goal-and-strategy.md#how-a-candidate-is-judged). This page is the order of work for a mechanical rule; an agent candidate has no backtest step.

1. **Open an idea issue** labelled `candidate`.
   - State the rule, the reason it might earn EV and the pre-registered grid: the parameter values, the data range and how a cell is selected. Fix these before any backtest runs, as [#626](https://github.com/Acusick1/odds/issues/626) does.
   - A rule that is not in `odds_lambda.candidate_rules.rules.RULES` needs a rule function first ([candidate rules](candidate-rules.md#rule-interface)); a rule reads only the [market state](candidate-rules.md#market-state), whose sharp inputs (the Betfair Exchange, Polymarket, priced by `odds_core.book_price.polymarket_price`) and any [fixture context](candidate-rules.md#fixture-context) (each team's previous result, the concurrent kick-offs) it declares. A [model rule](candidate-rules.md#model-rules) can only be backtested, because the live runner does not run one; its draft's params also fix the tuning trials, seed and folds, its result records the library versions, and `--compare` takes a rule without a model.
2. **Backtest a draft on development data.**
   - Write a draft file outside `candidates/` with every required field of `odds_core.candidates.MechanicalCandidate` except its hash: name, description, kind (`mechanical`), sport, market (one of its sport's in `odds_lambda.candidate_rules.rules.SPORT_MARKETS`, such as `1x2` or `totals` of a football league or NBA `h2h`; one without a close in `odds_core.close.CLOSES`, such as MLB's, gets no EV verdict), version, issue, start date, status, rule and the rule's params, which include `minutes_before_kickoff`, the list of decision times, and `max_input_age_minutes` (10 for a version to register; a draft may widen it to reach older snapshots, [rule interface](candidate-rules.md#rule-interface)). A candidate with several decision times bets each selection of a match (an outcome on a line) at most once, at the first time it qualifies, and its verdict covers the set of times ([once per selection](candidate-rules.md#once-per-selection)).
   - For example:

     ```yaml
     name: back-longest
     description: Back the longest retail price when it beats the exchange fair price by 2%.
     kind: mechanical
     market: 1x2
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

   - Run `uv run odds candidates backtest <path> --from 2024-08-01 --to 2026-07-31`, with `--json <file>` to keep the result, its summary carrying the draft's registry status ([replay backtest](candidate-backtest.md)).
   - Read the coverage first: decisions (one per event and decision time) with every input the rule needs (a UK retail snapshot, with Betfair Exchange prices together for an exchange rule, and a Polymarket snapshot for a Polymarket rule) within the maximum input age, with an exchange price, a Polymarket price and a retail quote at the decision time, and bets with a close. Such snapshots start with live OddsPortal collection in March 2026. Then read k, the family's versions with a recorded decision plus this draft, and the checkpoint and overall estimates, whose promote interval widens as k grows.
   - Development data can kill an idea and never promotes one: the output says `meets_promote_bar`, not `promote`, and the range has been examined many times.
3. **Run the holdout once.** Development screening ends with one selected cell, chosen by the rule fixed in step 1; only that cell runs on the sealed holdout ([sealed holdout](candidate-backtest.md#sealed-holdout)).
   - Before the Odds API backfill reaches the database, commit `candidates/preregistrations/<issue>-<market>-v<version>.yaml` in one commit: the issue, name, version, rule, market, league set, the full grid (or listed cells, when parameters vary together) and the selection rule's link, with no cell.
   - After screening, commit `candidates/selections/<issue>-<market>-v<version>.yaml` in one commit: the grid file's sha256 and the selected cell, a member of the grid or the listed cells.
   - Run `uv run odds candidates holdout candidates/selections/<issue>-<market>-v<version>.yaml` from a clean checkout. It reserves the version in `candidates/holdout.jsonl`, replays each league's sealed window, pools the bets and prints the gate; commit the ledger.
   - The gate passes at mean EV above 0 and not killed. Under 30 bets, not killed alone passes: flag it on the idea issue. A second run of the idea and version is refused, a failure after the reservation, a crash or power loss included, spends the run, and a holdout result never promotes.
   - An idea that fails the gate is not registered. An idea with no holdout data (EPL-only while [#730](https://github.com/Acusick1/odds/issues/730) is open) goes straight to the forward test.
4. **Register it** with a pull request that adds `candidates/<name>.yaml`.
   - Take the hash from `uv run odds candidates rule-hash <rule>`, and set `version: 1`, the issue, a `start_date` and `status: active` ([file format](candidate-registry.md#file-format)).
   - `uv run odds candidates list` validates the file, and the `candidate-versions` hook checks that a later change carries a new version once the start date has arrived ([version check](candidate-registry.md#version-check)). Any code edit in a module the rule imports shows up in the file as a new hash, so it needs a new version, unless the edit leaves every bet the same (step 8).
5. **Deploy.** The [run-candidates job](run-candidates-job.md) places the candidate's paper trades from its start date and records every decision. Nothing else is needed.
6. **Read the report on the production host.** `odds copy from-prod` copies no trades or decisions, so a local database cannot show them ([report](candidate-registry.md#odds-candidates-report)).
   - `ssh` to the production host, `cd ~/odds`, then run `uv run odds candidates report`, or open the candidate's page in the [web UI](../operations/odds-web.md#candidate-page), which reads the same report.
   - The verdict is read only at 100, 200, … bets with a close. Below 100 it is `not_yet`; do not act on a running mean.
7. **Run the parity check on the production host**: `ssh` to it, `cd ~/odds`, then `uv run odds candidates parity <name>` ([parity check](candidate-parity.md)).
   - It replays each decision from the snapshots it recorded. A mismatch means the rule or state code does not reproduce the live bets, which is a bug to fix before trusting either result; it exits 1.
   - A data difference (the database now yields other bets than the recorded inputs did) is informational: the backtest, which loads from the database, would not reproduce that decision.
   - "No decisions to check" means the candidate has not decided yet.
8. **Rehash after a code change that leaves its bets alone.** When a pull request edits a module the rule imports, `odds candidates list` fails on the stale hash. If the change should leave every bet the same, commit it, then run on the production host from the branch's checkout `uv run odds candidates rehash <rule> --database-url "$WEB_DATABASE_URL" --from 2015-08-01 --to <yesterday> --evidence <PR link>` ([rehash ledger](candidate-registry.md#rehash-ledger)).
   - It replays every candidate of the rule under the base revision's code and the branch's on one read-only snapshot of production, and checks parity.
   - It writes the new hashes and the ledger row only if the candidates record the base's hash, the replays are byte-identical with decisions and bets, and parity holds. Commit both; the `candidate-versions` hook accepts the hash change at the same version.
   - If it refuses, the change alters bets: give each candidate a new version instead.
9. **Promote or kill** by changing the status in a pull request, which needs no version bump.
   - Promote when the report says `promote` (each checkpoint is judged at the k of its nth bet, the family's versions that have decided, so a later variant never reverses it), then the stake-capacity gate before real money (not yet defined).
   - Kill when it says `kill`, or by choice. Record the result on the idea issue and close it. Never delete the file.
