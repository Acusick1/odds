---
title: Backtests and model research
description: The backtest engine and strategies, the candidate registry, the retail-EV evaluation and the retail-EV experiment record.
---

# Backtests and model research

Backtests replay a strategy over stored events. The experiment pages record what the retail-EV model was and was not shown to do, against the target of [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md).

<!-- pages:start -->
### Reference

- [Candidate registry](candidate-registry.md) — The candidate file format, the odds candidates list, rule-hash, report and rehash commands, which commands replay a candidate, how paper trades are scored and judged per candidate, and the version check that stops a candidate changing without a new version.
- [Candidate rules](candidate-rules.md) — The market state and fixture context a mechanical rule reads at decision time, the bets it returns, the rule registry and module layout, decision times and the once-per-selection rule, maximum input age, params validation, rule pinning, the one-bet helper, the exchange-gap and Polymarket-gap rules, the exchange-steam rule, the retail-consensus rules (retail-quorum lag, subsidised outcome and single-book outlier) and the retail-EV model rule.
- [Cell selection](cell-selection.md) — What odds candidates select replays for a pre-registered grid, the development blocks, the eligibility rule, the bootstrap lower bound and tie-breaks that pick a cell, the power-devig score it reports for information, the selection file it writes and its JSON output.
- [Experiment scripts and artifacts](experiment-scripts.md) — Layout of experiments/, config and script conventions, the verification artifact files, and the inventory of scripts.
- [Parity check](candidate-parity.md) — What odds candidates parity replays, how it rebuilds each decision from its recorded input snapshots and events, how it reports data differences from the current database, and its exit code.
- [Replay backtest](candidate-backtest.md) — What odds candidates backtest replays, the decision state and timing it uses, how a model rule is tuned and refitted during the replay, the comparison candidate, how bets are scored, the coverage it reports and its JSON output, each placed bet included.
- [Substitute-close comparison](substitute-close-comparison.md) — What the substitute-close comparison script compares (the Odds API's Betfair Exchange price near kick-off against FDUK's Betfair Exchange close), the pass thresholds, the report it prints and writes, and how its prices are chosen.
- [The run-candidates job](run-candidates-job.md) — How the scheduled run-candidates job takes each active mechanical candidate's decisions at their decision times, what it records, when it records a decision as missed, and how it books its next run.

### Explanation

- [As-of-t feature groups on the retail-EV target](exp-epl-retail-ev-as-of-t-groups.md) — The October 2026 exploratory rerun of the price-only retail-EV protocol with the Polymarket, dynamics, schedule and standings groups added one at a time and together, against the price-only model, on live-collected and FDUK-only events, per fold and after a Bonferroni correction, with the timing leak an earlier version of the Polymarket group had on FDUK opening rows.
- [Nested tuning on the as-of-t columns](exp-epl-retail-ev-nested-tuning.md) — The October 2026 exploratory rerun of the residual retail-EV models with hyperparameters re-tuned for every scoring fold by nested walk-forward inside 2025-26, so tuning sees the Polymarket and live-collected rows; price-only, Polymarket at t and all four as-of-t groups against the baseline, price-only and the frozen 2024-25 settings, on live-collected and FDUK-only events, per fold, after a Bonferroni correction, and where the tuned settings sit.
- [Power of the retail-EV keep bar](exp-retail-ev-power.md) — The October 2026 simulation of how often ADR-0012's two keep-bar tests pass a synthetic model on a fresh season of 100 to 1,140 events, under FDUK-only rows and dense live collection, for a diffuse calibrated model and a sparse information-edge model, with the minimum detectable effects, the size check and the assumptions behind them.
- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) — The October 2026 exploratory run, and its rerun with the pipeline residual arm and a fixed loss-weight scale, of price-only XGBoost on the ADR-0012 retail-EV target against the martingale baseline and a constant shrinkage of it, on 2024-25 and 2025-26 development data, with the tuning and scoring layout, per-fold results, the keep-bar metrics by horizon and retail source, how the gain spreads over events, and feature importances.

### How-to guides

- [Run and record a modelling experiment](howto-run-experiment.md) — Run an experiment script, check its verification artifacts, and record the result in the wiki.
- [Take an idea to a verdict](howto-take-an-idea-to-a-verdict.md) — The steps from a candidate idea issue through a development backtest, the cell selection, the one-shot holdout run, registration, the forward test and its checkpoints and the parity check, to promoting or killing the candidate.
<!-- pages:end -->
