---
title: Backtests and model research
description: The backtest engine and strategies, what the CLV model has been shown to do, and the experiment record.
---

# Backtests and model research

Backtests replay a strategy over stored events; experiments record what the CLV model was and was not shown to do. Start with [CLV model findings](clv-model-findings.md) for the current conclusions, then the experiment pages for the evidence. The experiment pages are the evidence for the modelling ADRs ([0002](../../docs/adr/0002-bet365-clv-target.md), [0003](../../docs/adr/0003-hybrid-sharp-reference.md), [0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md), [0011](../../docs/adr/0011-bet365-clv-target-on-like-for-like-evidence.md) and [0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)).

<!-- pages:start -->
### Reference

- [Backtest engine](backtest-engine.md) — BacktestEngine event loop, sizing, settlement, metrics as computed, BacktestResult and its JSON and CSV forms.
- [Backtest strategies](backtest-strategies.md) — The registered strategies, their constructor defaults and the STRATEGIES registry.
- [Experiment scripts and artifacts](experiment-scripts.md) — Layout of experiments/, config and script conventions, the FINDINGS.md sections, the verification artifact files, and the inventory of scripts.
- [Retail-EV evaluation](retail-ev-evaluation.md) — The one-look evaluation of a frozen retail-EV model on a window of events, with its spec, the window-end guard, the refit blocks, the keep-bar statistics and verdict, the coverage report, the descriptive slices and the output files.

### Explanation

- [As-of-t feature groups on the retail-EV target](exp-epl-retail-ev-as-of-t-groups.md) — The October 2026 exploratory rerun of the price-only retail-EV protocol with the Polymarket, dynamics, schedule and standings groups added one at a time and together, against the price-only model, on live-collected and FDUK-only events, per fold and after a Bonferroni correction, with the timing leak an earlier version of the Polymarket group had on FDUK opening rows.
- [Betting simulation (Exp 7, 7b)](exp-betting-simulation.md) — Whether the CLV signal survives the vig with flat stakes and with line shopping, and how to read backtest-engine metrics against it.
- [CV protocol grid](exp-cv-protocol-grid.md) — The window-type by validation-step grid behind ADR-0004, the 32-feature rerun, and what the grid can and cannot show.
- [CV protocol on a common validation range](exp-cv-protocol-common-range.md) — The October 2026 comparison of expanding, sliding-380 and sliding-760 walk-forward CV scored on the same validation events, the tuning-range event counts, and the fold layout the re-baseline uses.
- [Early NBA signal search and LSTM](exp-early-nba-signal-and-lstm.md) — February 2026 NBA runs (XGBoost v1, Exp 1, Exp 2, LSTM variants): no signal at about 230-1K events, sharp-retail divergence as the best correlate, the IN_PLAY leak, and why LSTM was ruled out.
- [EPL CLV re-baseline against the simple baseline](exp-epl-clv-rebaseline.md) — The October 2026 out-of-sample test of the tabular EPL bet365 CLV model against a one-feature OLS baseline over two seeds on the evaluation range, with its two runs, the decision-row rule, event counts, the pre-registered keep test and its result.
- [EPL feature ablation and principled search space](exp-epl-feature-ablation-and-search-space.md) — The April 2026 EPL runs: per-group and FPL ablations (match_stats helps, schedule and FPL hurt), the sliding-window collapse to constant predictors, and the revised Optuna space that produced the deployed model.
- [EPL sharp-reference experiments](exp-epl-sharp-reference.md) — The March 2026 EPL runs covering Pinnacle as sharp reference, the
- [NBA timing and injury experiments](exp-nba-timing-and-injuries.md) — When NBA books post lines, how CLV and divergence vary with hours to the game (Exp 4), and why injury features add nothing once tuned equally (Exp 6, 6b).
- [Nested tuning on the as-of-t columns](exp-epl-retail-ev-nested-tuning.md) — The October 2026 exploratory rerun of the residual retail-EV models with hyperparameters re-tuned for every scoring fold by nested walk-forward inside 2025-26, so tuning sees the Polymarket and live-collected rows; price-only, Polymarket at t and all four as-of-t groups against the baseline, price-only and the frozen 2024-25 settings, on live-collected and FDUK-only events, per fold, after a Bonferroni correction, and where the tuned settings sit.
- [Pinnacle against bet365 as the EPL CLV target](exp-epl-clv-pinnacle-retest.md) — The October 2026 test of the devigged Pinnacle CLV target against bet365 on the same EPL events, folds and features, with the event counts, the pre-registered retarget test and its result, and the skill comparison behind the target ADR.
- [Power of the retail-EV keep bar](exp-retail-ev-power.md) — The October 2026 simulation of how often ADR-0012's two keep-bar tests pass a synthetic model on a fresh season of 100 to 1,140 events, under FDUK-only rows and dense live collection, for a diffuse calibrated model and a sparse information-edge model, with the minimum detectable effects, the size check and the assumptions behind them.
- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) — The October 2026 exploratory run, and its rerun with the pipeline residual arm and a fixed loss-weight scale, of price-only XGBoost on the ADR-0012 retail-EV target against the martingale baseline and a constant shrinkage of it, on 2024-25 and 2025-26 development data, with the tuning and scoring layout, per-fold results, the keep-bar metrics by horizon and retail source, how the gain spreads over events, and feature importances.
- [Target bookmaker and data volume experiments](exp-target-and-data-volume.md) — The NBA bet365-versus-Pinnacle runs and the Exp 6 learning curve behind ADR-0002.
- [What the CLV model can and cannot do](clv-model-findings.md) — Hub for the CLV model, with the deployed model, current conclusions linked to ADRs and experiments, the data used, and a dated index.

### How-to guides

- [Add a backtest strategy](howto-add-backtest-strategy.md) — Subclass BettingStrategy, implement evaluate_opportunity, register it, and test it.
- [Run a backtest](howto-run-backtest.md) — Check data prerequisites, list strategies, run, show, compare and export, or run from code.
- [Run and record a modelling experiment](howto-run-experiment.md) — Run an experiment script, check its verification artifacts and FINDINGS.md, and record the result in the wiki.
- [Run the retail-EV evaluation](howto-run-retail-ev-evaluation.md) — Write the evaluation spec, run odds evaluate retail-ev once from a clean tree on a local copy of the data, and read the verdict, manifest and coverage.
<!-- pages:end -->
