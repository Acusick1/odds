---
title: Backtests and model research
description: The backtest engine and strategies, what the CLV model has been shown to do, and the experiment record.
---

# Backtests and model research

Backtests replay a strategy over stored events; experiments record what the CLV model was and was not shown to do. Start with [CLV model findings](clv-model-findings.md) for the current conclusions, then the experiment pages for the evidence.

<!-- pages:start -->
### Concepts

- [What the CLV model can and cannot do](clv-model-findings.md) — Hub for the CLV model: the deployed model, current conclusions with links to each decision and experiment, the data used, and the betting simulation showing the signal does not survive the vig.

### Components

- [Backtest engine, metrics and running a backtest](backtesting.md) — How BacktestEngine replays a strategy (odds at decision time, sizing, settlement), how each metric is computed and saved, how to run, compare and export backtests, and the data and sizing traps that invalidate results.
- [Backtest strategies and adding one](backtest-strategies.md) — The flat, basic EV, arbitrage and model strategies, their US/NBA-flavoured defaults and the STRATEGIES registry, and how to add and register a new strategy.

### Experiments

- [Early NBA signal search and LSTM](exp-early-nba-signal-and-lstm.md) — February 2026 NBA runs (XGBoost v1, Exp 1, Exp 2, LSTM variants): no signal at about 230-1K events, sharp-retail divergence as the best correlate, the IN_PLAY leak, and why LSTM was ruled out.
- [EPL feature ablation and principled search space](exp-epl-feature-ablation-and-search-space.md) — The April 2026 EPL runs: per-group and FPL ablations (match_stats helps, schedule and FPL hurt), the sliding-window collapse to constant predictors, and the revised Optuna space that produced the deployed model.
- [NBA timing and injury experiments](exp-nba-timing-and-injuries.md) — When NBA books post lines, how CLV and divergence vary with hours to the game (Exp 4), and why injury features add nothing once tuned equally (Exp 6, 6b).

### How-tos

- [Run and record a modelling experiment](howto-run-experiment.md) — Script conventions, the required FINDINGS.md and verification artifacts with their checks, and how to record the result in the wiki.
<!-- pages:end -->
