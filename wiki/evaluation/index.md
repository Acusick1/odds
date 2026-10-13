---
title: Backtests and model research
description: The backtest engine and strategies, what the CLV model has been shown to do, and the experiment record.
---

# Backtests and model research

Backtests replay a strategy over stored events; experiments record what the CLV model was and was not shown to do. Start with [CLV model findings](clv-model-findings.md) for the current conclusions, then the experiment pages for the evidence. The experiment pages are the evidence for the modelling ADRs ([0002](../../docs/adr/0002-bet365-clv-target.md), [0003](../../docs/adr/0003-hybrid-sharp-reference.md), [0004](../../docs/adr/0004-walk-forward-cv-protocol.md)).

<!-- pages:start -->
### Reference

- [Backtest engine](backtest-engine.md) — BacktestEngine event loop, sizing, settlement, metrics as computed, BacktestResult and its JSON and CSV forms.
- [Backtest strategies](backtest-strategies.md) — The registered strategies, their constructor defaults and the STRATEGIES registry.

### Explanation

- [Betting simulation (Exp 7, 7b)](exp-betting-simulation.md) — Whether the CLV signal survives the vig with flat stakes and with line shopping.
- [CV protocol grid](exp-cv-protocol-grid.md) — The window-type by validation-step grid behind ADR-0004, the 32-feature rerun, and what the grid can and cannot show.
- [Early NBA signal search and LSTM](exp-early-nba-signal-and-lstm.md) — February 2026 NBA runs (XGBoost v1, Exp 1, Exp 2, LSTM variants): no signal at about 230-1K events, sharp-retail divergence as the best correlate, the IN_PLAY leak, and why LSTM was ruled out.
- [EPL feature ablation and principled search space](exp-epl-feature-ablation-and-search-space.md) — The April 2026 EPL runs: per-group and FPL ablations (match_stats helps, schedule and FPL hurt), the sliding-window collapse to constant predictors, and the revised Optuna space that produced the deployed model.
- [EPL sharp-reference experiments](exp-epl-sharp-reference.md) — The March 2026 EPL runs covering Pinnacle as sharp reference, the
- [NBA timing and injury experiments](exp-nba-timing-and-injuries.md) — When NBA books post lines, how CLV and divergence vary with hours to the game (Exp 4), and why injury features add nothing once tuned equally (Exp 6, 6b).
- [Target bookmaker and data volume experiments](exp-target-and-data-volume.md) — The NBA bet365-versus-Pinnacle runs and the Exp 6 learning curve behind ADR-0002.
- [What the CLV model can and cannot do](clv-model-findings.md) — Hub for the CLV model, with the deployed model, current conclusions linked to ADRs and experiments, the data used, and a dated index.

### How-to guides

- [Add a backtest strategy](howto-add-backtest-strategy.md) — Subclass BettingStrategy, implement evaluate_opportunity, register it, and test it.
- [Run a backtest](howto-run-backtest.md) — Check data prerequisites, list strategies, run, show, compare and export, or run from code.
- [Run and record a modelling experiment](howto-run-experiment.md) — Script conventions, the required FINDINGS.md and verification artifacts with their checks, and how to record the result in the wiki.
<!-- pages:end -->
