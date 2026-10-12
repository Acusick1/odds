---
type: component
title: Backtest strategies
summary: The flat, basic EV, arbitrage and model-driven strategies, their defaults, and the STRATEGIES registry.
sources:
  - packages/odds-analytics/odds_analytics/strategies.py::FlatBettingStrategy
  - packages/odds-analytics/odds_analytics/strategies.py::BasicEVStrategy
  - packages/odds-analytics/odds_analytics/strategies.py::ArbitrageStrategy
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-cli/odds_cli/commands/backtest.py::list_strategies
  - docs/BACKTESTING_GUIDE.md#available-strategies
  - docs/CLI.md#list-strategies
verified_at: 7887039
---

# Backtest strategies

Strategies subclass `BettingStrategy` and return `BetOpportunity` rows from `evaluate_opportunity` ([backtest-engine](backtest-engine.md)). The CLI resolves `--strategy` through `packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES`; defaults below are constructor defaults in `packages/odds-analytics/odds_analytics/strategies.py`. To add one see [howto-add-betting-strategy](howto-add-betting-strategy.md).

| Name | Class | Defaults |
|---|---|---|
| `flat` | `FlatBettingStrategy` | `market` h2h, `outcome_pattern` home (or away, draw, favorite), `bookmaker` fanduel; confidence 0.5 |
| `basic_ev` | `BasicEVStrategy` | `sharp_book` pinnacle, `retail_books` fanduel/draftkings/betmgm, `min_ev_threshold` 0.03, `markets` h2h/spreads/totals |
| `arbitrage` | `ArbitrageStrategy` | `min_profit_margin` 0.01, `max_hold` 0.10, six US books incl. pinnacle |
| `xgb_line_movement`, `lstm_line_movement` | model strategies | need `--model-path`; see [line-movement-models](line-movement-models.md) |

- **flat** bets one outcome per event at one bookmaker; it is the baseline.
- **basic_ev** treats the sharp book's implied probability as the true probability, computes EV at each retail book via `calculate_ev`, and bets where EV is at least the threshold. `confidence` is that probability.
- **arbitrage** takes the best price per outcome across its books, skips markets whose hold exceeds `max_hold`, and bets all legs when `detect_arbitrage` finds a gap of at least `min_profit_margin`. It covers h2h, 1x2, spreads and totals.
- `odds backtest list-strategies` instantiates every registered class with no arguments and prints its docstring and parameters; every strategy module is imported when the command loads, including the torch-based LSTM one.

## Hazards

- **Defaults are US/NBA-flavoured.** Rosters are US books, `flat` uses `h2h` and the default `basic_ev` markets omit `1x2`, so on EPL data (`1x2`, UK books) they match nothing unless overridden; the CLI cannot override them ([sport-and-market-keys](sport-and-market-keys.md), [project-pivots](project-pivots.md); AUD T2/A7, re-verified).
- **`basic_ev` uses vig-inclusive sharp probability** as truth ([devigging](devigging.md)), and its `confidence` feeds Kelly sizing as a win probability.
- **Arbitrage is not hedged.** The engine sizes each leg independently by `confidence` 1.0; the stake split from `detect_arbitrage` is not used, and legs settle separately, so a run can show losses.
- **`--help` mentions `lstm`**, which is not a registry key; the valid names are those in the table.

## Related

- [Backtest metrics](backtest-metrics.md), [Howto: run a backtest](howto-run-backtest.md)
