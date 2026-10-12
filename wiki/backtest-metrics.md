---
type: concept
title: Backtest metrics and results
summary: ROI, win rate, Sharpe, Sortino, drawdown, profit factor and Calmar as the code computes them, and the BacktestResult model with its JSON and CSV forms.
sources:
  - packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestResult
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine
  - packages/odds-analytics/odds_analytics/utils.py::calculate_sharpe_ratio
  - packages/odds-analytics/odds_analytics/utils.py::calculate_sortino_ratio
  - packages/odds-analytics/odds_analytics/utils.py::calculate_max_drawdown
  - packages/odds-analytics/odds_analytics/utils.py::calculate_profit_factor
  - docs/BACKTESTING_GUIDE.md#performance-metrics
  - docs/CLI.md#show-results
  - docs/CLI.md#compare-results
  - docs/CLI.md#export-to-csv
verified_at: 7887039
---

# Backtest metrics and results

`BacktestEngine._calculate_metrics` ([backtest-engine](backtest-engine.md)) computes the metrics below from the list of bets; helpers are in `packages/odds-analytics/odds_analytics/utils.py`.

| Metric | As computed |
|---|---|
| ROI | `total_profit / initial_bankroll * 100`: return on starting bankroll, not on turnover |
| Win rate | `wins / (wins + losses) * 100`; pushes excluded |
| Sharpe | mean of per-day profit divided by its sample standard deviation (n-1); risk-free 0; not annualised; 0 with fewer than 2 days |
| Sortino | mean divided by downside deviation (negative values only, divided by all days) |
| Max drawdown | largest peak-to-trough fall of the bet-by-bet bankroll curve starting at `initial_bankroll`; the percentage is the one at the deepest absolute drawdown |
| Profit factor | gross winning profit / gross losing loss; `inf` if no losses |
| Calmar | `abs(ROI / max_drawdown_pct)`; 0 if no drawdown |

The engine also reports streaks, average and largest win and loss, and breakdowns by market, bookmaker and month.

## BacktestResult

`packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestResult` is a frozen dataclass: `bet_stats`, `risk_metrics`, `breakdowns`, an `equity_curve` of daily points, and the `bets` tuple of `BetRecord`. About 25 properties re-expose nested fields (`result.roi`, `result.sharpe_ratio`).

- `to_json(path)` writes `metadata` (strategy, period, `config.initial_bankroll`, data quality), `summary` (rounded metrics and breakdowns), `equity_curve` and `bets`; `from_json` rebuilds it.
- `to_csv(path)` writes one row per bet with `roi` and `cumulative_profit` columns.
- `odds backtest show` prints the summary (`--verbose` adds breakdowns), `compare` tabulates two or more files with best ROI and best Sharpe, and `export` takes the JSON and CSV paths positionally ([howto-run-backtest](howto-run-backtest.md)).

## Hazards

- **"Daily returns" are dollar P&L.** The series fed to Sharpe and Sortino is profit per game date in currency units, not a return on bankroll, so values are not comparable across bankroll sizes or with the guide's rubric (AUD B6, re-verified).
- **JSON omits the run config.** Only the initial bankroll is stored; sizing, constraints, `sport_key` and `decision_hours_before_game` are lost, and values are rounded, so a reloaded result is not a lossless copy.
- **No CLV in backtests.** `BetRecord.opening_odds` and `closing_odds` exist but the engine never fills them.
- **`BetRecord.our_odds` duplicates `odds`** (set in `__post_init__`).

## Related

- [Backtest strategies](backtest-strategies.md), [Experiment: betting simulation](exp-betting-simulation.md)
