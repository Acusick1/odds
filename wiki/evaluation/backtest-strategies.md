---
type: component
title: Backtest strategies and adding one
description: The flat, basic EV, arbitrage and model strategies, their US/NBA-flavoured defaults and the STRATEGIES registry, and how to add and register a new strategy.
sources:
  - packages/odds-analytics/odds_analytics/strategies.py::FlatBettingStrategy
  - packages/odds-analytics/odds_analytics/strategies.py::BasicEVStrategy
  - packages/odds-analytics/odds_analytics/strategies.py::ArbitrageStrategy
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BettingStrategy
  - packages/odds-analytics/odds_analytics/backtesting/models.py::BetOpportunity
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-cli/odds_cli/commands/backtest.py::list_strategies
  - docs/BACKTESTING_GUIDE.md#available-strategies
  - docs/BACKTESTING_GUIDE.md#creating-custom-strategies
  - docs/CLI.md#list-strategies
  - docs/DEVELOPMENT.md#adding-betting-strategies
verified: "7887039"
status: stable
---

# Backtest strategies and adding one

Strategies subclass `BettingStrategy` and return `BetOpportunity` rows from `evaluate_opportunity` ([backtest engine](backtesting.md)). The CLI resolves `--strategy` through `packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES`; the defaults below are constructor defaults in `packages/odds-analytics/odds_analytics/strategies.py`.

## Registered strategies

| Name | Class | Defaults |
|---|---|---|
| `flat` | `FlatBettingStrategy` | `market` h2h, `outcome_pattern` home (or away, draw, favorite), `bookmaker` fanduel; confidence 0.5 |
| `basic_ev` | `BasicEVStrategy` | `sharp_book` pinnacle, `retail_books` fanduel/draftkings/betmgm, `min_ev_threshold` 0.03, `markets` h2h/spreads/totals |
| `arbitrage` | `ArbitrageStrategy` | `min_profit_margin` 0.01, `max_hold` 0.10, six US books incl. pinnacle |
| `xgb_line_movement`, `lstm_line_movement` | model strategies | need `--model-path`; see [line-movement models](../modelling/line-movement-models.md) |

- **flat** bets one outcome per event at one bookmaker; it is the baseline.
- **basic_ev** treats the sharp book's implied probability as the true probability, computes EV at each retail book via `calculate_ev`, and bets where EV is at least the threshold. `confidence` is that probability.
- **arbitrage** takes the best price per outcome across its books, skips markets whose hold exceeds `max_hold`, and bets all legs when `detect_arbitrage` finds a gap of at least `min_profit_margin`. It covers h2h, 1x2, spreads and totals.
- `odds backtest list-strategies` instantiates every registered class with no arguments and prints its docstring and parameters; every strategy module is imported when the command loads, including the torch-based LSTM one.

## Add a strategy

1. **Subclass.** Create the class in `packages/odds-analytics/odds_analytics/strategies.py` inheriting `BettingStrategy` (exported by `odds_analytics.backtesting`). Call `super().__init__(name=..., **params)`; the kwargs become `self.params`, are returned by `get_params()` and are stored in saved results ([results](backtesting.md#metrics)).
2. **Implement** `async evaluate_opportunity(self, event, odds_snapshot, config, session=None) -> list[BetOpportunity]`:
   - `event` is a `BacktestEvent`; its scores are present during backtests.
   - `odds_snapshot` is the list of `Odds` rows at decision time ([decision time](../modelling/decision-time-and-look-ahead.md)); filter by `market_key`, `bookmaker_key` and `outcome_name` yourself.
   - Return `BetOpportunity(event_id, market, outcome, bookmaker, odds, line, confidence, rationale)`. The engine reads `confidence` as the win probability for Kelly sizing; `recommended_stake` is not consulted.
3. **Register** it in the `STRATEGIES` dict in `packages/odds-cli/odds_cli/commands/backtest.py`; the key is the `--strategy` value.
4. **Test.** Add unit tests under `tests/unit/` next to `test_backtesting_engine.py`, and run `odds backtest run` on a small range ([running a backtest](backtesting.md#run-a-backtest)).
5. **Document** it on this page if it is non-trivial.

Traps when adding one:

- Every constructor argument needs a default, because `list-strategies` instantiates each class with no arguments.
- Use the market key the data actually has: EPL uses `1x2` where NBA uses `h2h` ([market keys](../collection/sources-and-keys.md)).
- The engine hands you its reader's `AsyncSession`; any query you make through it is not time-fenced ([look-ahead bias](../modelling/decision-time-and-look-ahead.md#hazards)).
- A zero or negative Kelly stake is floored to the minimum bet size, so return only opportunities you want bet.
- Model-driven strategies also implement training.

## Hazards

- **Defaults are US/NBA-flavoured.** Rosters are US books, `flat` uses `h2h` and the default `basic_ev` markets omit `1x2`, so on EPL data (`1x2`, UK books) they match nothing unless overridden; the CLI cannot override them ([market keys](../collection/sources-and-keys.md), [project overview](../foundations/project-overview.md); AUD T2/A7, re-verified).
- **`basic_ev` uses vig-inclusive sharp probability** as truth ([devigging](../foundations/odds-math-and-devigging.md)), and its `confidence` feeds Kelly sizing as a win probability.
- **Model strategies need `--model-path`** and a feature vector whose width matches the trained model; the XGBoost strategy builds a 6-value tabular vector that a model trained by the pipeline (more groups) will not accept ([line-movement models](../modelling/line-movement-models.md)).
- **The session passed to strategies is not time-fenced**, so queries through it can read data after the decision time.
- **Arbitrage is not hedged.** The engine sizes each leg independently by `confidence` 1.0; the stake split from `detect_arbitrage` is not used, and legs settle separately, so a run can show losses.
- **`--help` mentions `lstm`**, which is not a registry key; the valid names are those in the table.

## Related

- [Backtest engine, metrics and running a backtest](backtesting.md)
