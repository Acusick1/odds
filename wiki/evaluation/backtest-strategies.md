---
type: reference
title: Backtest strategies
description: The registered strategies, their constructor defaults and the STRATEGIES registry.
sources:
  - odds_analytics.strategies.FlatBettingStrategy
  - odds_analytics.strategies.BasicEVStrategy
  - odds_analytics.strategies.ArbitrageStrategy
  - odds_analytics.backtesting.services.BettingStrategy
  - odds_analytics.backtesting.models.BetOpportunity
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.backtest.list_strategies
status: stable
---

# Backtest strategies

Strategies subclass `odds_analytics.backtesting.services.BettingStrategy` and return `odds_analytics.backtesting.models.BetOpportunity` rows from `odds_analytics.backtesting.services.BettingStrategy.evaluate_opportunity` ([backtest engine](backtest-engine.md)). The CLI resolves `--strategy` through `odds_cli.commands.backtest.STRATEGIES`; the defaults below are constructor defaults in `odds_analytics.strategies`. To add one, see [Add a backtest strategy](howto-add-backtest-strategy.md).

## Registered strategies

| Name | Class | Defaults |
|---|---|---|
| `flat` | `odds_analytics.strategies.FlatBettingStrategy` | `market` h2h, `outcome_pattern` home (or away, draw, favorite), `bookmaker` fanduel; confidence 0.5 |
| `basic_ev` | `odds_analytics.strategies.BasicEVStrategy` | `sharp_book` pinnacle, `retail_books` fanduel/draftkings/betmgm, `min_ev_threshold` 0.03, `markets` h2h/spreads/totals |
| `arbitrage` | `odds_analytics.strategies.ArbitrageStrategy` | `min_profit_margin` 0.01, `max_hold` 0.10, six US books incl. pinnacle |
| `xgb_line_movement` | model strategy | need `--model-path`; see [line-movement models](../modelling/line-movement-models.md) |

- **flat** bets one outcome per event at one bookmaker; it is the baseline.
- **basic_ev** treats the sharp book's implied probability as the true probability, computes EV at each retail book via `odds_analytics.utils.calculate_ev` on the book's American quote (`odds_core.models.Odds.american_price`), and bets where EV is at least the threshold. `confidence` is that probability, and it is vig-inclusive ([devigging](../foundations/devigging.md)).
- **arbitrage** takes the best price per outcome across its books by `odds_core.models.Odds.price_decimal`, skips markets whose hold exceeds `max_hold`, and bets all legs when `odds_analytics.utils.detect_arbitrage` (which takes decimal prices) finds a gap of at least `min_profit_margin`. It covers h2h, 1x2, spreads and totals. The engine sizes each leg by `confidence` 1.0 and the stake split from `odds_analytics.utils.detect_arbitrage` is unused, so legs settle separately and a run can show losses (#436).
- `odds backtest list-strategies` instantiates every registered class with no arguments and prints its docstring and parameters; every strategy module is imported when the command loads.

## Defaults and the CLI

The rosters are US books, `flat` uses `h2h` and the default `basic_ev` markets omit `1x2`, so on EPL data (`1x2`, UK books) they match nothing unless overridden; the CLI cannot override them (#465; [market keys](../collection/sport-and-market-keys.md#market-key-dialects)). Model strategies need a feature vector whose width matches the trained model ([line-movement models](../modelling/line-movement-models.md#backtest-behaviour), #476). The session passed to strategies is not time-fenced ([backtest engine](backtest-engine.md#event-loop), #435).
