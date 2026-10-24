---
type: howto
title: Add a backtest strategy
description: Subclass BettingStrategy, implement evaluate_opportunity, register it, and test it.
sources:
  - odds_analytics.backtesting.services.BettingStrategy
  - odds_analytics.backtesting.models.BetOpportunity
  - odds_analytics.backtesting.models.BacktestEvent
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.backtest.list_strategies
status: stable
---

# Add a backtest strategy

1. **Subclass.** Create the class in `odds_analytics.strategies` inheriting `odds_analytics.backtesting.services.BettingStrategy` (exported by `odds_analytics.backtesting`).
   - Call `super().__init__(name=..., **params)`; the kwargs become `self.params`, are returned by `odds_analytics.backtesting.services.BettingStrategy.get_params` and are stored in saved results.
   - Give every constructor argument a default, because `odds backtest list-strategies` instantiates each class with no arguments.
2. **Implement** `async evaluate_opportunity(self, event, odds_snapshot, config, session=None) -> list[BetOpportunity]`.
   - `event` is a `odds_analytics.backtesting.models.BacktestEvent`; its scores are present during backtests.
   - `odds_snapshot` is the list of `odds_core.models.Odds` rows at decision time ([decision time](../modelling/look-ahead-bias.md)); each row stores the source's decimal price (`price_decimal`, always above 1.0) and derives `american_price` and `implied_probability`; filter by `market_key`, `bookmaker_key` and `outcome_name` yourself, using the key the data actually has (EPL uses `1x2` where NBA uses `h2h`; [market keys](../collection/sport-and-market-keys.md#market-key-dialects)).
   - Return `BetOpportunity(event_id, market, outcome, bookmaker, odds, line, confidence, rationale)`, with `odds` the integer American quote (`odds_core.models.Odds.american_price`); compare prices by `price_decimal`. The engine reads `confidence` as the win probability for Kelly sizing and does not consult `recommended_stake`.
   - Return only opportunities you want bet: a zero or negative Kelly stake is floored to the minimum bet size ([backtest engine](backtest-engine.md#sizing)).
   - Avoid queries through `session` that read past the decision time; the engine hands you its reader's `AsyncSession`, which is not time-fenced ([look-ahead bias](../modelling/look-ahead-bias.md#where-the-rule-breaks)).
3. **Register** it in the `odds_cli.commands.backtest.STRATEGIES` dict; the key is the `--strategy` value, and `xgb_line_movement` is the one model strategy there.
4. **Test.** Add unit tests under `tests/unit/` next to `test_backtesting_engine.py`, and run `odds backtest run` on a small range ([run a backtest](howto-run-backtest.md)).
5. **Document** it in the table on [Backtest strategies](backtest-strategies.md).
   - Model-driven strategies also implement training ([line-movement models](../modelling/line-movement-models.md)).
