---
type: howto
title: Add a betting strategy
summary: Subclass BettingStrategy, implement evaluate_opportunity with BacktestEvent, register it in STRATEGIES, and test it.
sources:
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BettingStrategy
  - packages/odds-analytics/odds_analytics/backtesting/models.py::BetOpportunity
  - packages/odds-analytics/odds_analytics/strategies.py::FlatBettingStrategy
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - docs/BACKTESTING_GUIDE.md#creating-custom-strategies
  - docs/DEVELOPMENT.md#adding-betting-strategies
verified_at: 7887039
---

# Add a betting strategy

1. **Subclass.** Create the class in `packages/odds-analytics/odds_analytics/strategies.py` inheriting `BettingStrategy` (exported by `odds_analytics.backtesting`). Call `super().__init__(name=..., **params)`; the kwargs become `self.params`, are returned by `get_params()` and are stored in saved results ([backtest-metrics](backtest-metrics.md)).
2. **Implement** `async evaluate_opportunity(self, event, odds_snapshot, config, session=None) -> list[BetOpportunity]`:
   - `event` is a `BacktestEvent`; its scores are present during backtests.
   - `odds_snapshot` is the list of `Odds` rows at decision time ([decision-time](decision-time.md)); filter by `market_key`, `bookmaker_key` and `outcome_name` yourself.
   - Return `BetOpportunity(event_id, market, outcome, bookmaker, odds, line, confidence, rationale)`. The engine reads `confidence` as the win probability for Kelly sizing; `recommended_stake` is not consulted ([backtest-engine](backtest-engine.md)).
3. **Register** it in the `STRATEGIES` dict in `packages/odds-cli/odds_cli/commands/backtest.py`; the key is the `--strategy` value.
4. **Test.** Add unit tests under `tests/unit/` next to `test_backtesting_engine.py`, and run `odds backtest run` on a small range ([howto-run-backtest](howto-run-backtest.md)).
5. **Document** it on [backtest-strategies](backtest-strategies.md) if it is non-trivial.

## Gotchas

- Every constructor argument needs a default: `odds backtest list-strategies` instantiates each class with no arguments ([backtest-strategies](backtest-strategies.md)).
- Use the market key the data actually has: EPL uses `1x2` where NBA uses `h2h` ([sport-and-market-keys](sport-and-market-keys.md)).
- The engine hands you its reader's `AsyncSession`; any query you make through it is not time-fenced ([look-ahead-bias](look-ahead-bias.md)).
- A zero or negative Kelly stake is floored to the minimum bet size, so return only opportunities you want bet ([backtest-engine](backtest-engine.md)).
- Model-driven strategies also implement training; see [line-movement-models](line-movement-models.md).

## Related

- [Backtest strategies](backtest-strategies.md)
