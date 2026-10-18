---
type: howto
title: Run a backtest
description: Check data prerequisites, list strategies, run, show, compare and export, or run from code.
sources:
  - odds_cli.commands.backtest.run_backtest
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.backtest.list_strategies
  - odds_cli.commands.backtest.show_results
  - odds_cli.commands.backtest.compare_results
  - odds_cli.commands.backtest.export_csv
  - odds_analytics.backtesting.config.BacktestConfig
  - odds_analytics.backtesting.config.BetSizingConfig
  - odds_analytics.backtesting.services.BacktestEngine
status: stable
---

# Run a backtest

1. **Check the data.** Confirm that the events in range have `odds_core.models.Odds` rows near the decision time and that the strategy's bookmakers and market key exist in them.
   - Historical OddsPortal and FDUK events have none, so they yield no bets ([data prerequisites](backtest-engine.md#data-prerequisites)).
2. **List the strategies.** `odds backtest list-strategies`.
   - `--strategy` is a key of `odds_cli.commands.backtest.STRATEGIES`: `flat`, `basic_ev`, `arbitrage`, `xgb_line_movement`, `lstm_line_movement` ([backtest strategies](backtest-strategies.md)).
3. **Run.**

   ```bash
   odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
   ```

   - `--bankroll` defaults to 10000, `--bet-sizing` to `fractional_kelly` with `--kelly-fraction` 0.25; `flat` and `percentage` use the `odds_analytics.backtesting.config.BetSizingConfig` defaults (100, 0.02) because the CLI cannot set them.
   - `--model-path` is for the two model strategies ([line-movement models](../modelling/line-movement-models.md)).
   - `--sport` filters by sport key; the default is all sports.
   - `--output-json` and `--output-csv` save results.
4. **View, compare and export.**

   ```bash
   odds backtest show results.json --verbose
   odds backtest compare a.json b.json
   odds backtest export results.json bets.csv
   ```

   - `show` prints the summary (`--verbose` adds breakdowns), `compare` needs at least two readable files and tabulates them with best ROI and best Sharpe, and `export` takes the JSON and CSV paths positionally.

## From code

```python
config = BacktestConfig(
    initial_bankroll=10000.0,
    start_date=datetime(2024, 10, 1, tzinfo=UTC),
    end_date=datetime(2024, 12, 31, tzinfo=UTC),
    sport_key="soccer_epl",
    sizing=BetSizingConfig(method="fractional_kelly", kelly_fraction=0.25),
)
async with async_session_maker() as session:
    result = await BacktestEngine(BasicEVStrategy(), config, OddsReader(session)).run()
```

Imports: `odds_analytics.backtesting` (`odds_analytics.backtesting.config.BacktestConfig`, `odds_analytics.backtesting.services.BacktestEngine`, `odds_analytics.backtesting.config.BetSizingConfig`), `odds_analytics.strategies`, `odds_core.database.async_session_maker`, `odds_lambda.storage.readers.OddsReader` (the reader the engine takes).

## Reading results

Interpretation of the metrics is on [Reading backtest-engine results](exp-betting-simulation.md#reading-backtest-engine-results); what each metric computes is on [backtest engine](backtest-engine.md#metrics).
