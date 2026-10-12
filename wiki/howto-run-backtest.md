---
type: howto
title: Run a backtest
summary: Backtests from the CLI or from code, data prerequisites, and viewing, comparing and exporting results.
sources:
  - packages/odds-cli/odds_cli/commands/backtest.py::run_backtest
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine
  - docs/BACKTESTING_GUIDE.md#quick-start
  - docs/BACKTESTING_GUIDE.md#programmatic-usage
  - docs/BACKTESTING_GUIDE.md#data-requirements
  - docs/CLI.md#backtesting-commands
verified_at: 7887039
---

# Run a backtest

## 1. Check the data

The engine needs FINAL events with scores and normalized `Odds` rows near the decision time (default 1 h before kickoff, tolerance 30 minutes); see [backtest-engine](backtest-engine.md). Populate them with [howto-historical-backfill](howto-historical-backfill.md). Historical OddsPortal and FDUK events have snapshots but no `Odds` rows, so they produce no bets. Confirm your strategy's bookmakers and market key exist in the data ([backtest-strategies](backtest-strategies.md)).

## 2. Run from the CLI

```bash
odds backtest list-strategies
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
```

- `--strategy` is a key of `STRATEGIES`: `flat`, `basic_ev`, `arbitrage`, `xgb_line_movement`, `lstm_line_movement`.
- `--bankroll` defaults to 10000, `--bet-sizing` to `fractional_kelly` with `--kelly-fraction` 0.25; `flat` and `percentage` use the `BetSizingConfig` defaults (100, 0.02) because the CLI cannot set them.
- `--model-path` is for the two model strategies ([line-movement-models](line-movement-models.md)).
- `--sport` filters by sport key; the default is all sports.
- `--output-json` and `--output-csv` save results.

## 3. View, compare, export

```bash
odds backtest show results.json --verbose
odds backtest compare a.json b.json
odds backtest export results.json bets.csv
```

`compare` needs at least two readable files; `export` takes the JSON and CSV paths positionally ([backtest-metrics](backtest-metrics.md)).

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

Imports: `odds_analytics.backtesting` (`BacktestConfig`, `BacktestEngine`, `BetSizingConfig`), `odds_analytics.strategies`, `odds_core.database.async_session_maker`, `odds_lambda.storage.readers.OddsReader`. `result.to_json(path)` and `result.to_csv(path)` export.

## Hazards

- The CLI parses `--start` and `--end` as naive datetimes; the code example above uses UTC-aware ones ([timezone-handling](timezone-handling.md)).
- Results carry the look-ahead and sizing caveats in [look-ahead-bias](look-ahead-bias.md) and [backtest-engine](backtest-engine.md).
- The guide's `basic_ev` EPL example needs bookmakers present in the data; default rosters are US books.

## Related

- [Howto: add a betting strategy](howto-add-betting-strategy.md), [CLI overview](cli-overview.md)
