---
type: component
title: Backtest engine, metrics and running a backtest
description: How BacktestEngine replays a strategy (odds at decision time, sizing, settlement), how each metric is computed and saved, how to run, compare and export backtests, and the data and sizing traps that invalidate results.
sources:
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BettingStrategy
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BacktestConfig
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BetSizingConfig
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BetConstraintsConfig
  - packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestResult
  - packages/odds-analytics/odds_analytics/utils.py::calculate_kelly_stake
  - packages/odds-analytics/odds_analytics/utils.py::calculate_sharpe_ratio
  - packages/odds-analytics/odds_analytics/utils.py::calculate_sortino_ratio
  - packages/odds-analytics/odds_analytics/utils.py::calculate_max_drawdown
  - packages/odds-analytics/odds_analytics/utils.py::calculate_profit_factor
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-cli/odds_cli/commands/backtest.py::run_backtest
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-cli/odds_cli/commands/backtest.py::show_results
  - packages/odds-cli/odds_cli/commands/backtest.py::compare_results
  - packages/odds-cli/odds_cli/commands/backtest.py::export_csv
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine._calculate_metrics
status: stable
---

# Backtest engine, metrics and running a backtest

`packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine` replays a [`packages/odds-analytics/odds_analytics/backtesting/services.py::BettingStrategy`](backtest-strategies.md) over historical events and returns a `packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestResult`. Build it as `BacktestEngine(strategy, config, OddsReader(session))`; it takes the reader, not a raw session ([OddsReader](../storage/odds-reader-and-storage-layer.md)). The CLI front end is `odds backtest` in `packages/odds-cli/odds_cli/commands/backtest.py`.

## Data prerequisites

The engine needs `FINAL` events with scores and normalized `packages/odds-core/odds_core/models.py::Odds` rows near the decision time. Historical OddsPortal and football-data.co.uk (FDUK) events have `packages/odds-core/odds_core/models.py::OddsSnapshot` rows but no `packages/odds-core/odds_core/models.py::Odds` rows: `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` write snapshots only, so those events log `no_odds_at_decision_time` and produce no bets ([historical loads](../collection/howto-load-historical-odds.md), [odds storage](../storage/odds-storage.md)). The Odds API historical backfill does write `packages/odds-core/odds_core/models.py::Odds` rows but is NBA-only and paid, so EPL backtests have little or no usable data. Confirm that the strategy's bookmakers and market key exist in the data before running ([backtest strategies](backtest-strategies.md)).

## Event loop

`packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine.run` loads `FINAL` events in the date range (optionally one `sport_key`) and keeps those with scores (`packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestEvent.from_db_event`). For each event the decision time is `commence_time - decision_hours_before_game` (default 1.0, see [decision time](../modelling/decision-time-and-look-ahead.md)); odds come from `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_odds_at_time` with `tolerance_minutes=30`; events with none are skipped. The strategy returns `packages/odds-analytics/odds_analytics/backtesting/models.py::BetOpportunity` rows, each staked, settled and appended with bankroll before and after. Bankroll compounds bet by bet.

## Sizing

`packages/odds-analytics/odds_analytics/backtesting/config.py::BetSizingConfig.method` is one of:

- `fractional_kelly` (default): `packages/odds-analytics/odds_analytics/utils.py::calculate_kelly_stake` with the opportunity's `confidence` as win probability, `kelly_fraction` 0.25, capped at `max_bet_percentage` 0.05 of bankroll. Quarter Kelly is meant to cut variance while keeping most of full Kelly's return `(inferred)`.
- `flat`: `flat_stake_amount` 100.
- `percentage`: `percentage_stake` 0.02 of bankroll.

Then `max_bet_size` (if set) caps the stake and `min_bet_size` (10) floors it. The CLI sets only `method` and `kelly_fraction`; the rest are programmatic.

## Settlement

h2h and `1x2` moneylines (a draw wins only the Draw outcome when any odds in the snapshot are `1x2`, else it is a push), spreads and totals (a push is profit 0). An unknown market logs a warning and settles as `unknown` with 0 profit. Settlement uses `odds_core.odds_math` helpers ([odds maths](../foundations/odds-math-and-devigging.md)).

## Metrics

`packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine._calculate_metrics` computes the metrics from the bet list; helpers are in `packages/odds-analytics/odds_analytics/utils.py`.

| Metric | As computed |
|---|---|
| ROI | `total_profit / initial_bankroll * 100`: return on starting bankroll, not on turnover |
| Win rate | `wins / (wins + losses) * 100`; pushes excluded |
| Sharpe | mean of per-day profit divided by its sample standard deviation (n-1); risk-free 0; not annualised; 0 with fewer than 2 days |
| Sortino | mean divided by downside deviation (negative values only, divided by all days) |
| Max drawdown | largest peak-to-trough fall of the bet-by-bet bankroll curve starting at `initial_bankroll`; the percentage is the one at the deepest absolute drawdown |
| Profit factor | gross winning profit / gross losing loss; `inf` if no losses |
| Calmar | `abs(ROI / max_drawdown_pct)`; 0 if no drawdown |

Conventional readings (the original guide, 2c0bc80; not derived from this engine): Sharpe above 1.0 good, above 2.0 excellent, above 3.0 exceptional; profit factor above 1.0 profitable, above 2.0 very good. Sortino reads higher than Sharpe for a profitable strategy because only downside is penalised. The Sharpe thresholds assume a return series, but this engine feeds dollar P&L (see Hazards), so they do not transfer.

The engine also reports streaks, average and largest win and loss, and breakdowns by market, bookmaker and month.

`packages/odds-analytics/odds_analytics/backtesting/models.py::BacktestResult` is a frozen dataclass: `bet_stats`, `risk_metrics`, `breakdowns`, an `equity_curve` of daily points, and the `bets` tuple of `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord`. About 25 properties re-expose nested fields (`result.roi`, `result.sharpe_ratio`). `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade.to_bet_record` reuses `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord`, so paper trades feed the same analysis tools ([paper trading](../agent/match-briefs-and-paper-trades.md)).

- `to_json(path)` writes `metadata` (strategy, period, `config.initial_bankroll`, data quality), a summary block (rounded metrics and breakdowns), `equity_curve` and `bets`; `from_json` rebuilds it.
- `to_csv(path)` writes one row per bet with `roi` and `cumulative_profit` columns.

## Run a backtest

```bash
odds backtest list-strategies
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
```

- `--strategy` is a key of `packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES`: `flat`, `basic_ev`, `arbitrage`, `xgb_line_movement`, `lstm_line_movement`.
- `--bankroll` defaults to 10000, `--bet-sizing` to `fractional_kelly` with `--kelly-fraction` 0.25; `flat` and `percentage` use the `packages/odds-analytics/odds_analytics/backtesting/config.py::BetSizingConfig` defaults (100, 0.02) because the CLI cannot set them.
- `--model-path` is for the two model strategies ([line-movement models](../modelling/line-movement-models.md)).
- `--sport` filters by sport key; the default is all sports.
- `--output-json` and `--output-csv` save results.

View, compare and export:

```bash
odds backtest show results.json --verbose
odds backtest compare a.json b.json
odds backtest export results.json bets.csv
```

`show` prints the summary (`--verbose` adds breakdowns), `compare` needs at least two readable files and tabulates them with best ROI and best Sharpe, and `export` takes the JSON and CSV paths positionally.

From code:

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

Imports: `odds_analytics.backtesting` (`packages/odds-analytics/odds_analytics/backtesting/config.py::BacktestConfig`, `packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine`, `packages/odds-analytics/odds_analytics/backtesting/config.py::BetSizingConfig`), `odds_analytics.strategies`, `odds_core.database.async_session_maker`, `odds_lambda.storage.readers.OddsReader`.

## Hazards

- **Historical OddsPortal and FDUK events have no `packages/odds-core/odds_core/models.py::Odds` rows**, so they yield no bets (see Data prerequisites).
- **Look-ahead.** `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_odds_at_time` is symmetric: the ±30 minute window around a 1 h decision time can return odds after the decision, and the session handed to strategies is not time-fenced ([look-ahead bias](../modelling/decision-time-and-look-ahead.md#hazards), [odds reader](../storage/odds-reader-and-storage-layer.md#get-odds-at-time)).
- **Negative-edge bets are placed.** Kelly returns 0 for no edge, then the floor raises it to `min_bet_size`; the later `stake < min_bet_size` skip can never trigger.
- **Constraint fields are inert.** Only `min_bet_size`, `max_bet_size` and `max_bet_percentage` are read; `min_odds`, `max_odds`, `allowed_markets`, `allowed_bookmakers`, `include_transaction_costs` and `transaction_cost_rate` are never consulted, and `progress_callback` is stored but unused (Rich progress is hard-wired).
- **Flat stakes are not bounded by bankroll.**
- **Default strategy rosters are US books and `h2h`**, matching nothing in EPL data ([backtest strategies](backtest-strategies.md#hazards)).
- **The CLI parses `--start` and `--end` as naive datetimes**; the code example uses UTC-aware ones ([timezones](../foundations/conventions.md#timezones)).
- **"Daily returns" are dollar P&L.** The series fed to Sharpe and Sortino is profit per game date in currency units, not a return on bankroll, so values are not comparable across bankroll sizes.
- **JSON omits the run config.** Only the initial bankroll is stored; sizing, constraints, `sport_key` and `decision_hours_before_game` are lost, and values are rounded, so a reloaded result is not a lossless copy.
- **No CLV in backtests.** `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord.opening_odds` and `closing_odds` exist but the engine never fills them. `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord.our_odds` duplicates `odds` (set in `__post_init__`).
- **Experiment numbers are not from this engine.** Exp 7 in [betting simulation](clv-model-findings.md#betting-simulation-exp-7-7b) comes from a standalone script with its own settlement.

## Related

- [Backtest strategies](backtest-strategies.md), [CLV model findings](clv-model-findings.md), [Line-movement models](../modelling/line-movement-models.md), [CLI overview](../foundations/packages-and-cli.md)
