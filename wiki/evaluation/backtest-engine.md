---
type: reference
title: Backtest engine
description: BacktestEngine event loop, sizing, settlement, metrics as computed, BacktestResult and its JSON and CSV forms.
sources:
  - odds_analytics.backtesting.services.BacktestEngine
  - odds_analytics.backtesting.services.BettingStrategy
  - odds_analytics.backtesting.config.BacktestConfig
  - odds_analytics.backtesting.config.BetSizingConfig
  - odds_analytics.backtesting.config.BetConstraintsConfig
  - odds_analytics.backtesting.models.BacktestResult
  - odds_analytics.utils.calculate_kelly_stake
  - odds_analytics.utils.calculate_sharpe_ratio
  - odds_analytics.utils.calculate_sortino_ratio
  - odds_analytics.utils.calculate_max_drawdown
  - odds_analytics.utils.calculate_profit_factor
  - odds_lambda.storage.readers.OddsReader
  - odds_core.paper_trade_models.PaperTrade
  - odds_cli.commands.backtest.run_backtest
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.backtest.show_results
  - odds_cli.commands.backtest.compare_results
  - odds_cli.commands.backtest.export_csv
  - odds_analytics.backtesting.services.BacktestEngine._calculate_metrics
status: stable
---

# Backtest engine

`odds_analytics.backtesting.services.BacktestEngine` replays a [`odds_analytics.backtesting.services.BettingStrategy`](backtest-strategies.md) over historical events and returns a `odds_analytics.backtesting.models.BacktestResult`. Build it as `BacktestEngine(strategy, config, OddsReader(session))`; it takes the reader (`odds_lambda.storage.readers.OddsReader`), not a raw session ([OddsReader](../storage/odds-reader.md)). The CLI front end is `odds backtest` in `odds_cli.commands.backtest`; steps are on [Run a backtest](howto-run-backtest.md).

## Data prerequisites

The engine needs `FINAL` events with scores and normalized `odds_core.models.Odds` rows near the decision time. Historical OddsPortal and football-data.co.uk (FDUK) events have `odds_core.models.OddsSnapshot` rows but no `odds_core.models.Odds` rows: `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` write snapshots only, so those events log `no_odds_at_decision_time` and produce no bets (#435; [historical ingest scripts](../collection/historical-ingest-scripts.md), [odds tables](../storage/odds-tables.md)). The Odds API historical backfill does write `odds_core.models.Odds` rows but is NBA-only and paid, so EPL backtests have little or no usable data ([Odds API backfill](../collection/odds-api-backfill.md)).

## Event loop

`odds_analytics.backtesting.services.BacktestEngine.run` loads `FINAL` events in the date range (optionally one `sport_key`) and keeps those with scores (`odds_analytics.backtesting.models.BacktestEvent.from_db_event`). For each event the decision time is `commence_time - decision_hours_before_game` (default 1.0, see [decision time](../modelling/look-ahead-bias.md)); odds come from `odds_lambda.storage.readers.OddsReader.get_odds_at_time` with `tolerance_minutes=30`; events with none are skipped. The strategy returns `odds_analytics.backtesting.models.BetOpportunity` rows, each staked, settled and appended with bankroll before and after. Bankroll compounds bet by bet.

- `odds_lambda.storage.readers.OddsReader.get_odds_at_time` is symmetric, so the 30 minute window around a 1 h decision time can return odds after the decision (#435; [where the rule breaks](../modelling/look-ahead-bias.md#where-the-rule-breaks)).
- The engine passes its reader's session to `odds_analytics.backtesting.services.BettingStrategy.evaluate_opportunity`, so a strategy can query any time (#435).

## Sizing

`odds_analytics.backtesting.config.BetSizingConfig.method` is one of:

- `fractional_kelly` (default): `odds_analytics.utils.calculate_kelly_stake` with the opportunity's `confidence` as win probability, `kelly_fraction` 0.25, capped at `max_bet_percentage` 0.05 of bankroll.
- `flat`: `flat_stake_amount` 100. Flat stakes are not bounded by bankroll (#436).
- `percentage`: `percentage_stake` 0.02 of bankroll.

Then `max_bet_size` (if set) caps the stake and `min_bet_size` (10) floors it. Kelly returns 0 for no edge and the floor lifts it to `min_bet_size`, so every opportunity is bet and the later `stake < min_bet_size` skip never triggers (#436). The CLI sets only `method` and `kelly_fraction`; the rest are programmatic.

Only `min_bet_size`, `max_bet_size` and `max_bet_percentage` of `odds_analytics.backtesting.config.BetConstraintsConfig` are read; `min_odds`, `max_odds`, `allowed_markets`, `allowed_bookmakers`, `include_transaction_costs` and `transaction_cost_rate` are never read (#436). `progress_callback` is stored but unused, because Rich progress is hard-wired (#473). The default strategy rosters are US books and `h2h` ([backtest strategies](backtest-strategies.md)).

## Settlement

h2h and `1x2` moneylines (a draw wins only the Draw outcome when any odds in the snapshot are `1x2`, else it is a push), spreads and totals (a push is profit 0). An unknown market logs a warning and settles as `unknown` with 0 profit. Settlement uses `odds_core.odds_math` helpers ([odds arithmetic](../foundations/odds-math.md)).

## Metrics

`odds_analytics.backtesting.services.BacktestEngine._calculate_metrics` computes the metrics from the bet list; helpers are in `odds_analytics.utils`.

| Metric | As computed |
|---|---|
| ROI | `total_profit / initial_bankroll * 100`: return on starting bankroll, not on turnover |
| Win rate | `wins / (wins + losses) * 100`; pushes excluded |
| Sharpe | mean of per-day profit divided by its sample standard deviation (n-1); risk-free 0; not annualised; 0 with fewer than 2 days |
| Sortino | mean divided by downside deviation (negative values only, divided by all days) |
| Max drawdown | largest peak-to-trough fall of the bet-by-bet bankroll curve starting at `initial_bankroll`; the percentage is the one at the deepest absolute drawdown |
| Profit factor | gross winning profit / gross losing loss; `inf` if no losses |
| Calmar | `abs(ROI / max_drawdown_pct)`; 0 if no drawdown |

The series fed to Sharpe and Sortino is profit per game date in currency units, not a return on bankroll, so values are not comparable across bankroll sizes (#478). The engine also reports streaks, average and largest win and loss, and breakdowns by market, bookmaker and month.

## Result forms

`odds_analytics.backtesting.models.BacktestResult` is a frozen dataclass: `bet_stats`, `risk_metrics`, `breakdowns`, an `equity_curve` of daily points, and the `bets` tuple of `odds_analytics.backtesting.models.BetRecord`. About 25 properties re-expose nested fields (`result.roi`, `result.sharpe_ratio`). `odds_core.paper_trade_models.PaperTrade.to_bet_record` reuses `odds_analytics.backtesting.models.BetRecord`, so paper trades feed the same analysis tools ([paper trading](../agent/paper-trading.md); odds-core therefore depends on odds-analytics, #459).

- `to_json(path)` writes `metadata` (strategy, period, `config.initial_bankroll`, data quality), a summary block (rounded metrics and breakdowns), `equity_curve` and `bets`; `odds_analytics.backtesting.models.BacktestResult.from_json` rebuilds it. Sizing, constraints, `sport_key` and `decision_hours_before_game` are not stored and values are rounded, so a reloaded result is not a lossless copy (#478).
- `to_csv(path)` writes one row per bet with `roi` and `cumulative_profit` columns.
- `odds_analytics.backtesting.models.BetRecord.opening_odds` and `closing_odds` exist but the engine never fills them, so backtests report no CLV; `odds_analytics.backtesting.models.BetRecord.our_odds` duplicates `odds` (set in `__post_init__`) (#478).

## CLI date parsing

The CLI parses `--start` and `--end` as naive datetimes, while code callers pass UTC-aware ones ([timezones](../foundations/timezones.md), #461).

## Related

- [Run a backtest](howto-run-backtest.md), [Backtest strategies](backtest-strategies.md), [CLV model findings](clv-model-findings.md), [Line-movement models](../modelling/line-movement-models.md)
