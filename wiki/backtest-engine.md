---
type: component
title: Backtest engine
summary: BacktestEngine and BacktestConfig - the event loop, odds at decision time through OddsReader, bet sizing (Kelly, flat, percentage), constraints and settlement.
sources:
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BettingStrategy
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BacktestConfig
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BetSizingConfig
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BetConstraintsConfig
  - packages/odds-analytics/odds_analytics/utils.py::calculate_kelly_stake
  - docs/BACKTESTING_GUIDE.md#bet-sizing-methods
  - docs/BACKTESTING_GUIDE.md#data-requirements
verified_at: 7887039
---

# Backtest engine

`packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine` replays a [`BettingStrategy`](backtest-strategies.md) over historical events and returns a [`BacktestResult`](backtest-metrics.md). Build it as `BacktestEngine(strategy, config, OddsReader(session))`; it takes the reader, not a raw session ([odds-reader](odds-reader.md)). Usage is in [howto-run-backtest](howto-run-backtest.md).

## Event loop

`run()` loads FINAL events in the date range (optionally one `sport_key`) and keeps those with scores (`BacktestEvent.from_db_event`). For each event: decision time is `commence_time - decision_hours_before_game` (default 1.0, [decision-time](decision-time.md)); odds come from `OddsReader.get_odds_at_time(..., tolerance_minutes=30)`; events with none are skipped. The strategy returns `BetOpportunity` rows, each is staked, settled and appended with bankroll before and after. Bankroll compounds bet by bet.

## Sizing

`BetSizingConfig.method` is one of:

- `fractional_kelly` (default): `calculate_kelly_stake` with the opportunity's `confidence` as win probability, `kelly_fraction` 0.25, capped at `max_bet_percentage` 0.05 of bankroll. Quarter Kelly is meant to cut variance while keeping most of full Kelly's return `(inferred)`.
- `flat`: `flat_stake_amount` 100. `percentage`: `percentage_stake` 0.02 of bankroll.

The CLI sets only `method` and `kelly_fraction`; the rest are programmatic. Then `max_bet_size` (if set) caps the stake and `min_bet_size` (10) floors it.

## Settlement

h2h and `1x2` moneylines (a draw wins only the Draw outcome when any odds in the snapshot are `1x2`, else it is a push), spreads and totals (a push is profit 0); an unknown market logs a warning and settles as `unknown` with 0 profit. Settlement uses `odds_core.odds_math` helpers.

## Data requirements

FINAL events with scores, and normalized `Odds` rows near the decision time. Backfill commands populate these ([howto-historical-backfill](howto-historical-backfill.md)).

## Hazards

- **Historical OddsPortal and FDUK events have no `Odds` rows.** `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` insert events and `OddsSnapshot` rows only; the engine reads only `Odds`, so those events log `no_odds_at_decision_time` ([historical-odds-ingestion](historical-odds-ingestion.md)).
- **Negative-edge bets are placed.** Kelly returns 0 for no edge, then the floor raises it to `min_bet_size`; the later `stake < min_bet_size` skip can never trigger (AUD B1/B6, re-verified).
- **Constraint fields are inert.** Only `min_bet_size`, `max_bet_size` and `max_bet_percentage` are read; `min_odds`, `max_odds`, `allowed_markets`, `allowed_bookmakers`, `include_transaction_costs` and `transaction_cost_rate` are never consulted, and `progress_callback` is stored but unused (Rich progress is hard-wired).
- **Stakes are not bounded by bankroll** for `flat`.
- **Look-ahead.** The ±30 minute window and the unfenced session are covered in [look-ahead-bias](look-ahead-bias.md).

## Related

- [Backtest metrics](backtest-metrics.md), [Line-movement models](line-movement-models.md)
