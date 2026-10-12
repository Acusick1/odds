---
type: concept
title: Look-ahead bias
summary: The no-future-data rule across backtesting and features, how each layer enforces it, and the symmetric get_odds_at_time window that breaks it.
sources:
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-lambda/odds_lambda/storage/polymarket_reader.py::PolymarketReader
  - packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/injury_features.py::extract_injury_features
  - packages/odds-analytics/odds_analytics/lstm_line_movement.py::LSTMLineMovementStrategy
  - AGENTS.md
  - docs/MODELING.md#guards-against-known-failure-modes
  - docs/INJURIES.md#feature-extraction
verified_at: 7887039
---

# Look-ahead bias

Historical analysis must see only data available at the decision instant ([decision-time](decision-time.md)). The repo rule is: use `get_odds_at_time()` for historical analysis, never current odds, and let backtests enforce `decision_time` (`AGENTS.md`). Enforcement is by convention at each layer, not by one shared mechanism.

## Layer by layer

- **Backtest odds.** `packages/odds-analytics/odds_analytics/backtesting/services.py::BacktestEngine.run` computes `decision_time = commence_time - decision_hours_before_game` and calls `OddsReader.get_odds_at_time(..., tolerance_minutes=30)` ([backtest-engine](backtest-engine.md), [odds-reader](odds-reader.md)).
- **Polymarket odds.** `packages/odds-lambda/odds_lambda/storage/polymarket_reader.py::PolymarketReader.get_price_at_time` and `get_orderbook_at_time` only return snapshots at or before the target.
- **Sampling.** `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` excludes `IN_PLAY` snapshots unless that is the decision tier. Including them leaked post-kickoff odds and produced a spurious R² of +0.45 before the fix ([exp-early-nba-signal-search](exp-early-nba-signal-search.md)).
- **Feature groups.** Injuries use only reports with `report_time <= snapshot_time` (`packages/odds-analytics/odds_analytics/injury_features.py::extract_injury_features`); standings, match stats and EPL schedule use matches strictly before `commence_time`; the LSTM adapter keeps sequences whose first entry is at or before the snapshot ([feature-groups](feature-groups.md), [feature-pipeline](feature-pipeline.md)).
- **Cross-source contamination** (#231) is a related bug, not a time leak: see [exp-epl-sharp-reference](exp-epl-sharp-reference.md).

## Hazards

- **`get_odds_at_time` is symmetric.** It searches `[t - tol, t + tol]` and returns the closest `odds_timestamp` by absolute distance, so it can return odds up to `tolerance_minutes` after the decision time. The default is 5 minutes; the backtest engine hard-codes 30 against a default decision time of 1 hour before the game (`packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_odds_at_time`). The Polymarket readers are backward-only and their docstrings claim the same guarantee as the main reader (AUD T4/B1, re-verified).
- **Strategies get an unfenced session.** The engine passes `self.reader.session` to `evaluate_opportunity`, so a strategy can query any time.
- **LSTM backtests have no explicit cutoff.** `LSTMLineMovementStrategy.evaluate_opportunity` loads every snapshot for the event and resamples on a grid anchored to `commence_time` that ends `lookback_hours / timesteps` before kickoff (3 h at the 72/24 defaults), so it leaks only if half that interval is shorter than the decision lead time; the training adapter, by contrast, filters sequences to the snapshot time ([line-movement-models](line-movement-models.md)).
- **EPL lineup features need the match's own XI**, which is announced about 60 minutes before kickoff, later than any decision tier used in training ([features-epl-context](features-epl-context.md)).

## Related

- [Walk-forward CV](walk-forward-cv.md) — training-side temporal ordering
- [NBA injury reports](nba-injury-reports.md) — injury timing
