---
type: concept
title: Decision time and look-ahead bias
description: The as-of instant for bets and training rows, how backtests and samplers choose it, the no-future-data rule per layer, and the places that break it (symmetric odds window, unfenced sessions, lineup timing).
sources:
  - odds_analytics.backtesting.config.BacktestConfig
  - odds_analytics.backtesting.services.BacktestEngine
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.injury_features.extract_injury_features
  - odds_analytics.lstm_line_movement.LSTMLineMovementStrategy
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.polymarket_reader.PolymarketReader
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - AGENTS.md
status: stable
---

# Decision time and look-ahead bias

Decision time is the instant a prediction or bet is assumed to be made. Everything after it is off limits; the target is measured from it to the close ([CLV target](clv-target.md)). It is set separately for backtests and for training. The repo rule is: use `get_odds_at_time()` for historical analysis, never current odds, and let backtests enforce `decision_time` (`AGENTS.md`). Enforcement is by convention at each layer, not by one shared mechanism.

## Backtests

`odds_analytics.backtesting.config.BacktestConfig` field `decision_hours_before_game` defaults to 1.0. `odds_analytics.backtesting.services.BacktestEngine.run` computes `decision_time = commence_time - decision_hours_before_game` and calls `OddsReader.get_odds_at_time(..., tolerance_minutes=30)` ([backtesting](../evaluation/backtesting.md), [reader](../storage/odds-reader-and-storage-layer.md#get-odds-at-time)). The CLI exposes no flag for it, and the 30-minute tolerance is hard-coded.

## Training

`odds_analytics.training.config.SamplingConfig` selects the snapshot sampler ([feature pipeline](feature-pipeline.md)):

- `strategy: time_range` (the default) uses `odds_analytics.feature_groups.TimeRangeSampler`: snapshots between `min_hours` 3.0 and `max_hours` 12.0 before kickoff, at most `max_samples_per_event` 5, picked from equal-width time bins.
- `strategy: tier` uses `odds_analytics.feature_groups.TierSampler`: one row per event, the latest snapshot whose tier is the `decision_tier` (default `pregame`) or any tier further from kickoff. `IN_PLAY` is excluded ([fetch tier](../storage/fetch-tier.md): pregame is 3-12 h, sharp 12-24 h).

The pregame window (3-12 h) is the documented default because sharp-retail divergence peaks 3-6 h before NBA games (`experiments/results/exp4_hours_to_game/FINDINGS.md`, PR #166); timing on EPL is untested ([NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md), [CLV model findings](../evaluation/clv-model-findings.md)).

## No future data, layer by layer

- **Backtest odds.** As above: `get_odds_at_time` around `decision_time`.
- **Polymarket odds.** `odds_lambda.storage.polymarket_reader.PolymarketReader.get_price_at_time` and `get_orderbook_at_time` only return snapshots at or before the target.
- **Sampling.** `odds_analytics.feature_groups.TierSampler` excludes `IN_PLAY` snapshots unless that is the decision tier. Including them leaked post-kickoff odds and produced a spurious R² of +0.45 before the fix (commit 2a84d88, `experiments/results/exp2_feature_group_isolation/FINDINGS.md`; [early NBA signal](../evaluation/exp-early-nba-signal-and-lstm.md)).
- **Feature groups.** Injuries use only reports with `report_time <= snapshot_time` (`odds_analytics.injury_features.extract_injury_features`); standings, match stats and EPL schedule use matches strictly before `commence_time`; the LSTM adapter keeps sequences whose first entry is at or before the snapshot ([feature pipeline](feature-pipeline.md), [EPL context features](epl-context-features.md)).
- **Cross-source contamination** (PR #231) is a related bug, not a time leak: see [hybrid sharp reference](decision-hybrid-sharp-reference.md).

## Hazards

- **`get_odds_at_time` is symmetric.** It searches `[t - tol, t + tol]` and returns the closest `odds_timestamp` by absolute distance, so it can return odds up to `tolerance_minutes` after the decision time. The default is 5 minutes; the backtest engine hard-codes 30 against a default decision time of 1 hour before the game (`odds_lambda.storage.readers.OddsReader.get_odds_at_time`), so post-decision odds can win. The Polymarket readers are backward-only and their docstrings claim the same guarantee as the main reader (AUD T4/B1, re-verified).
- **Strategies get an unfenced session.** The engine passes `self.reader.session` to `evaluate_opportunity`, so a strategy can query any time ([backtest strategies](../evaluation/backtest-strategies.md)).
- **Production uses the sharp tier.** `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` sets `sampling.strategy: tier` with `decision_tier: sharp` (12-24 h), not the documented pregame window. For events that only have opening and closing snapshots, the candidate set is just the opening snapshot, so the effective decision time is wherever that snapshot was taken (read from code, not run).
- **Tier fallback dilutes the window.** A `pregame` request also accepts sharp, early and opening snapshots, so about 93% of OddsPortal events fell back to the same sharp-tier snapshot as a `sharp` request (Exp 6, see [data volume and target](decision-bet365-target.md)).
- **Backtest and training differ.** Backtests default to 1 h before kickoff; the model is trained at 3-24 h. A model-driven strategy evaluated at 1 h sees inputs unlike its training rows (inferred) ([line-movement models](line-movement-models.md)).
- **LSTM backtests have no explicit cutoff.** `odds_analytics.lstm_line_movement.LSTMLineMovementStrategy.evaluate_opportunity` loads every snapshot for the event and resamples on a grid anchored to `commence_time` that ends `lookback_hours / timesteps` before kickoff (3 h at the 72/24 defaults), so it leaks only if half that interval is shorter than the decision lead time; the training adapter, by contrast, filters sequences to the snapshot time.
- **EPL lineup features need the match's own XI**, announced about 60 minutes before kickoff, later than any decision tier used in training ([EPL context features](epl-context-features.md#hazards)).

## Related

- [Walk-forward CV](decision-walk-forward-cv.md) — training-side temporal ordering
- [NBA injuries and stats](../collection/nba-injuries-and-stats.md) — injury timing
- [Run a backtest](../evaluation/backtesting.md) — data prerequisites
