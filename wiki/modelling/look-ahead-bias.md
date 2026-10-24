---
type: explanation
title: Decision time and look-ahead bias
description: The as-of instant for bets and training rows, the no-future-data rule per layer, and where it breaks.
sources:
  - odds_analytics.backtesting.config.BacktestConfig
  - odds_analytics.backtesting.services.BacktestEngine
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.retail_ev.build_event_rows
  - odds_analytics.as_of_team_features.results_known_before
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.injury_features.extract_injury_features
  - odds_analytics.lstm_line_movement.LSTMLineMovementStrategy
  - odds_lambda.storage.polymarket_reader.PolymarketReader
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Decision time and look-ahead bias

Decision time is the instant a prediction or bet is assumed to be made. Everything after it is off limits; the target is measured from it to the close ([CLV target](clv-target.md)). It is set separately for backtests and for training. The repo rule is to use `odds_lambda.storage.readers.OddsReader.get_odds_at_time` for historical analysis, never current odds, and let backtests enforce `decision_time` ([AGENTS.md constraints](../../AGENTS.md#constraints)). Enforcement is by convention at each layer, not by one shared mechanism.

## Backtests

`odds_analytics.backtesting.config.BacktestConfig.decision_hours_before_game` sets the lead time ([backtest engine](../evaluation/backtest-engine.md#event-loop) has the default). `odds_analytics.backtesting.services.BacktestEngine.run` computes `decision_time = commence_time - decision_hours_before_game` and calls `odds_lambda.storage.readers.OddsReader.get_odds_at_time` with a hard-coded tolerance ([odds at a point in time](../storage/odds-reader.md#odds-at-a-point-in-time)). The CLI exposes no flag for the lead time.

## Training

`odds_analytics.training.config.SamplingConfig` selects the snapshot sampler, unless the caller passes one to `odds_analytics.feature_groups.prepare_training_data` ([feature pipeline](feature-pipeline.md)):

- `strategy: time_range` (the default) uses `odds_analytics.feature_groups.TimeRangeSampler`: snapshots inside a `min_hours` to `max_hours` window before kickoff, at most `max_samples_per_event`, picked from equal-width time bins.
- `strategy: tier` uses `odds_analytics.feature_groups.TierSampler`: one row per event, the latest snapshot whose tier is the `decision_tier` or any tier further from kickoff. `IN_PLAY` is excluded ([fetch tier](../storage/fetch-tier.md) has the tier boundaries). The default values are in [training config](training-config.md).

The pregame window is the documented default because sharp-retail divergence peaks 3-6 h before NBA games (`experiments/results/exp4_hours_to_game/FINDINGS.md`, PR #166); timing on EPL is untested ([NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md), [CLV model findings](../evaluation/clv-model-findings.md)).


## No future data, layer by layer

- **Backtest odds.** As above: `odds_lambda.storage.readers.OddsReader.get_odds_at_time` within its tolerance of `decision_time`.
- **Polymarket odds.** `odds_lambda.storage.polymarket_reader.PolymarketReader.get_price_at_time` and `odds_lambda.storage.polymarket_reader.PolymarketReader.get_orderbook_at_time` only return snapshots at or before the target.
- **Sampling.** `odds_analytics.feature_groups.TierSampler` excludes `IN_PLAY` snapshots unless that is the decision tier. Including them leaked post-kickoff odds and produced a spurious R² of +0.45 (commit 2a84d88, `experiments/results/exp2_feature_group_isolation/FINDINGS.md`; [early NBA signal](../evaluation/exp-early-nba-signal-and-lstm.md#the-in_play-leak)).
- **Retail-EV rows.** `odds_analytics.retail_ev.build_event_rows` takes each row's retail prices from the row's own snapshot at t, the exchange and Polymarket prices from the latest snapshots at or before t, and the lagged prices of the `dynamics` group from the latest snapshots of the same source at or before t − Δ; only the target's close, a live Betfair snapshot priced for every outcome within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before kick-off or the FDUK closing price (`odds_core.betfair_close.event_close`), lies after t. The FDUK closing price is treated as taken at kick-off, so it never stands in for a price at t. The schedule and standings groups count a match only once its result is known at t (`odds_analytics.as_of_team_features.results_known_before`). An FDUK opening row's t is FDUK's latest documented collection time, an upper bound: the price was taken at or before it, possibly a day or more earlier ([historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)). So such a row takes no other source's price at t: its exchange price, for the baseline too, is its own snapshot's or the row is dropped (a `residual` arm takes that same `odds_analytics.retail_ev.BASELINE_FEATURE` value as its base margin), and it gets no Polymarket price, lagged exchange price, schedule or standings (`odds_analytics.retail_ev.UPPER_BOUND_TIME_KINDS`). #548 shows the leak this rule prevents: Polymarket prices at the FDUK stamp predicted those rows' residual, and shifting them back 3 hours removed the gain. Unit tests in `tests/unit/test_retail_ev.py` rebuild the groups from truncated data and require the same features: `TestNoLookAhead` rebuilds the rows and price groups at every retail snapshot time in the window from the snapshots at or before it plus the close; `TestTeamFeaturesAsOfT` rebuilds the schedule and standings groups at a fixed set of decision times, including times during and just after a match played between t and kick-off, from only the matches finished by then. The event-level feature groups are refused for `retail_ev` (next item), and walk-forward folds leave a 7-day gap so no training close falls after a validation decision ([cross-validation](cross-validation.md#how-folds-are-built), [target functions](clv-target-functions.md#retail-ev-rows)).
- **Feature groups.** Injuries use only reports with `report_time <= snapshot_time` (`odds_analytics.injury_features.extract_injury_features`); standings, match stats and EPL schedule use matches strictly before `commence_time`, which is safe only when the decision is near kick-off: for a decision days out, a match played between the decision and kick-off reaches the features, so `odds_analytics.training.config.FeatureConfig` rejects them under `retail_ev`, which has its own as-of-t schedule and standings groups ([retail-EV features](clv-target-functions.md#retail-ev-features)). The LSTM adapter keeps sequences whose first entry is at or before the snapshot ([feature groups](feature-groups.md), [EPL context features](epl-context-features.md)).
- **Cross-source contamination** (PR #231) is a related bug, not a time leak: see [EPL sharp-reference experiments](../evaluation/exp-epl-sharp-reference.md).
- **Validation-fold leakage** is not a time leak either: XGBoost has no early stopping, so validation rows never influence the fitted trees; see [cross-validation](cross-validation.md#cv-methods-and-metrics).

## Where the rule breaks

- **`odds_lambda.storage.readers.OddsReader.get_odds_at_time` is symmetric.** It searches either side of the target and returns the closest `odds_timestamp` by absolute distance, so it can return odds up to the tolerance after the decision time. The backtest engine passes a tolerance that is a large share of its default lead time, so post-decision odds can win (#435; [odds reader](../storage/odds-reader.md#odds-at-a-point-in-time)). The Polymarket readers are backward-only and their docstrings claim the same guarantee as the main reader.
- **Strategies get an unfenced session**, so a strategy can query any time ([backtest engine](../evaluation/backtest-engine.md), #435).
- **Production uses the sharp tier.** `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` sets `sampling.strategy: tier` with `decision_tier: sharp`, so one row per event is sampled at the sharp tier rather than in the documented pregame window. For events that only have opening and closing snapshots, the candidate set is just the opening snapshot, so the effective decision time is wherever that snapshot was taken (read from code, not run).
- **Tier fallback dilutes the window.** A `pregame` request also accepts sharp, early and opening snapshots, so about 93% of OddsPortal events fell back to the same sharp-tier snapshot as a `sharp` request (Exp 6, see [target bookmaker and data volume](../evaluation/exp-target-and-data-volume.md)).
- **Backtest and training differ.** Backtests default to a short lead time ([backtest engine](../evaluation/backtest-engine.md#event-loop)) while the model is trained on the windows above. A model-driven strategy evaluated at its short lead time sees inputs unlike its training rows `(inferred)` ([line-movement models](line-movement-models.md)).
- **LSTM backtests have no explicit cutoff.** `odds_analytics.lstm_line_movement.LSTMLineMovementStrategy.evaluate_opportunity` loads every snapshot's `raw_data` for the event and resamples on a grid anchored to `commence_time` that ends `lookback_hours / timesteps` before kickoff ([line-movement models](line-movement-models.md)), so it leaks only if half that interval is shorter than the decision lead time; the training adapter, by contrast, filters sequences to the snapshot time.
- **Lineup features cannot be used.** They would need the starting XI, which is public about an hour before kick-off, after every training decision tier ([lineup signal check](../evaluation/exp-epl-feature-ablation-and-search-space.md#lineup-signal-check), #481, [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md)).

## Related

- [Cross-validation](cross-validation.md): training-side temporal ordering
- [NBA injuries and stats](../collection/nba-injuries-and-stats.md): injury timing
- [Run a backtest](../evaluation/howto-run-backtest.md): data prerequisites
