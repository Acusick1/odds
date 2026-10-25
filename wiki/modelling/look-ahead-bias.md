---
type: explanation
title: Decision time and look-ahead bias
description: The as-of instant for bets and training rows, the no-future-data rule per layer, and where it breaks.
sources:
  - odds_analytics.candidate_backtest.replay
  - odds_lambda.candidate_rules.market_state.load_market_state
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.retail_ev.build_event_rows
  - odds_analytics.as_of_team_features.results_known_before
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.injury_features.extract_injury_features
  - odds_lambda.storage.polymarket_reader.PolymarketReader
status: stable
---

# Decision time and look-ahead bias

Decision time is the instant a prediction or bet is assumed to be made. Everything after it is off limits; the target is measured from it to the close ([CLV target](clv-target.md)). It is set separately for backtests and for training. The repo rule is to use `odds_lambda.storage.readers.OddsReader.get_odds_at_time` for historical analysis, never current odds, and let backtests enforce the decision time ([AGENTS.md constraints](../../AGENTS.md#constraints)). Enforcement is by convention at each layer, not by one shared mechanism.

## Backtests

`odds_analytics.candidate_backtest.replay` decides each completed event at `odds_lambda.candidate_rules.rules.decision_time` before its kick-off and builds the market state with `odds_lambda.candidate_rules.market_state.load_market_state`, which reads only snapshots stamped at or before that time ([replay backtest](../evaluation/candidate-backtest.md#replay)). `odds_lambda.storage.readers.OddsReader.get_odds_at_time` is a separate reader for other analyses ([odds at a point in time](../storage/odds-reader.md#odds-at-a-point-in-time)).

## Training

`odds_analytics.training.config.SamplingConfig` selects the snapshot sampler, unless the caller passes one to `odds_analytics.feature_groups.prepare_training_data` ([feature pipeline](feature-pipeline.md)):

- `strategy: time_range` (the default) uses `odds_analytics.feature_groups.TimeRangeSampler`: snapshots inside a `min_hours` to `max_hours` window before kickoff, at most `max_samples_per_event`, picked from equal-width time bins.
- `strategy: tier` uses `odds_analytics.feature_groups.TierSampler`: one row per event, the latest snapshot whose tier is the `decision_tier` or any tier further from kickoff. `IN_PLAY` is excluded ([fetch tier](../storage/fetch-tier.md) has the tier boundaries). The default values are in [training config](training-config.md).

The pregame window is the documented default because sharp-retail divergence peaks 3-6 h before NBA games (PR #166); timing on EPL is untested.

## No future data, layer by layer

- **Backtest odds.** As above: snapshots stamped at or before the decision time, within `odds_lambda.candidate_rules.market_state.SNAPSHOT_LOOKBACK` of it.
- **Polymarket odds.** `odds_lambda.storage.polymarket_reader.PolymarketReader.get_price_at_time` and `odds_lambda.storage.polymarket_reader.PolymarketReader.get_orderbook_at_time` only return snapshots at or before the target.
- **Sampling.** `odds_analytics.feature_groups.TierSampler` excludes `IN_PLAY` snapshots unless that is the decision tier. Including them leaked post-kickoff odds and produced a spurious R² of +0.45 (commit 2a84d88).
- **Retail-EV rows.** `odds_analytics.retail_ev.build_event_rows` takes each row's retail prices from the row's own snapshot at t, the exchange and Polymarket prices from the latest snapshots at or before t, and the lagged prices of the `dynamics` group from the latest snapshots of the same source at or before t − Δ; only the target's close, a live Betfair snapshot priced for every outcome within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before kick-off or the FDUK closing price (`odds_core.betfair_close.event_close`), lies after t. The FDUK closing price is treated as taken at kick-off, so it never stands in for a price at t. The schedule and standings groups count a match only once its result is known at t (`odds_analytics.as_of_team_features.results_known_before`). An FDUK opening row's t is FDUK's latest documented collection time, an upper bound: the price was taken at or before it, possibly a day or more earlier ([historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)). So such a row takes no other source's price at t: its exchange price, for the baseline too, is its own snapshot's or the row is dropped (a `residual` arm takes that same `odds_analytics.retail_ev.BASELINE_FEATURE` value as its base margin), and it gets no Polymarket price, lagged exchange price, schedule or standings (`odds_core.as_of.UPPER_BOUND_TIME_KINDS`). #548 shows the leak this rule prevents: Polymarket prices at the FDUK stamp predicted those rows' residual, and shifting them back 3 hours removed the gain. Unit tests in `tests/unit/test_retail_ev.py` rebuild the groups from truncated data and require the same features: `TestNoLookAhead` rebuilds the rows and price groups at every retail snapshot time in the window from the snapshots at or before it plus the close; `TestTeamFeaturesAsOfT` rebuilds the schedule and standings groups at a fixed set of decision times, including times during and just after a match played between t and kick-off, from only the matches finished by then. The event-level feature groups are refused for `retail_ev` (next item), and walk-forward folds leave a 7-day gap so no training close falls after a validation decision ([cross-validation](cross-validation.md#how-folds-are-built), [target functions](clv-target-functions.md#retail-ev-rows)).
- **Feature groups.** Injuries use only reports with `report_time <= snapshot_time` (`odds_analytics.injury_features.extract_injury_features`). The event-level groups are computed as of the event's kick-off, which is safe only when the decision is near kick-off: for a decision days out, a match played between the decision and kick-off would reach team-level features, so `odds_analytics.training.config.FeatureConfig` rejects them under `retail_ev`, which has its own as-of-t schedule and standings groups ([retail-EV features](clv-target-functions.md#retail-ev-features), [feature groups](feature-groups.md), [EPL standings and schedule features](epl-context-features.md)).
- **Cross-source contamination** (PR #231) is a related bug, not a time leak: a snapshot from a source without the sharp book won the tier sampler by timestamp, which `required_bookmakers` now prevents.
- **Validation-fold leakage** is not a time leak either: XGBoost has no early stopping, so validation rows never influence the fitted trees; see [cross-validation](cross-validation.md#cv-methods-and-metrics).

## Where the rule breaks

- **`odds_lambda.storage.readers.OddsReader.get_odds_at_time` is symmetric.** It searches either side of the target and returns the closest `odds_timestamp` by absolute distance, so it can return odds up to the tolerance after the decision time (#435; [odds reader](../storage/odds-reader.md#odds-at-a-point-in-time)). The replay backtest does not use it. The Polymarket readers are backward-only and their docstrings claim the same guarantee as the main reader.
- **Tier fallback dilutes the window.** A `pregame` request also accepts sharp, early and opening snapshots, so an event with no pregame snapshot is sampled at an earlier tier.
- **Backtest and training differ.** A rule replays at its own lead time ([replay backtest](../evaluation/candidate-backtest.md)) while a model is trained on the windows above, so a model evaluated at a different lead time sees inputs unlike its training rows `(inferred)`.
- **Lineup features cannot be used.** They would need the starting XI, which is public about an hour before kick-off, after every training decision tier (#481, [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md)).

## Related

- [Cross-validation](cross-validation.md): training-side temporal ordering
- [NBA injuries and stats](../collection/nba-injuries-and-stats.md): injury timing
- [Replay backtest](../evaluation/candidate-backtest.md): the decision state a replay builds
