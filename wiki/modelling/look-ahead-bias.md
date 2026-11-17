---
type: explanation
title: Decision time and look-ahead bias
description: Decision time for bets and training rows, the no-future-data rule per layer, and where it breaks.
sources:
  - odds_lambda.candidate_rules.rules.run_rule
  - odds_lambda.candidate_rules.market_state.load_market_state
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.feature_groups.TimeRangeSampler
  - odds_analytics.retail_ev.build_event_rows
  - odds_analytics.as_of_team_features.results_known_before
  - odds_analytics.training.config.SamplingConfig
  - odds_analytics.injury_features.extract_injury_features
  - odds_lambda.tier_utils.calculate_tier
status: stable
---

# Decision time and look-ahead bias

Everything after the [decision time](../foundations/glossary.md#decision-time) is off limits, to avoid [look-ahead bias](../foundations/glossary.md#look-ahead-bias); the target is measured from it to the close, or to the [last snapshot](../foundations/glossary.md#last-snapshot) for a `raw` target ([training targets](clv-target.md)). It is set separately for backtests and for training. The repo rule is to read historical prices through `odds_core.quotes.quotes` bounded by `until` (and by `known_at` for what was stored by then), never current odds, and let backtests enforce the decision time ([AGENTS.md constraints](../../AGENTS.md#constraints), [reading quotes](../storage/price-tables.md#reading-prices)). Enforcement is by convention at each layer, not by one shared mechanism.

## Backtests

The replay backtest decides each event at every one of its `odds_lambda.candidate_rules.rules.decision_times`, and `odds_lambda.candidate_rules.market_state.load_market_state` reads only snapshots stamped at or before the decision time (or a lookback point) and within the rule's maximum input age of it ([market state](../evaluation/candidate-rules.md#market-state), [replay backtest](../evaluation/candidate-backtest.md#replay)). A rule's [fixture context](../evaluation/candidate-rules.md#fixture-context) counts a previous result only once it is known at the decision time, but the stored kick-offs it reads carry no history. A [model rule](../evaluation/candidate-backtest.md#model-rules) is tuned and refitted only on events that kicked off before the block it scores, labelled by a live Betfair close, which exists by then.

## Training

`odds_analytics.training.config.SamplingConfig` selects the snapshot sampler, unless the caller passes one to `odds_analytics.feature_groups.prepare_training_data`, which samples only of the sources whose last snapshot has a sharp book and measures each sampled snapshot against its own source's last snapshot ([feature pipeline](feature-pipeline.md)):

- `strategy: time_range` (the default) uses `odds_analytics.feature_groups.TimeRangeSampler`: snapshots inside a `min_hours` to `max_hours` window before kick-off, at most `max_samples_per_event`, picked from equal-width time bins.
- `strategy: tier` uses `odds_analytics.feature_groups.TierSampler`: one row per event, the latest snapshot whose tier, derived from its time, time basis, source and market (`odds_analytics.feature_groups.tiers_of`, over `odds_lambda.tier_utils.snapshot_tiers`), is the `decision_tier` or any tier further from kick-off. `IN_PLAY` is excluded ([fetch tier](../storage/fetch-tier.md) has the tier boundaries). The default values are in [training config](training-config.md).

The pregame window is the documented default because sharp-retail divergence peaks 3-6 h before NBA events (PR #166); timing on EPL is untested.

## No future data, layer by layer

- **Sampling.** `odds_analytics.feature_groups.TierSampler` excludes `IN_PLAY` snapshots unless that is the decision tier. The tier is derived from the snapshot's time (`odds_lambda.tier_utils.calculate_tier`), so a live snapshot stamped after kick-off is in-play: the OddsPortal live scraper keeps scraping after kick-off. A `raw` target's last snapshot is never such a snapshot either ([last snapshot](clv-target-functions.md#last-snapshot-of-a-raw-target)). Including them leaked post-kick-off prices and produced a spurious R² of +0.45 (commit 2a84d88).
- **Retail-EV rows.** `odds_analytics.retail_ev.build_event_rows` takes each row's retail prices from the row's own snapshot at t, the exchange and Polymarket prices (`odds_core.book_price.polymarket_price`) from the latest snapshots at or before t, and the lagged prices of the `dynamics` group from the latest snapshots of the same source and time basis at or before t − Δ; only the target's [close](clv-target.md#the-retail-ev-target) lies after t, and an FDUK close is treated as taken at kick-off, so it never stands in for a price at t. The schedule and standings groups count an event only once its result is known at t (`odds_analytics.as_of_team_features.results_known_before`). An FDUK opening row's t is only an upper bound, so it takes no other source's price at t ([FDUK opening rows](clv-target.md#the-retail-ev-target), `odds_analytics.retail_ev.is_exact_time`; a `residual` arm takes its own-snapshot `odds_analytics.retail_ev.BASELINE_FEATURE` as its base margin); in #548, shifting the Polymarket prices at the FDUK stamp back 3 hours removed their gain. Unit tests in `tests/unit/test_retail_ev.py` rebuild the groups from truncated data and require the same features: `TestNoLookAhead` rebuilds the rows and price groups at every retail snapshot time in the window from the snapshots at or before it plus the close; `TestTeamFeaturesAsOfT` rebuilds the schedule and standings groups at a fixed set of decision times, including times during and just after an event played between t and kick-off, from only the events finished by then. The event-level feature groups are refused for `retail_ev` (next item), and walk-forward folds leave a 7-day gap so no training close falls after a validation decision ([cross-validation](cross-validation.md#how-folds-are-built), [target functions](clv-target-functions.md#retail-ev-rows)).
- **Feature groups.** Injuries (teams matched through `odds_core.team.normalize_team_name`) use only reports with `report_time <= snapshot_time` (`odds_analytics.injury_features.extract_injury_features`). The event-level groups are computed as of the event's kick-off, which is safe only when the decision is near kick-off: for a decision days out, an event played between the decision and kick-off would reach team-level features, so `odds_analytics.training.config.FeatureConfig` rejects them under `retail_ev`, which has its own as-of-t schedule and standings groups ([retail-EV features](clv-target-functions.md#retail-ev-features), [feature groups](feature-groups.md), [EPL standings and schedule features](epl-context-features.md)).
- **Cross-source contamination** (PR #231) is a related bug, not a time leak: a snapshot from a source without the sharp book won the tier sampler by timestamp, which `required_bookmakers` now prevents.
- **Validation-fold leakage** is not a time leak either: XGBoost has no early stopping, so validation rows never influence the fitted trees; see [cross-validation](cross-validation.md#cv-methods-and-metrics).

## Where the rule breaks

- **Tier fallback dilutes the window.** A `pregame` request also accepts day-before, early and opening snapshots, so an event with no pregame snapshot is sampled at an earlier tier.
- **Backtest and training differ.** A rule replays at its own lead time ([replay backtest](../evaluation/candidate-backtest.md)) while a model is trained on the windows above, so a model evaluated at a different lead time sees inputs unlike its training rows `(inferred)`.
- **Lineup features cannot be used.** They would need the starting XI, which is public about an hour before kick-off, after every training decision tier (#481, [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md)).

## Related

- [Cross-validation](cross-validation.md): training-side temporal ordering
- [NBA injuries and stats](../collection/nba-injuries-and-stats.md): injury timing
- [Replay backtest](../evaluation/candidate-backtest.md): the decision state a replay builds
