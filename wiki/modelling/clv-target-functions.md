---
type: reference
title: Target functions
description: The devigged and raw target functions, the `_compute_target` dispatch, closing-snapshot selection, and the target fields of FeatureConfig.
sources:
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.sequence_loader.calculate_devigged_totals_target
  - odds_analytics.sequence_loader.calculate_regression_target
  - odds_analytics.sequence_loader.extract_devigged_h2h_probs
  - odds_analytics.feature_groups._compute_target
  - odds_analytics.feature_groups._select_closing_snapshot
  - odds_analytics.training.config.FeatureConfig
status: stable
---

# Target functions

`odds_analytics.sequence_loader` holds the target maths for every model, although its name suggests only LSTM sequence loading. Its `odds_analytics.sequence_loader.TargetType` enum is referenced nowhere else in the repo (#473). Why the target is a delta of one bookmaker is on [CLV and the training target](clv-target.md).

## Devigged target

- `odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` returns the closing minus snapshot devigged probability of the configured outcome (`home`, `away` or `draw`) for one bookmaker, or `None` if either side lacks that bookmaker or the outcome. It handles 2-way and 3-way markets; `draw` exists only in 3-way markets.
- `odds_analytics.sequence_loader.extract_devigged_h2h_probs` converts prices with `odds_core.odds_math.calculate_implied_probability` and normalises with `odds_core.odds_math.devig_probabilities`. Devigging is proportional only, and the 3-way (`Draw`) probability is devigged together with home and away ([odds arithmetic](../foundations/odds-math.md), [devigging](../foundations/devigging.md)).
- `odds_analytics.sequence_loader.calculate_devigged_totals_target` does the same for Over/Under.

## Dispatch

`odds_analytics.feature_groups._compute_target` dispatches on `odds_analytics.training.config.FeatureConfig.target_type`:

- `"devigged_bookmaker"` uses the functions above.
- `"raw"` uses `odds_analytics.sequence_loader.calculate_regression_target`, the mean vig-inclusive implied-probability delta. Devigging therefore applies to the target only when the YAML sets `target_type: devigged_bookmaker`.

`odds_analytics.training.config.FeatureConfig` requires `target_type`. `target_bookmaker` is required for `devigged_bookmaker` and must be omitted for `raw`; `devigged_bookmaker` is valid only for `h2h`, `1x2` and `totals`, and `outcome` must be `over` or `under` exactly when the market is `totals`. A config that breaks these rules fails validation.

`odds_analytics.feature_groups._compute_target` passes `outcome` to every branch, so the target and the tabular features describe the same side. 

## Closing snapshot

`odds_analytics.feature_groups._select_closing_snapshot` takes the snapshots in `closing_tier`, prefers those that contain the target bookmaker, then the first match in `closing_source_priority`, else the latest. `closing_source_priority` is matched against `odds_core.models.OddsSnapshot.api_request_id`, whose values are provenance tags such as `football_data_uk` and `oddsportal_closing` ([provenance tags](../storage/odds-tables.md#provenance)). Events with no usable closing snapshot are dropped in `odds_analytics.feature_groups.prepare_training_data` ([feature pipeline](feature-pipeline.md)).

## Config fields

`odds_analytics.training.config.FeatureConfig` carries `target_type`, `target_bookmaker`, `closing_tier` and `closing_source_priority`; all fields and defaults are on [Training config](training-config.md). The tabular group's `retail_sharp_diff` is the strongest predictor of this target ([sharp benchmark](sharp-benchmark.md)).
