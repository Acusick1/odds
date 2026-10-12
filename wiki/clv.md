---
type: concept
title: Closing line value (CLV) and the training target
summary: What CLV means here, the devigged CLV-delta regression target, target_bookmaker, and where the target maths lives.
sources:
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/feature_groups.py::_compute_target
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - docs/DATA_MODELS.md#training-target
  - docs/MODELING.md#what-the-model-predicts
verified_at: 7887039
---

# Closing line value (CLV) and the training target

CLV is how far a bookmaker's line moves between a decision snapshot and the close. The CLV model regresses that movement: `target = devigged_fair_close - devigged_fair_at_snapshot` for one bookmaker (`target_bookmaker`). A positive value means the price at the snapshot was too low and the line moved up ([docs/DATA_MODELS.md](../docs/DATA_MODELS.md#training-target)). Why a delta rather than the closing price is `(inferred)` in the source doc: the decision needs the size of the move, and a delta is stationary and roughly mean-zero.

## Where the maths lives

- `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target` returns `closing home prob - snapshot home prob` for one bookmaker, or `None` if either side lacks that bookmaker. `extract_devigged_h2h_probs` converts prices with `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` and normalises with `packages/odds-core/odds_core/odds_math.py::devig_probabilities` (proportional devig; see [devigging](devigging.md), [odds-math](odds-math.md)). It handles 2-way and 3-way (`Draw` outcome) markets.
- `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_totals_target` does the same for Over/Under.
- `packages/odds-analytics/odds_analytics/feature_groups.py::_compute_target` dispatches on `FeatureConfig.target_type`: `"devigged_bookmaker"` uses the functions above; `"raw"` uses `calculate_regression_target`, the mean vig-inclusive implied-probability delta.
- The closing snapshot comes from `_select_closing_snapshot`: snapshots in `closing_tier`, preferring those that contain the target bookmaker, then the first match in `closing_source_priority` (matched on `OddsSnapshot.api_request_id`), else the latest. Events with no usable closing snapshot are dropped in `prepare_training_data` (see [feature-pipeline](feature-pipeline.md)).

The EPL production target is bet365 ([decision-bet365-target](decision-bet365-target.md)); the tabular group's `retail_sharp_diff` is the strongest predictor of it ([retail-sharp-divergence](retail-sharp-divergence.md), [sharp-benchmark](sharp-benchmark.md)).

## Hazards

- **Config default is the raw target.** `FeatureConfig.target_type` defaults to `"raw"` and `target_bookmaker` to `"pinnacle"`; the docs describe the devigged bet365 target as the model target, so every training YAML must set both (`packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig`).
- **h2h target ignores `outcome`.** `_compute_target` calls `calculate_devigged_bookmaker_target` without the outcome, which always returns the home-side delta, while the tabular features are built for `resolve_outcome_name(config, event)`. With `outcome: away` the features and target describe different sides (read from code; not run). Totals respect `outcome`.
- **Module name misleads.** The target and devig functions live in `sequence_loader.py`, whose name suggests only LSTM sequence loading; its `TargetType` enum is referenced nowhere else in the repo.

## Related

- [Walk-forward CV](walk-forward-cv.md) — how the target is evaluated
- [CLV model findings](clv-model-findings.md) — what the model can predict
- [Look-ahead bias](look-ahead-bias.md) — the snapshot must precede the close
