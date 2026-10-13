---
type: concept
title: CLV and the training target
description: What CLV means here, the devigged close-minus-snapshot target and the raw alternative, how the closing snapshot is chosen, where the target maths lives, and the code defaults every config must override.
sources:
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/feature_groups.py::_compute_target
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
status: stable
---

# CLV and the training target

CLV (closing line value) is how far a bookmaker's line moves between a decision snapshot and the close. The CLV model regresses that movement: `target = devigged_fair_close - devigged_fair_at_snapshot` for one bookmaker (`target_bookmaker`). A positive value means the price at the snapshot was too low and the line moved up (`packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target` returns closing probability minus snapshot probability). Why a delta rather than the closing price is `(inferred)`: the decision needs the size of the move, and a delta is stationary and roughly mean-zero. The delta is measured from the sampled snapshot (see [decision time](decision-time-and-look-ahead.md)) to the close, so the same event can yield different targets under different samplers.

The production target is the devigged bet365 delta; Pinnacle's lines move too little to predict (CV R² -0.017 on NBA; bet365 reaches 0.036 on NBA and 0.058 on EPL). The evidence is on [bet365 as the CLV target](decision-bet365-target.md).

## Where the maths lives

- `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target` returns `closing home prob - snapshot home prob` for one bookmaker, or `None` if either side lacks that bookmaker. `packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_h2h_probs` converts prices with `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` and normalises with `packages/odds-core/odds_core/odds_math.py::devig_probabilities`. Devigging is proportional only, and the 3-way (`Draw`) target still uses the home probability only ([odds arithmetic and devigging](../foundations/odds-math-and-devigging.md)). It handles 2-way and 3-way markets.
- `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_totals_target` does the same for Over/Under.
- `packages/odds-analytics/odds_analytics/feature_groups.py::_compute_target` dispatches on `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig.target_type`: `"devigged_bookmaker"` uses the functions above; `"raw"` uses `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_regression_target`, the mean vig-inclusive implied-probability delta. Devigging therefore applies to the target only when the YAML sets `target_type: devigged_bookmaker`.
- The closing snapshot comes from `packages/odds-analytics/odds_analytics/feature_groups.py::_select_closing_snapshot`: snapshots in `closing_tier`, preferring those that contain the target bookmaker, then the first match in `closing_source_priority`, else the latest. `closing_source_priority` is matched against `packages/odds-core/odds_core/models.py::OddsSnapshot.api_request_id`, whose values are provenance tags such as `football_data_uk` and `oddsportal_closing` ([odds storage](../storage/odds-storage.md)). Events with no usable closing snapshot are dropped in `packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data` ([feature pipeline](feature-pipeline.md)).

The tabular group's `retail_sharp_diff` is the strongest predictor of this target ([sharp benchmark](sharp-benchmark.md)).

## Hazards

- **Config default is the raw target.** `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig.target_type` defaults to `"raw"` and `target_bookmaker` to `"pinnacle"`; the model is documented as the devigged bet365 target, so every training YAML must set both (same class). A config that omits them trains the rejected Pinnacle target.
- **h2h target ignores `outcome`.** `packages/odds-analytics/odds_analytics/feature_groups.py::_compute_target` calls `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target` without the outcome, which always returns the home-side delta, while the tabular features are built for `resolve_outcome_name(config, event)`. With `outcome: away` the features and target describe different sides (read from code; not run). Totals respect `outcome`.
- **Closing-tier targets are near zero.** Targets measured from a closing-tier snapshot have std 0.0016, so any R² near 0.6 on them is an artefact, not skill ([NBA timing and injury experiments](../evaluation/exp-nba-timing-and-injuries.md)).
- **Module name misleads.** The target and devig functions live in `packages/odds-analytics/odds_analytics/sequence_loader.py`, whose name suggests only LSTM sequence loading; its `packages/odds-analytics/odds_analytics/sequence_loader.py::TargetType` enum is referenced nowhere else in the repo.

## Related

- [Training pipeline](training-pipeline.md) and [walk-forward CV](decision-walk-forward-cv.md) — how the target is trained and evaluated
- [CLV model findings](../evaluation/clv-model-findings.md) — what the model can predict
- [Decision time and look-ahead bias](decision-time-and-look-ahead.md) — the snapshot must precede the close
