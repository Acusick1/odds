---
type: decision
title: bet365 as the CLV target
summary: Why the model predicts bet365 line movement rather than Pinnacle's.
sources:
  - docs/MODELING.md#current-conclusions
  - docs/MODELING.md#pinnacle-as-target
  - docs/DATA_MODELS.md#training-target
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
verified_at: 7887039
---

# bet365 as the CLV target

**Decision.** The CLV model predicts the devigged bet365 close-minus-snapshot delta (`target_bookmaker: bet365`), not Pinnacle's ([clv](clv.md)). No ADR exists; the rationale is compiled from `docs/MODELING.md`.

## Context

The target bookmaker is configurable. Pinnacle was the first target (Feb 2026 NBA runs); bet365 became the target for NBA at scale and for EPL.

## Evidence

- Pinnacle CLV was unpredictable with public features: XGBoost CV R² -0.017 ± 0.015 (about 800 NBA events, tuner converged to maximum regularisation) and LSTM -0.075 ([exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md), [exp-lstm-sequence-models](exp-lstm-sequence-models.md)).
- Pinnacle lines move less: its MSE (0.000431) is 6.7x lower than bet365's (0.002883), leaving little variance to explain.
- bet365 reached CV R² 0.036 on about 5K NBA events and 0.058 on EPL ([exp-principled-search-space](exp-principled-search-space.md)).

## Consequences

- `target_bookmaker` must belong to the selected data source: bet365 exists only in OddsPortal data for NBA, and for EPL the combined OddsPortal + FDUK set ([training-config](training-config.md)).
- Pinnacle (with Betfair fallback) serves as the sharp reference feature instead ([decision-hybrid-sharp-reference](decision-hybrid-sharp-reference.md)).
- That bet365 is a retail book a bettor can actually use is a plausible reason but is not stated in the source `(inferred)`.

## Hazards

- The code default is `target_bookmaker: pinnacle` (and `target_type: raw`), so a config that omits them trains the rejected target (`packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig`).
- The production YAML lists bet365 in `retail_bookmakers` as well, so the target bookmaker's snapshot price is also an input to `retail_sharp_diff` ([retail-sharp-divergence](retail-sharp-divergence.md)).

## Related

- [CLV model findings](clv-model-findings.md)
