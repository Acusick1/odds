---
type: experiment
title: EPL sharp reference (Pinnacle, then hybrid)
summary: EPL runs on combined OddsPortal and FDUK data - signal needs a sharp reference, the #231 contamination bug, and the Betfair fallback that recovers events after Pinnacle's shutdown.
sources:
  - docs/MODELING.md#epl-with-pinnacle-sharp-reference
  - docs/MODELING.md#hybrid-sharp-reference
  - docs/MODELING.md#guards-against-known-failure-modes
  - docs/MODELING.md#experiment-log
  - experiments/configs/xgboost_epl_combined_tuning.yaml
  - experiments/configs/xgboost_epl_hybrid_sharp.yaml
  - experiments/scripts/hybrid_sharp_validation.py
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
verified_at: 7887039
---

# EPL sharp reference (Pinnacle, then hybrid)

March 2026, about 1.7K combined OddsPortal + FDUK EPL events, devigged bet365 target, 100 trials per run. Findings: signal comes from having a sharp reference, not from more data; a contamination bug (#231) invalidated early results; Betfair Exchange can substitute for Pinnacle ([decision-hybrid-sharp-reference](decision-hybrid-sharp-reference.md), [sharp-benchmark](sharp-benchmark.md)).

## Pinnacle as sharp reference

- **Control** (03-10): combined data with bet365 as its own sharp reference, CV R² = -0.005 ± 0.008. More data alone adds no signal.
- **Invalid run** (03-10): Pinnacle sharp, CV R² 0.031 ± 0.014, inflated by #231.
- **#231 bug** (fixed 2026-03-15): `TierSampler` chose the latest snapshot by wall-clock time, so OddsPortal snapshots without Pinnacle beat FDUK snapshots with it for about 37 of 150 events in the final fold, and `np.nan_to_num` turned missing Pinnacle probabilities into 0.0. Now `TierSampler` takes `required_bookmakers` and features stay NaN ([feature-pipeline](feature-pipeline.md)). Results from 2026-03-10 are invalid.
- **Post-fix, expanding:150** (03-15/16): tabular 0.019 ± 0.043; + standings 0.030 ± 0.040; + match_stats 0.025 ± 0.042; + standings + match_stats 0.052 ± 0.045 (reproduced 03-19).
- The MSE drop against the OddsPortal-only baseline (0.0024 to 0.0007) reflects lower target variance with Pinnacle closing, not necessarily better prediction.

Sliding-380, val step 50, about 26 folds (2026-03-18):

| Experiment | Features | CV MSE | CV R² |
|---|---|---|---|
| OddsPortal-only, bet365 sharp | 7 | 0.002410 | n/a |
| Combined, Pinnacle | 7 | 0.000732 | 0.026 ± 0.060 |
| + standings | 18 | 0.000726 | 0.023 ± 0.069 |
| + epl_schedule | 11 | 0.000736 | 0.023 ± 0.057 |
| + match_stats | 21 | 0.000740 | 0.010 ± 0.063 |
| + standings + epl_schedule | 22 | 0.000741 | 0.015 ± 0.064 |
| + standings + match_stats | 32 | 0.000759 | 0.004 ± 0.050 |
| + standings + match_stats + FPL | 41 | 0.000745 | 0.004 ± 0.050 |

Expanding, step 150 (about 7 folds): tabular 0.000793, R² 0.019 ± 0.043; tabular + standings + match_stats 0.000767, R² 0.052 ± 0.045. The disagreement was attributed to training-set size (about 12 events per feature at 32 features under sliding-380) `(inferred)`; the later [principled search re-test](exp-principled-search-space.md) showed a search-space artefact instead.

## Hybrid sharp reference (2026-03-19)

1.8K EPL events. Pinnacle's API shutdown (January 2026) excluded 71 events. Betfair Exchange `sharp_prob` correlates 0.9994 with Pinnacle on 589 overlapping events, with a roughly 1.1 pp level shift that XGBoost absorbs through split points `(inferred)`. `sharp_bookmakers: [pinnacle, betfair_exchange]` (first available wins) recovers all 71 events (1,800 vs 1,729); folds 0-19 match Pinnacle-only and the two new folds score +7.4% and +2.7%. Fixed-parameter run (expanding:50): R² 0.060 ± 0.080. Config: `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; script: `experiments/scripts/hybrid_sharp_validation.py`.

## Hazards

- Sliding-380 numbers above do not reproduce under current data and the hybrid reference; feature counts are as recorded (the tabular group now has 6 features, not 7) ([feature-groups](feature-groups.md)).
- `required_bookmakers` is any-of, so a hybrid run can mix Pinnacle and Betfair rows in one training set.

## Related

- [CV protocol grid](exp-cv-protocol-grid.md), [Feature ablation](exp-epl-feature-group-ablation.md), [CLV model findings](clv-model-findings.md)
