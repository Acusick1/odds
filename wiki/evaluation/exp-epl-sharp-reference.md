---
type: explanation
title: EPL sharp-reference experiments
description: The March 2026 EPL runs covering Pinnacle as sharp reference, the #231 contamination bug, sliding-380 runs and the hybrid reference behind ADR-0003.
sources:
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_groups.TierSampler
  - experiments/configs/xgboost_epl_hybrid_sharp.yaml
  - experiments/configs/xgboost_epl_combined_tuning.yaml
  - experiments/scripts/hybrid_sharp_validation.py
status: stable
---

# EPL sharp-reference experiments

These runs are the evidence for [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md): PRs #217, #231, #253 and #234, plus `experiments/configs/xgboost_epl_hybrid_sharp.yaml` and `experiments/scripts/hybrid_sharp_validation.py`. The result tables below are as recorded in those PRs; no results artefact holds them (PR #253's MODELING.md entry points to `experiments/results/hybrid_sharp_validation/`, which was never committed).

## Context

FDUK sources its Pinnacle odds from Pinnacle's public API. PR #217 recorded that the API became unreliable after July 2025 (81 events in 2025-26 without Pinnacle closing odds); PR #253 recorded a Pinnacle API shutdown in January 2026 that excluded 71 of about 1.8K events from training. Betfair Exchange closing odds exist in football-data.co.uk from 2024-25 and as live snapshots from the Betfair job ([Betfair Exchange](../collection/betfair-exchange.md)). `betfair_exchange` has three writers: that job, FDUK `BFE*` columns and OddsPortal's Betfair row ([market-key dialects](../collection/sport-and-market-keys.md#market-key-dialects)).

## Pinnacle as sharp reference (March 2026)

About 1.7K combined OddsPortal + FDUK EPL events, devigged bet365 target, 100 trials per run. Signal comes from having a sharp reference, not from more data.

- **Control** (03-10): combined data with bet365 as its own sharp reference, CV R² = -0.005 ± 0.008.
- **Invalid run** (03-10): Pinnacle sharp, CV R² 0.031 ± 0.014, inflated by #231.
- **#231 bug** (fixed 2026-03-15): `odds_analytics.feature_groups.TierSampler` chose the latest snapshot by wall-clock time, so OddsPortal snapshots without Pinnacle beat FDUK snapshots with it for about 37 of 150 events in the final fold, and `np.nan_to_num` turned missing Pinnacle probabilities into 0.0. `odds_analytics.feature_groups.TierSampler` takes `required_bookmakers` and missing Pinnacle features stay NaN ([feature pipeline](../modelling/feature-pipeline.md)). Results from 2026-03-10 are invalid.
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

Expanding, step 150 (about 7 folds): tabular 0.000793, R² 0.019 ± 0.043; tabular + standings + match_stats 0.000767, R² 0.052 ± 0.045. The disagreement was attributed to training-set size (about 12 events per feature at 32 features under sliding-380) `(inferred)`; the later re-test showed a search-space artefact instead ([principled search space](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)). See also [CV protocol grid](exp-cv-protocol-grid.md).

## Hybrid reference (2026-03-19)

On 589 overlapping events Betfair Exchange `sharp_prob` correlates 0.9994 with Pinnacle, with a roughly 1.1 pp level shift. PR #253's MODELING.md entry states that XGBoost handles the shift through split-point adjustment; no run measures it. The hybrid recovers all 71 events (1,800 vs 1,729); folds 0-19 match Pinnacle-only and the two new folds score R² +7.4% and +2.7%; fixed-parameter run (expanding:50) R² 0.060 ± 0.080. Config: `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; script: `experiments/scripts/hybrid_sharp_validation.py`.

## Caveats

- A single training set mixes Pinnacle-based and Betfair-based `sharp_prob` values, and the model cannot tell which row is which except through the level shift ([sharp benchmark](../modelling/sharp-benchmark.md#code-copies)).
- `experiments/configs/xgboost_epl_hybrid_sharp.yaml` predates the principled search space, and the 2026-03-18 sliding-380 feature-group numbers were a search-space artefact and should not be used.
- The sliding-380 numbers above do not reproduce under current data and the hybrid reference; feature counts are as recorded (the results used a 7-column tabular group; the code defines 6).
- Fold counts disagree between sources: the 2026-03-15/16 expanding:150 runs are logged as 6 folds (tabular, + standings) and 5 folds (+ match_stats, + standings + match_stats), while this page says about 7 for the same protocol; `experiments/results/xgboost_epl_production_expanding_150/verification` holds no fold metadata, so neither can be confirmed.
- Results recorded before #474 come from runs where XGBoost early stopping monitored the validation fold, so their fold metrics are optimistic; current configs have no early stopping.

## Related

- [CLV model findings](clv-model-findings.md), [Target bookmaker and data volume](exp-target-and-data-volume.md)
