---
type: decision
title: Pinnacle with Betfair Exchange fallback as sharp reference
description: "Why sharp_bookmakers is [pinnacle, betfair_exchange] with first-available semantics, the EPL sharp-reference runs and #231 contamination bug behind it, and where the hybrid is and is not the default."
sources:
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_groups.TierSampler
  - experiments/configs/xgboost_epl_hybrid_sharp.yaml
  - experiments/configs/xgboost_epl_combined_tuning.yaml
  - experiments/scripts/hybrid_sharp_validation.py
  - odds_analytics.training.config.DataConfig
status: stable
---

# Pinnacle with Betfair Exchange fallback as sharp reference

**Decision (2026-03-19).** EPL training uses `sharp_bookmakers: [pinnacle, betfair_exchange]`; the first bookmaker in the list that has odds wins ([sharp benchmark](sharp-benchmark.md)). No ADR exists; the evidence is PRs #217, #231, #253 and #234, plus `experiments/configs/xgboost_epl_hybrid_sharp.yaml` and `experiments/scripts/hybrid_sharp_validation.py`. The result tables below are as recorded in those PRs; no results artefact holds them.

## Context

Pinnacle's public API stopped covering EPL events in January 2026, so 71 of about 1.8K events had no Pinnacle closing odds and were excluded. Betfair Exchange closing odds exist in football-data.co.uk from 2024-25 and as live snapshots from the Betfair job ([Betfair Exchange](../collection/betfair-exchange.md)). `betfair_exchange` has three writers: that job, FDUK `BFE*` columns and OddsPortal's Betfair row ([sources and keys](../collection/sources-and-keys.md)).

## Evidence

### Pinnacle as sharp reference (March 2026)

About 1.7K combined OddsPortal + FDUK EPL events, devigged bet365 target, 100 trials per run. Signal comes from having a sharp reference, not from more data.

- **Control** (03-10): combined data with bet365 as its own sharp reference, CV R² = -0.005 ± 0.008.
- **Invalid run** (03-10): Pinnacle sharp, CV R² 0.031 ± 0.014, inflated by #231.
- **#231 bug** (fixed 2026-03-15): `odds_analytics.feature_groups.TierSampler` chose the latest snapshot by wall-clock time, so OddsPortal snapshots without Pinnacle beat FDUK snapshots with it for about 37 of 150 events in the final fold, and `np.nan_to_num` turned missing Pinnacle probabilities into 0.0. Now `odds_analytics.feature_groups.TierSampler` takes `required_bookmakers` and features stay NaN ([feature pipeline](feature-pipeline.md)). Results from 2026-03-10 are invalid.
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

Expanding, step 150 (about 7 folds): tabular 0.000793, R² 0.019 ± 0.043; tabular + standings + match_stats 0.000767, R² 0.052 ± 0.045. The disagreement was attributed to training-set size (about 12 events per feature at 32 features under sliding-380) `(inferred)`; the later re-test showed a search-space artefact instead ([principled search space](../evaluation/exp-epl-feature-ablation-and-search-space.md)). See also [walk-forward CV](decision-walk-forward-cv.md).

### Hybrid reference (2026-03-19)

On 589 overlapping events Betfair Exchange `sharp_prob` correlates 0.9994 with Pinnacle, with a roughly 1.1 pp level shift that XGBoost absorbs through split points `(inferred)`. The hybrid recovers all 71 events (1,800 vs 1,729); folds 0-19 match Pinnacle-only and the two new folds score R² +7.4% and +2.7%; fixed-parameter run (expanding:50) R² 0.060 ± 0.080. Config: `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; script: `experiments/scripts/hybrid_sharp_validation.py`.

## Consequences

- The semantics changed from all-of to any-of: `TierSampler.required_bookmakers`, the event filter in `odds_analytics.feature_groups.prepare_training_data` and `odds_analytics.feature_extraction.TabularFeatureExtractor` all try bookmakers in priority order.
- The EPL production setup is `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]`, `closing_source_priority: [football_data_uk]` ([training pipeline](training-pipeline.md)).
- The agent's MCP screener defaults to the same list, and `odds_mcp.server.save_match_brief` stamps it ([MCP market tools](../agent/mcp-market-tools.md)).

## Hazards

- **Code default is Pinnacle-only.** The default is `["pinnacle"]`; the hybrid exists only where a config or MCP default sets it.
- **Mixed training set.** A single training set mixes Pinnacle-based and Betfair-based `sharp_prob` values (the model cannot tell which row is which except via the level shift). `required_bookmakers` is any-of, so a hybrid run can mix them.
- **Old configs.** `experiments/configs/xgboost_epl_hybrid_sharp.yaml` predates the principled search space.
- **Unreproducible counts.** The sliding-380 numbers above do not reproduce under current data and the hybrid reference; feature counts are as recorded (the tabular group now has 6 features, not 7).

## Related

- [CLV model findings](../evaluation/clv-model-findings.md), [bet365 as the CLV target](decision-bet365-target.md)
