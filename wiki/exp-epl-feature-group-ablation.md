---
type: experiment
title: EPL feature group ablation
summary: Per-group EPL ablation plus the FPL availability runs - match_stats helps, standings is neutral, and schedule and FPL hurt.
sources:
  - docs/MODELING.md#epl-feature-group-ablation
  - docs/MODELING.md#experiment-log
  - docs/AGENT_DATA_SOURCES.md#evaluated-and-rejected
  - experiments/configs/xgboost_epl_production_all_features.yaml
verified_at: 7887039
---

# EPL feature group ablation

Which feature groups help CLV prediction on EPL? Answer: `match_stats` is the best single addition; `standings` is neutral; `epl_schedule` hurts slightly; `rest` produces nothing for EPL; FPL availability degrades results ([features-epl-context](features-epl-context.md), [feature-groups](feature-groups.md)).

## Per-group ablation (2026-04-23)

1,813 events through 2026-04-20, expanding:50 (22 folds, val step 50), hybrid sharp, 100 trials each, original search space. Config: `experiments/configs/xgboost_epl_production_all_features.yaml`.

| Config | Features | val R² | val MSE | Delta R² vs tabular |
|---|---|---|---|---|
| tabular only | 7 | +0.0354 ± 0.0643 | 0.000765 | n/a |
| + rest | 7 (+0) | +0.0354 ± 0.0643 | 0.000765 | 0.000 |
| + epl_schedule | 15 (+8) | +0.0277 ± 0.0564 | 0.000770 | -0.008 |
| + standings | 18 (+11) | +0.0367 ± 0.0657 | 0.000764 | +0.001 |
| **+ match_stats** | 21 (+14) | **+0.0514 ± 0.0791** | **0.000751** | **+0.016** |
| all 5 groups | 40 | +0.0429 ± 0.0781 | 0.000757 | +0.008 |

Effect sizes are small against fold-to-fold standard deviation (±0.06-0.08). An offline comparison scored 17 upcoming events with the tabular-only and tabular + match_stats models (1,088 shared event-snapshot pairs): they agreed on direction for 86.9% of pairs, and match_stats was on average 0.4 pp more bearish on the home side (mean delta -0.0040, max 0.0220). Without realised closing lines this shows only that the models differ.

The same-day sliding-380:50 ablation converged to constant predictors and is reported in [exp-principled-search-space](exp-principled-search-space.md).

## FPL availability (2026-03-20)

41 features (tabular 7 + standings 11 + match_stats 14 + fpl_availability 9), hybrid sharp: CV R² = -0.017 ± 0.031 under sliding-380:50 (28 folds) and 0.022 ± 0.037 under expanding:150 (9 folds), both below the baselines without it (0.026; 0.052). Reading in the log: the market already prices squad availability. The agent-data inventory rejects FPL ownership and transfer data on this evidence and because transfer spikes lag the news the agent searches directly ([decision-rejected-data-sources](decision-rejected-data-sources.md)).

## Hazards

- **FPL is not reproducible.** The FPL tables and reader remain (see [epl-context-data](epl-context-data.md)) but `packages/odds-analytics/odds_analytics/feature_groups.py` has no `fpl_availability` group, so this run cannot be repeated as recorded.
- `rest` is a no-op for EPL by construction: its extractor needs NBA game logs, so the columns are NaN and the variance filter drops them.
- Feature counts are as recorded (7 tabular then, 6 now).
- Under the original search space the sliding protocol was unusable, so this table is expanding-only.

## Related

- [CV protocol grid](exp-cv-protocol-grid.md), [CLV model findings](clv-model-findings.md)
