---
type: experiment
title: EPL feature ablation and principled search space
description: "The April 2026 EPL runs: per-group and FPL ablations (match_stats helps, schedule and FPL hurt), the sliding-window collapse to constant predictors, and the revised Optuna space that produced the deployed model."
sources:
  - experiments/configs/xgboost_epl_production_all_features.yaml
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/configs/xgboost_epl_production_tabular_only.yaml
  - experiments/configs/xgboost_epl_lineup.yaml
  - experiments/configs/xgboost_epl_hybrid_sharp.yaml
  - odds_analytics.feature_groups
status: stable
---

# EPL feature ablation and principled search space

Two linked questions on EPL, both answered in April 2026: which feature groups help CLV prediction, and why sliding-380:50 tuning produced identical, degenerate models. Answers: `match_stats` is the best single addition, `standings` is neutral, `epl_schedule` hurts slightly, `rest` produces nothing for EPL and FPL availability degrades results; and the original Optuna space, not the protocol, caused the collapse ([EPL context features](../modelling/epl-context-features.md), [feature pipeline](../modelling/feature-pipeline.md)).

## Per-group ablation (2026-04-23)

Recorded in #361; the results directories hold only verification CSVs (`experiments/results/xgboost_epl_production_all_features/verification`), so the metric tables below have no results file behind them.

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

## FPL availability (2026-03-20, #255)

41 features (tabular 7 + standings 11 + match_stats 14 + fpl_availability 9), hybrid sharp: CV R² = -0.017 ± 0.031 under sliding-380:50 (28 folds) and 0.022 ± 0.037 under expanding:150 (9 folds), both below the baselines without it (0.026; 0.052). Reading in the log: the market already prices squad availability. The agent-data inventory rejects FPL ownership and transfer data on this evidence and because transfer spikes lag the news the agent searches directly ([rejected sources](../agent/agent-data-sources.md#rejected-sources)).

## Principled search space re-test

Same day, 1,813 EPL events, hybrid sharp, 100 trials each. Under sliding-380:50 the original Optuna space produced byte-identical results for every feature set: each trial was a constant predictor (one unique prediction across the 50 validation examples, early-stopped at iteration 9; every config converged to the same trial 86 with identical val MSE to 16 decimals). A manual low-regularisation config (`min_child_weight` 5, `gamma` 0) did not degenerate and beat the tuner's best by about 12% MSE, so the search space, not the protocol, was at fault ([tuner](../modelling/hyperparameter-tuner.md), [walk-forward CV](../modelling/decision-walk-forward-cv.md)). The sliding-380:50 ablation therefore could not be reported alongside the table above, which is expanding-only.

### Revised space

Drawn from published XGBoost tuning guidance for small-N, low-SNR regression; defined in `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`:

| Parameter | Original | Principled |
|---|---|---|
| `n_estimators` | 50-400 | 100-600, step 50 |
| `max_depth` | 2-6 | 2-4 |
| `learning_rate` | 0.01-0.3 (log) | 0.01-0.1 (log) |
| `min_child_weight` | 5-50 | 1-15 |
| `subsample`, `colsample_bytree` | 0.5-1.0, step 0.1 | same |
| `reg_lambda` | 0.5-5.0 linear | 0.01-5.0 log |
| `gamma` | 0-1 tuned | 0.0 locked |
| `reg_alpha` | 0-2 tuned | 0.0 locked |
| `early_stopping_rounds` | 15 | 50 |

Reasons, all `(inferred)`:

- `min_child_weight` up to 50 forces each leaf to hold at least 13% of 380 rows, which guarantees collapse; a rule of thumb of at most sqrt(N)/10 suggests about 2.
- `gamma` and `min_child_weight` are redundant regularisers at low depth; locking `gamma` removes a joint over-regularisation trap.
- Tuning L1 and L2 together is avoided, so `reg_alpha` is locked; log-scale `reg_lambda` from 0.01 lets the tuner find lighter L2.
- With learning rates at or below 0.05 and noisy 50-example validation sets, improvement rarely shows before tree 20, so early stopping at 15 aborted before any tree helped. This was the largest single fix.

### Results (val R², val MSE)

| Config | Original space | Principled space |
|---|---|---|
| expanding:50 + tabular | +0.0354 | +0.0472, 0.000753 |
| expanding:50 + tabular + match_stats | +0.0514 | +0.0579, 0.000743 |
| sliding-380:50 + tabular | constant predictor | +0.0517, 0.000721 |
| sliding-380:50 + tabular + match_stats | constant predictor | **+0.0581, 0.000716** |

Under the principled space sliding-380:50 yields 48-50 unique predictions on 50 validation examples (best iteration 31-33, against 9) and slightly beats expanding:50 on MSE; the match_stats over tabular-only ranking holds under both protocols. Sliding-380:50 only works with this space.

## Outcome

Deployed as `epl-clv-home` `v308058c` (CV R² +0.058, MSE 0.000716), replacing the tabular-only model (+0.035) ([model hub](clv-model-findings.md#deployed-model), [publishing](../modelling/model-publishing-and-scoring.md)).

## Hazards

- **Search spaces in older YAMLs.** Most older committed configs still carry the original space (`min_child_weight` up to 50, early stopping 15), for example `experiments/configs/xgboost_epl_lineup.yaml` and `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; copy the search space from the `experiments/configs/xgboost_epl_production_*` configs when starting a new sliding-window run.
- The 2026-03-18 sliding-380 feature-group numbers (#234) were a search-space artefact and should not be used ([hybrid sharp reference](../modelling/decision-hybrid-sharp-reference.md)).
- **FPL is not reproducible.** The FPL tables and reader remain ([EPL context data](../collection/epl-context-data.md)) but `odds_analytics.feature_groups` has no `fpl_availability` group, so the run cannot be repeated as recorded.
- The `min_child_weight` rule of thumb has no recorded source.
- `rest` is a no-op for EPL by construction: its extractor needs NBA game logs, so the columns are NaN and the variance filter drops them.
- Feature counts are as recorded: the tabular group had 7 columns then and has 6 now.

## Related

- [CLV model findings](clv-model-findings.md), [Run an experiment](howto-run-experiment.md)
