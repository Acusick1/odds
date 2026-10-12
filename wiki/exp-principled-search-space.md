---
type: experiment
title: Principled search space re-test
summary: Sliding-380:50 tuning collapsed to constant predictors; a revised Optuna space fixed this and produced the deployed model.
sources:
  - docs/MODELING.md#principled-search-space-and-protocol-re-test
  - docs/MODELING.md#deployed-model
  - docs/MODELING.md#experiment-log
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/configs/xgboost_epl_production_tabular_only.yaml
verified_at: 7887039
---

# Principled search space re-test

2026-04-23, 1,813 EPL events, hybrid sharp, 100 trials each. Under sliding-380:50 the original Optuna space produced byte-identical results for every feature set: each trial was a constant predictor (one unique prediction across the 50 validation examples, early-stopped at iteration 9; every config converged to the same trial 86 with identical val MSE to 16 decimals). A manual low-regularisation config (`min_child_weight` 5, `gamma` 0) did not degenerate and beat the tuner's best by about 12% MSE, so the search space, not the protocol, was at fault ([hyperparameter-tuner](hyperparameter-tuner.md), [decision-cv-protocol](decision-cv-protocol.md)).

## Revised space

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

## Results (val R², val MSE)

| Config | Original space | Principled space |
|---|---|---|
| expanding:50 + tabular | +0.0354 | +0.0472, 0.000753 |
| expanding:50 + tabular + match_stats | +0.0514 | +0.0579, 0.000743 |
| sliding-380:50 + tabular | constant predictor | +0.0517, 0.000721 |
| sliding-380:50 + tabular + match_stats | constant predictor | **+0.0581, 0.000716** |

Under the principled space sliding-380:50 yields 48-50 unique predictions on 50 validation examples (best iteration 31-33, against 9) and slightly beats expanding:50 on MSE; the match_stats over tabular-only ranking holds under both protocols.

## Outcome

Deployed as `epl-clv-home` `v308058c` (CV R² +0.058, MSE 0.000716), replacing the tabular-only model (+0.035) ([clv-model-findings](clv-model-findings.md), [s3-model-store](s3-model-store.md)).

## Hazards

- The 2026-03-18 sliding-380 feature-group numbers were a search-space artefact and should not be used ([exp-epl-sharp-reference](exp-epl-sharp-reference.md), [exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md)).
- Most older committed YAMLs still carry the original space (`min_child_weight` up to 50, early stopping 15), for example `experiments/configs/xgboost_epl_lineup.yaml` and `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; copy the search space from the `experiments/configs/xgboost_epl_production_*` configs when starting a new sliding-window run.

## Related

- [Walk-forward CV](walk-forward-cv.md)
