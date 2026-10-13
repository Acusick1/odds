---
type: explanation
title: CV protocol grid
description: The window-type by validation-step grid behind ADR-0004, the 32-feature rerun, and what the grid can and cannot show.
sources:
  - experiments/scripts/cv_protocol_grid.py
  - odds_analytics.training.cross_validation.make_walk_forward_splits
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# CV protocol grid

This grid is the evidence for [ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md) (the protocol was introduced in PR #230; the grid is PR #234, with PRs #253 and #361 for the reruns). How the folds are built is on [Cross-validation](../modelling/cross-validation.md).

## Rationale recorded

- Walk-forward is the primary metric because it simulates the production retrain cycle: train on the past, predict the next chunk, roll forward `(inferred)`.
- No test split: a fixed chronological test split wastes training data and does not reflect deployment, where the model is retrained on all available data before predicting new events `(inferred)`.
- Window and step were chosen by the grid below: sliding-380 gave the lowest MSE at validation steps 10 and 50; step 50 was chosen so the tuner can tell feature sets apart, since step 10 selects identical hyperparameters for every feature set.
- Once the search space stopped collapsing to constant predictors, sliding-380:50 also edged expanding windows. Sliding-380:50 only works with the principled search space; the original space collapses to constant predictors ([tuner](../modelling/hyperparameter-tuner.md#production-search-space), [search-space experiment](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
- CV R² from tuning is optimistically biased (best of N trials); relative comparisons between experiments tuned alike are fair. Under the original search space the protocol favoured expanding windows with more features; the production retraining strategy is the deciding factor `(inferred)` ([EPL sharp-reference experiments](exp-epl-sharp-reference.md)).

## Grid evidence

2026-03-16, about 1.7K EPL events, tabular features, Pinnacle sharp, 100 Optuna trials per cell. `experiments/scripts/cv_protocol_grid.py` crosses window type (expanding; sliding 95, 190, 380, 760 events) with validation step (10, 50, 150) and writes `experiments/results/cv_protocol_grid/grid_results.csv` (PR #234). CV MSE:

| val_step | expanding | sliding-95 | sliding-190 | sliding-380 | sliding-760 |
|---|---|---|---|---|---|
| 10 | 0.000728 | 0.000752 | 0.000749 | **0.000706** | 0.000789 |
| 50 | 0.000757 | 0.000748 | 0.000756 | **0.000732** | 0.000794 |
| 150 | 0.000793 | 0.000758 | 0.000742 | 0.000752 | 0.000758 |

- Sliding-380 (one season, about 380 EPL matches) is the best window; shorter windows starve the model and longer ones include stale data `(inferred)`. PR #234 states this as the rationale but the CSV does not measure it.
- `val_step = 10` gives the lowest MSE but leads the tuner to pick identical hyperparameters for every feature set. `val_step = 50` (about 5 matchdays) keeps the sliding-380 advantage with enough per-fold signal.
- A rerun with 32 features (2026-03-19) gave the same lowest-MSE cell (sliding-380:10, 0.000693), but R² favoured expanding windows: expanding:50 0.061, expanding:150 0.052, sliding-380:50 -0.013. Recorded reading at the time: expanding windows favour more features, sliding-380 penalises them. The weakness was later traced to the default search space, not the protocol (PRs #253 and #361). Those 32-feature figures are as recorded in the PRs; they are not in `grid_results.csv`, which holds the tabular run.
- MSE favours sliding-380:10 while R² in the same CSV does not (sliding-380:10 R² -0.038, sliding-380:50 +0.026 on tabular only); the choice of step trades the two. Lowest MSE alone reflects target variance as much as skill.

## What the grid cannot show

Expanding cells start validation after 700 training events and sliding-380 after 380, so MSEs are computed on different validation sets and the ranking is not like-for-like until the grid is re-run on a common validation range (#452).

## Related

- [Cross-validation](../modelling/cross-validation.md), [CLV model findings](clv-model-findings.md), [EPL feature ablation](exp-epl-feature-ablation-and-search-space.md)
