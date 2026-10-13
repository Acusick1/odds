---
status: proposed
---

# Evaluate with walk-forward CV: sliding 380 events, 50-event steps, no test split

Walk-forward CV with a sliding 380-event window and 50-event steps is the primary evaluation, and there is no held-out test set. The grid behind it is on [CV protocol grid](../../wiki/evaluation/exp-cv-protocol-grid.md); the rationale is only what that page and the linked PRs record.

## Context

`odds_analytics.training.cross_validation.run_cv` supports `walk_forward`, `timeseries` and `kfold`; the protocol was introduced in PR #230. A grid of window type (expanding; sliding 95, 190, 380, 760) by validation step (10, 50, 150) on about 1.7K EPL events gave the lowest MSE for sliding-380 at steps 10 and 50 (PR #234, `experiments/results/cv_protocol_grid/grid_results.csv`). Step 10 led the tuner to pick identical hyperparameters for every feature set. Under the original search space sliding-380:50 collapsed to constant predictors, and the principled space fixed it (PR #361).

## Decision

`cv_method: walk_forward`, `window_type: sliding`, `min_train_events = max_train_events = 380`, `val_step_events: 50`, `test_split: 0.0`, set in each training YAML. Walk-forward CV is the primary metric; there is no held-out test set.

## Considered options

- **Expanding windows.** Lowest MSE in no cell of the grid at steps 10 and 50; under the original search space they favoured more features, a weakness later traced to the search space (PRs #253, #361).
- **Other sliding windows (95, 190, 760).** Higher MSE than sliding-380 at steps 10 and 50.
- **Validation step 10.** Lowest MSE, but the tuner picks identical hyperparameters for every feature set.
- **A fixed chronological test split** (the code default `test_split: 0.2`). A fixed split wastes training data and does not reflect deployment, which retrains on all available data `(inferred)`.

## Consequences

- Walk-forward simulating the production retrain cycle, and dropping the test split because deployment retrains on all data, are `(inferred)`.
- Sliding-380 being "about one season" and shorter windows starving the model are stated in PR #234 but not measured by the CSV.
- The protocol only works with the principled search space (PR #361).
- The code defaults stay `cv_method: timeseries` and `test_split: 0.2`, so the protocol exists only where a YAML sets it.
- `test_split: 0.0` still holds out one event (#437).
- Grid cells validate on different event ranges (expanding from 700, sliding-380 from 380), so the MSE ranking is not like-for-like until re-run (#452).
- MSE favours step 10 while R² favours step 50; step 50 trades the two.
- Tuned CV R² is optimistically biased (best of N trials); comparisons between equally tuned runs are fair.

## Evidence

PRs #230, #234, #253, #361; `experiments/results/cv_protocol_grid/grid_results.csv`; `experiments/scripts/cv_protocol_grid.py`; #452. Wiki: [Cross-validation](../../wiki/modelling/cross-validation.md), [Training config](../../wiki/modelling/training-config.md).
