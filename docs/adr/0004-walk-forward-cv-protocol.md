---
status: superseded by ADR-0010
---

# Evaluate with walk-forward CV: sliding 380 events, 50-event steps, no test split

Walk-forward CV with a sliding 380-event window and 50-event steps is the primary evaluation, and there is no held-out test set. The grid behind it is on [CV protocol grid](../../wiki/evaluation/exp-cv-protocol-grid.md).

## Context

`odds_analytics.training.cross_validation.run_cv` supports `walk_forward`, `timeseries` and `kfold`. Walk-forward CV with expanding and sliding windows was added in PR #161 (2026-02-27), whose commit records that it better mimics the production retraining cadence. PR #230 (2026-03-15) dropped `test_split` from the walk-forward configs and recorded the evaluation rationale in the `CLAUDE.md` "Model Evaluation" section.

A grid of window type (expanding; sliding 95, 190, 380, 760) by validation step (10, 50, 150) on about 1.7K EPL events with tabular features gave the lowest MSE for sliding-380 at steps 10 and 50 (PR #234, `experiments/results/cv_protocol_grid/grid_results.csv`). PR #234 records that step 10 led the tuner to pick identical hyperparameters (250 trees, depth 2, `min_child_weight` 43) for every feature set, and that step 50 still let it select different ones.

## Decision

Adopted 2026-03-16 (PR #234, applied across the EPL tuning configs) and kept on 2026-04-23 (PRs #361, #362): `cv_method: walk_forward`, `window_type: sliding`, `min_train_events = max_train_events = 380`, `val_step_events: 50`, `test_split: 0.0`. Walk-forward CV is the primary metric; there is no held-out test set. PR #234 records that the protocol balances production fidelity with tuner signal quality, about 26 folds on about 1.8K EPL events with a realistic retrain cadence.

## Considered options

- **Expanding windows.** In the grid they have the lowest MSE in no cell at steps 10 and 50, but at step 50 their mean R² is higher (expanding:50 0.041 against sliding-380:50 0.026). A 32-feature rerun of the grid (2026-03-19, recorded in commit d34c267) gave expanding:50 R² 0.061 against sliding-380:50 −0.013, and the model published on 2026-04-21 used expanding:50 (d34c267). PR #361's description resolved the protocol in favour of expanding:50 after sliding-380:50 collapsed to a constant predictor. With the principled search space, sliding-380:50 scored R² 0.0581 and MSE 0.000716 against expanding:50's 0.0579 and 0.000743 on tabular plus match_stats (PR #362), and the `docs/MODELING.md` text merged in PR #361 records either protocol as defensible, with sliding-380:50 marginally better.
- **Other sliding windows (95, 190, 760).** Higher MSE than sliding-380 at steps 10 and 50.
- **Validation step 10.** Lowest MSE, but the tuner picks identical hyperparameters for every feature set.
- **Validation step 150.** Higher MSE than step 50 for sliding-380 (0.000752 against 0.000732).
- **A fixed chronological test split** (the code default `test_split: 0.2`). Rejected for the reason under Consequences.

## Consequences

- Walk-forward CV simulates the production retrain cycle (train on past, predict the next chunk, roll forward). A fixed chronological test split wastes training data and does not reflect deployment, which retrains on all available data before predicting new events (recorded reasons, `CLAUDE.md` "Model Evaluation", PR #230).
- Sliding-380 being "1 season", and shorter windows starving the model while longer ones include stale data, are stated in PR #234 but not measured by the CSV.
- Under the original search space the tuned sliding-380:50 model was a constant predictor, with one unique prediction across 50 validation examples (PR #361). The principled search space fixed that (PR #362).
- Of the six `experiments/configs/xgboost_epl_production_*.yaml` files, only `tabular_match_stats` (the deployed model) and `tabular_standings` use sliding-380:50. `all_features`, `expanding_50` and `tabular_only` use expanding windows from 700 events with step 50, and `expanding_150` uses step 150.
- The code defaults stay `cv_method: timeseries` and `test_split: 0.2`, so the protocol exists only where a YAML sets it.
- `test_split: 0.0` still holds out one event (#437).
- Grid cells validate on different event ranges (expanding from 700, sliding-380 from 380), so the MSE ranking is not like-for-like until re-run (#452). In PR #362 the committed expanding:50 tabular config also starts at 700 events against 380 for sliding-380:50; the expanding tabular plus match_stats config is not committed.
- `grid_results.csv` holds the tabular grid only. The 32-feature rerun's figures exist only in `docs/MODELING.md`, and the step-10 per-feature-set runs have no results artefact.
- For sliding-380, MSE favours step 10 (0.000706 against 0.000732) while mean R² favours step 50 (−0.038 against 0.026).
- Tuned CV R² is optimistically biased (best of N trials); comparisons between equally tuned runs are fair (`CLAUDE.md` "Model Evaluation").

## Evidence

- PRs #161, #230, #234, #361, #362; issues #437, #452.
- `CLAUDE.md` "Model Evaluation" section, added in 4b116fd (PR #230) and held in `AGENTS.md` until #458; text at `git show 7887039:AGENTS.md`.
- `docs/MODELING.md` commits a6f3d3a (PR #234), d34c267 and b61c8e8 (PR #361); text at `git show 7887039:docs/MODELING.md`.
- `experiments/results/cv_protocol_grid/grid_results.csv`; `experiments/scripts/cv_protocol_grid.py`; `experiments/configs/xgboost_epl_production_*.yaml`.
- Wiki: [Cross-validation](../../wiki/modelling/cross-validation.md), [Training config](../../wiki/modelling/training-config.md).
