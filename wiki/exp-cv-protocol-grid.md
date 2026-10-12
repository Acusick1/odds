---
type: experiment
title: CV protocol grid
summary: A 15-cell grid of window type x validation step on EPL - sliding-380 gives the lowest MSE, and a validation step of 50 keeps feature sets distinguishable.
sources:
  - docs/MODELING.md#cv-protocol-grid
  - docs/MODELING.md#experiment-log
  - experiments/scripts/cv_protocol_grid.py
verified_at: 7887039
---

# CV protocol grid

2026-03-16, about 1.7K EPL events, tabular features, Pinnacle sharp, 100 Optuna trials per cell. `experiments/scripts/cv_protocol_grid.py` crosses window type (expanding; sliding 95, 190, 380, 760 events) with validation step (10, 50, 150) and writes `experiments/results/cv_protocol_grid/grid_results.csv`. CV MSE:

| val_step | expanding | sliding-95 | sliding-190 | sliding-380 | sliding-760 |
|---|---|---|---|---|---|
| 10 | 0.000728 | 0.000752 | 0.000749 | **0.000706** | 0.000789 |
| 50 | 0.000757 | 0.000748 | 0.000756 | **0.000732** | 0.000794 |
| 150 | 0.000793 | 0.000758 | 0.000742 | 0.000752 | 0.000758 |

## Findings

- Sliding-380 (one season) is the best window; shorter windows starve the model and longer ones include stale data `(inferred)`.
- `val_step = 10` gives the lowest MSE but leads the tuner to pick identical hyperparameters for every feature set, so it cannot tell feature sets apart. `val_step = 50` keeps the sliding-380 advantage with enough per-fold signal. Decision: [decision-cv-protocol](decision-cv-protocol.md).
- A rerun with 32 features (2026-03-19) gave the same lowest-MSE cell (sliding-380:10, 0.000693), but R² favoured expanding windows: expanding:50 0.061, expanding:150 0.052, sliding-380:50 -0.013. Recorded reading at the time: expanding windows favour more features, sliding-380 penalises them, so the production retraining strategy decides.
- The sliding-380 R² weakness was later traced to the default search space, not the protocol ([exp-principled-search-space](exp-principled-search-space.md)).

## Hazards

- **Cells validate different events.** In `grid_results.csv` expanding cells start validation after 700 training events and sliding-380 after 380, so MSEs are computed on different fold sets ([walk-forward-cv](walk-forward-cv.md)).
- MSE favours sliding-380:10 while R² in the same CSV does not (sliding-380:10 R² -0.038, sliding-380:50 +0.026 on tabular only); the choice of step trades the two.
- Lowest MSE alone reflects target variance as much as skill ([exp-epl-sharp-reference](exp-epl-sharp-reference.md)).

## Related

- [Training config](training-config.md), [CLV model findings](clv-model-findings.md)
