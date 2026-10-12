---
type: decision
title: Walk-forward CV with sliding-380:50 and no test split
summary: Why walk-forward CV is the primary metric, test_split is 0.0, and the window is one season with 50-event steps.
sources:
  - AGENTS.md
  - docs/DATA_MODELS.md#cross-validation-protocol
  - docs/MODELING.md#cv-protocol-grid
  - docs/MODELING.md#current-conclusions
  - packages/odds-analytics/odds_analytics/training/cross_validation.py::make_walk_forward_splits
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
verified_at: 7887039
---

# Walk-forward CV with sliding-380:50 and no test split

**Decision.** Evaluate with walk-forward CV, `window_type: sliding`, `min_train_events = max_train_events = 380` (about one EPL season), `val_step_events: 50` (about 5 matchdays), `test_split: 0.0`. No ADR exists; rationale comes from `AGENTS.md`, `docs/DATA_MODELS.md` and `docs/MODELING.md`.

## Rationale

- **Walk-forward is the primary metric** because it simulates the production retrain cycle: train on the past, predict the next chunk, roll forward (`AGENTS.md`).
- **No test split.** A fixed chronological test split wastes training data and does not reflect deployment, where the model is retrained on all available data before predicting new events. There is no held-out test set.
- **Window and step** were chosen by grid search: sliding-380 gave the lowest MSE at validation steps 10 and 50; step 50 was chosen so the tuner can tell feature sets apart, since step 10 selects identical hyperparameters for every feature set ([exp-cv-protocol-grid](exp-cv-protocol-grid.md)).
- Once the search space stopped collapsing to constant predictors, sliding-380:50 also edged expanding windows ([exp-principled-search-space](exp-principled-search-space.md)).

## Caveats recorded

- CV R² from tuning is optimistically biased (best of N trials); relative comparisons between experiments tuned alike are fair.
- Under the original search space the protocol favoured expanding windows with more features; the production retraining strategy is the deciding factor `(inferred)` ([exp-epl-sharp-reference](exp-epl-sharp-reference.md)).

## Consequences

Folds: `floor((n_events - 380) / 50)`, about 26-28 for EPL ([walk-forward-cv](walk-forward-cv.md)).

## Hazards

- **`test_split: 0.0` still holds out one event** in `odds train run` (`max(1, ...)`), and the config default is `test_split: 0.2` with `cv_method: timeseries`, so the protocol exists only where a YAML sets it ([training-config](training-config.md), [training-pipeline](training-pipeline.md)).
- The docs said "~26 folds"; with 1,813 events the fold count is 28.

## Related

- [CLV model findings](clv-model-findings.md), [Hyperparameter tuner](hyperparameter-tuner.md)
