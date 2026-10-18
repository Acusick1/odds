---
title: CLV modelling
description: The CLV training target, the feature pipeline, the sharp benchmark, training, cross-validation and tuning, model publishing and inline scoring.
---

# CLV modelling

The XGBoost CLV model predicts how far a bookmaker's line moves between a decision snapshot and the close. These pages cover the target, the features, training and validation, and the deployed model's publishing and scoring path. The decisions that fixed the current setup are ADRs: [bet365 as the target](../../docs/adr/0002-bet365-clv-target.md), [the hybrid sharp reference](../../docs/adr/0003-hybrid-sharp-reference.md) and [the walk-forward CV protocol](../../docs/adr/0004-walk-forward-cv-protocol.md). For what the model has been shown to do, start at [CLV model findings](../evaluation/clv-model-findings.md).

<!-- pages:start -->
### Reference

- [Cross-validation](cross-validation.md) — How cross_validation.py builds walk-forward folds, the other CV methods, per-fold metrics and train_with_cv.
- [EPL context feature groups](epl-context-features.md) — The standings, match_stats and epl_schedule extractors with their inputs, windows, point-in-time rules and data dependencies.
- [Feature groups](feature-groups.md) — The catalogue of feature groups and column counts, the tabular extractor and its sharp and retail defaults, and the NBA injury, rest and Polymarket groups.
- [Feature pipeline](feature-pipeline.md) — The five layers of odds_analytics.feature_groups, caches, samplers, adapters, orchestrator, NaN-fill and variance policy.
- [Hyperparameter tuner (Optuna)](hyperparameter-tuner.md) — OptunaTuner, the objective, search-space schema, feature selection and CLI options, with the production search space cited by file.
- [Line-movement strategies (XGBoost, LSTM)](line-movement-models.md) — The two strategy classes, their shared confidence mapping, defaults, save and load, and ml_strategy_example.
- [Model store](model-store.md) — odds model publish, the S3 layout and bucket, load_model caching and model_supports_sport.
- [Prediction scoring](prediction-scoring.md) — The score_events flow, the predictions table, where scoring runs and how failures surface.
- [Target functions](clv-target-functions.md) — The devigged and raw target functions, the `_compute_target` dispatch, closing-snapshot selection, and the target fields of FeatureConfig (the feature group names it validates are on [Feature groups](feature-groups.md)).
- [Training config](training-config.md) — The MLTrainingConfig schema, every section's fields and defaults, data-source selection, and values the schema accepts that cannot train.
- [Training pipeline](training-pipeline.md) — What odds train run, tune, validate and list-configs do step by step, the splits, verification artifacts and MLflow tracking.

### Explanation

- [CLV and the training target](clv-target.md) — What CLV means here, why the target is a close-minus-snapshot delta of one devigged bookmaker, and how sampling changes the target.
- [Decision time and look-ahead bias](look-ahead-bias.md) — The as-of instant for bets and training rows, the no-future-data rule per layer, and where it breaks.
- [Sharp benchmark and retail-sharp divergence](sharp-benchmark.md) — What the sharp reference is and why, its priority semantics, the divergence feature and screener, and how the three code copies differ.

### How-to guides

- [Add a feature group](howto-add-feature-group.md) — Steps to add a static feature group to the pipeline and verify it.
- [Train and publish a CLV model](howto-train-and-publish-model.md) — Write a config, train or tune, check the artifacts, publish, and confirm scoring.
<!-- pages:end -->
