---
title: Modelling
description: The retail-EV target, the feature pipeline, the sharp benchmark, training, cross-validation and tuning.
---

# Modelling

These pages cover the training targets (the `retail_ev` target and the `raw` line-movement delta), the features, the sharp benchmark, and training, cross-validation and tuning. The decisions behind the current setup are ADRs: [the hybrid sharp reference](../../docs/adr/0003-hybrid-sharp-reference.md) and [a retail bet's EV against the Betfair Exchange close](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md). Results of the retail-EV runs are in [evaluation](../evaluation/index.md).

<!-- pages:start -->
### Reference

- [Cross-validation](cross-validation.md) — How cross_validation.py builds walk-forward folds, the other CV methods, per-fold metrics and train_with_cv.
- [EPL standings and schedule features](epl-context-features.md) — The standings and epl_schedule extractors behind the retail-EV as-of-t groups, with their inputs, windows, point-in-time rules and data dependencies.
- [Feature groups](feature-groups.md) — The catalogue of static feature groups and column counts, the tabular extractor and its sharp and retail defaults, and the NBA injury, rest and Polymarket groups.
- [Feature pipeline](feature-pipeline.md) — The five layers of odds_analytics.feature_groups, samplers, adapters, orchestrator, NaN-fill and variance policy.
- [Hyperparameter tuner (Optuna)](hyperparameter-tuner.md) — OptunaTuner, the objective, search-space schema, feature selection and CLI options, with the example search spaces cited by file.
- [Line-movement strategy (XGBoost)](line-movement-models.md) — The XGBoost strategy class, its confidence mapping, defaults, save and load, and ml_strategy_example.
- [Target functions](clv-target-functions.md) — The retail-EV row builder, its features, weights and keep-bar metrics, the raw delta function, the `_compute_target` dispatch, closing-snapshot selection, and the target fields of FeatureConfig.
- [Training config](training-config.md) — The MLTrainingConfig schema, every section's fields and defaults, data-source selection, and values the schema accepts that cannot train.
- [Training pipeline](training-pipeline.md) — What odds train run, tune, validate and list-configs do step by step, the splits, verification artifacts and MLflow tracking.

### Explanation

- [CLV and the training target](clv-target.md) — What CLV means here, the retail-EV target against the Betfair Exchange close and its baseline and weights, the raw close-minus-snapshot delta, and how sampling changes a target.
- [Decision time and look-ahead bias](look-ahead-bias.md) — The as-of instant for bets and training rows, the no-future-data rule per layer, and where it breaks.
- [Sharp benchmark and retail-sharp divergence](sharp-benchmark.md) — What the sharp reference is and why, its priority semantics, the divergence feature and screener, and how the three code copies differ.

### How-to guides

- [Add a feature group](howto-add-feature-group.md) — Steps to add a static feature group to the pipeline and verify it.
<!-- pages:end -->
