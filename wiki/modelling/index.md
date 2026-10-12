---
title: CLV modelling
description: The CLV training target, the feature pipeline, the sharp benchmark, training, cross-validation and tuning, model publishing and inline scoring, and the modelling decisions behind them.
---

# CLV modelling

The XGBoost CLV model predicts how far a bookmaker's line moves between a decision snapshot and the close. These pages cover the target, the features, training and validation, the deployed model's publishing and scoring path, and the decisions that fixed the current setup. For what the model has been shown to do, start at [CLV model findings](../evaluation/clv-model-findings.md).

<!-- pages:start -->
### Concepts

- [CLV and the training target](clv-target.md) — What CLV means here, the devigged close-minus-snapshot target and the raw alternative, how the closing snapshot is chosen, where the target maths lives, and the code defaults every config must override.
- [Decision time and look-ahead bias](decision-time-and-look-ahead.md) — The as-of instant for bets and training rows, how backtests and samplers choose it, the no-future-data rule per layer, and the places that break it (symmetric odds window, unfenced sessions, lineup timing).
- [Sharp benchmark and retail-sharp divergence](sharp-benchmark.md) — The sharp reference price and sharp_bookmakers priority semantics, the retail-minus-sharp divergence that is the model's strongest feature and the agent's screener, and the three code copies whose defaults differ.

### Components

- [EPL context feature groups](epl-context-features.md) — The standings, match_stats, epl_schedule and epl_lineup extractors — inputs, windows, point-in-time rules and data dependencies — and the undocumented lineup signal check.
- [Feature pipeline and feature groups](feature-pipeline.md) — How feature_groups.py turns events into (X, y) — collection, sampling, target, adapters, orchestrator — the catalogue of feature groups and their column counts, the NaN-fill and variance policies, and how to add a group.
- [Hyperparameter tuner (Optuna)](hyperparameter-tuner.md) — OptunaTuner, its objective and search spaces including the principled space behind the deployed model, study storage, feature selection, and how failed trials are silently scored.
- [Line-movement models (XGBoost, LSTM)](line-movement-models.md) — The XGBoost and LSTM strategy classes that train the CLV regressor and bet on it in backtests, their confidence-to-Kelly mapping, LSTM's ruled-out status, and the misleading ml_strategy_example.
- [Publishing, loading and scoring the CLV model](model-publishing-and-scoring.md) — How to train, check and publish a model to the versioned S3 store, how load_model caches by ETag, how inline scoring writes the predictions table, and why scoring runs only on a local scheduler.
- [Training pipeline and config](training-pipeline.md) — From an MLTrainingConfig YAML to a saved model with odds train run/tune — config schema and defaults, data selection, splits, verification artifacts, MLflow — and the invalid values the schema silently accepts.

### Decisions

- [bet365 as the CLV target](decision-bet365-target.md) — Why the model predicts bet365 rather than Pinnacle line movement, with the NBA target and learning-curve evidence and what the choice requires of the data source.
- [Pinnacle with Betfair Exchange fallback as sharp reference](decision-hybrid-sharp-reference.md) — Why sharp_bookmakers is [pinnacle, betfair_exchange] with first-available semantics, the EPL sharp-reference runs and #231 contamination bug behind it, and where the hybrid is and is not the default.
- [Walk-forward CV (sliding-380:50, no test split)](decision-walk-forward-cv.md) — Why the CV protocol is walk-forward, sliding 380 events with 50-event steps and no test split; the grid evidence; how cross_validation.py builds folds and metrics; and the traps (one-event holdout, silent downgrade, early stopping).
<!-- pages:end -->
