---
type: component
title: Publishing, loading and scoring the CLV model
description: How to train, check and publish a model to the versioned S3 store, how load_model caches by ETag, how inline scoring writes the predictions table, and why scoring runs only on a local scheduler.
sources:
  - packages/odds-cli/odds_cli/commands/model.py::publish_model
  - packages/odds-cli/odds_cli/commands/train.py::run_training
  - packages/odds-cli/odds_cli/commands/train.py::run_tuning
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
  - packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter
  - packages/odds-core/odds_core/prediction_models.py::Prediction
  - packages/odds-core/odds_core/config.py::ModelConfig
  - deployment/aws/bootstrap/s3_models.tf
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py
  - packages/odds-mcp/odds_mcp/server.py::get_predictions
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Publishing, loading and scoring the CLV model

Trained CLV models are published to a shared, versioned S3 bucket and loaded by inline scoring, which writes `packages/odds-core/odds_core/prediction_models.py::Prediction` rows. The bucket is created in `deployment/aws/bootstrap/s3_models.tf` (versioning on, AES256, public access blocked, `prevent_destroy`) so a dev `terraform destroy` cannot delete it ([AWS infrastructure](../operations/aws-infrastructure.md)).

The deployed model is `epl-clv-home` version `v308058c` (published 2026-04-23; CV R² +0.058, MSE 0.000716), from `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` (config adopted in PR #362, findings in PR #361; [CLV model findings](../evaluation/clv-model-findings.md)). The live S3 object is the authority and cannot be verified from the repo; check with `aws s3 ls s3://<model-bucket>/epl-clv-home/`.

## Train, check and publish

1. **Write a config.** Start from the production YAML ([training pipeline](training-pipeline.md)). Set `target_type: devigged_bookmaker` and `target_bookmaker` explicitly (code defaults are `raw` and `pinnacle`), `sport_key`, and the protocol from [walk-forward CV](decision-walk-forward-cv.md). Run `odds train validate --config <file>`.
2. **Train.** `odds train run --config <file>` (options: `--output`, `--dry-run`, `--verbose`, `--track`, `--tracking-uri`, `--sport`). It runs CV, fits a final model and writes `models/xgboost_line_movement_<experiment.name>.pkl` plus a sibling `.yaml` holding the full config.
3. **Or tune first.** `odds train tune --config <file> --n-trials 100`; the config needs a `tuning` section ([tuner](hyperparameter-tuner.md)). The best parameters go to `<config>_best.yaml`. Then run step 2 on that file. `--train-best` also trains a model but leaves only a metadata `.yaml` beside the `.pkl`, which is not what the scorer reads.
4. **Check the artifacts.** Read the CV metrics and `experiments/results/<experiment.name>/verification/` (no all-NaN or constant features, expected feature count; see [run an experiment](../evaluation/howto-run-experiment.md)).
5. **Publish.** `odds model publish --name epl-clv-home --path models/<file>.pkl [--version v1.2] [--bucket <bucket>]` (`packages/odds-cli/odds_cli/commands/model.py::publish_model`). It requires a config YAML beside the pickle (`x.yaml`, via `with_suffix`), else exits 1; version defaults to the short git SHA, else a UTC timestamp; `metadata.json` holds `model_name`, `version`, `git_sha`, `training_date`, `strategy_type`; it uploads to the versioned prefix first, then `latest/`. It needs AWS credentials.
6. **Confirm scoring.** Check with `aws s3 ls s3://<bucket>/<name>/latest/` and the `model_loaded` log line.

## Layout and loading

The bucket holds `{model_name}/latest/` and `{model_name}/{version}/`, each with `model.pkl`, `config.yaml`, `metadata.json`. The loader reads only `latest/`, so `latest` is overwritten on every publish; keep the version label if you may need to roll back.

`packages/odds-lambda/odds_lambda/model_loader.py::load_model(model_name, bucket)`:

1. Name and bucket default to `packages/odds-core/odds_core/config.py::ModelConfig` (`MODEL_NAME`, `MODEL_BUCKET`; [configuration](../foundations/configuration.md)); missing values raise `ValueError`, a missing key raises `FileNotFoundError`.
2. `head_object` on `latest/model.pkl` for the ETag. If it equals the module-level cached ETag and key, return the cached dict (warm Lambda or long-lived process).
3. Otherwise download to the temp dir, `joblib.load`, then fetch `config.yaml` and build `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig` (copying `training.data.sport_key` into it). A missing `config.yaml` gives `feature_config=None` with a warning.

`model_supports_sport(sport)` loads the model and compares its `sport_key`; load errors return false. `packages/odds-lambda/odds_lambda/model_loader.py::get_cached_version` returns the ETag; `packages/odds-lambda/odds_lambda/model_loader.py::clear_cache` resets it.

## Inline scoring

`packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events` scores snapshots of upcoming events (multi-feature-group models, including the standings, match_stats and epl_schedule caches, via `packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter`; PR #359) and writes `packages/odds-core/odds_core/prediction_models.py::Prediction` rows. It is not scheduled on its own: the [fetch-oddsportal job](../collection/oddsportal.md) calls it after ingest when `model_supports_sport(ctx.sport)`. `main(ctx)` is also registered as `score-predictions`, so `odds scheduler run-once score-predictions-epl` runs it by hand ([jobs](../scheduling/jobs-and-deployment-matrix.md)).

1. `packages/odds-lambda/odds_lambda/model_loader.py::load_model` fetches the model. `ValueError` or `FileNotFoundError` returns zero stats with an error log.
2. The bundled `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig` supplies `sport_key`. If the `sport` argument differs, the run aborts (`sport_mismatch`).
3. `XGBoostAdapter().feature_names(config)` must equal the model's saved `feature_names`, else `feature_name_mismatch` and abort, so the variance filter's dropped columns matter ([feature pipeline](feature-pipeline.md)).
4. For each SCHEDULED future event of that sport: select snapshots with no prediction yet for this `model_name`, collect event data once, then per snapshot `adapter.transform`, `model.predict`, and insert.
5. Insert is `INSERT ... ON CONFLICT DO NOTHING` on constraint `uq_prediction_event_snap_model`, so reruns are safe. `snapshots_skipped` counts snapshots the adapter returns `None` for.

Returned counts: `events_checked`, `snapshots_scored`, `snapshots_skipped`, `errors`.

`packages/odds-core/odds_core/prediction_models.py::Prediction` (`predictions`): `event_id`, `snapshot_id` (both FKs), `model_name`, `model_version` (the S3 ETag from `packages/odds-lambda/odds_lambda/model_loader.py::get_cached_version`, or `"unknown"`), `predicted_clv`, `created_at`, unique on `(event_id, snapshot_id, model_name)`, indexed on `(model_name, created_at)`. Each snapshot is scored, so one event accumulates a prediction per snapshot; consumers pick the latest (the [daily digest](../operations/reporting-and-daily-digest.md) uses the highest `snapshot_id`), and the MCP `packages/odds-mcp/odds_mcp/server.py::get_predictions` tool returns stored rows only ([MCP market tools](../agent/mcp-market-tools.md)). Predictions exist only where scoring runs.

## Hazards

- **Scoring runs only on a local scheduler.** It needs `odds_analytics` and xgboost; the scraper Lambda image has neither and `scraper.tf` sets no `MODEL_*`, so publishing a model does not enable scoring on AWS ([AWS infrastructure](../operations/aws-infrastructure.md), [local scheduler](../scheduling/local-scheduler.md)).
- **Failures are silent.** Every failure path inside `packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events` returns normally and the call site in fetch-oddsportal also swallows exceptions, so a broken model load never fails the scrape job and raises no alert.
- **Undeclared dependency.** `packages/odds-lambda/odds_lambda/model_loader.py` imports `odds_analytics` and `boto3` at module level, so it cannot be imported in the scraper image ([packages and CLI](../foundations/packages-and-cli.md)).
- **Bucket default is a literal.** `odds model publish` defaults `--bucket` to `odds-models-685946576110`, not `MODEL_BUCKET`; it is not derived from the caller's account (`packages/odds-cli/odds_cli/commands/model.py::publish_model`). Pass `--bucket` for any other account.
- **`tune --train-best` artifacts are not scorable.** The sibling `.yaml` has no `training` key, so the loader builds a default `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig` without `sport_key` and scoring refuses to run (read from code; not run).
- **`model_version` is an ETag**, not the version label given to `odds model publish`.
- **Scoring uses the bundled config.** Feature groups, sharp roster and `feature_names` must match what training produced.
- **One model in memory.** The cache holds one model; loading a second model name evicts the first.
- **Publish validates little.** `publish` only checks that the YAML exists; its `training.features` block is parsed first at load time by `FeatureConfig(**...)`.

## Related

- [Feature pipeline](feature-pipeline.md), [CLV target](clv-target.md), [Walk-forward CV](decision-walk-forward-cv.md)
