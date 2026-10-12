---
type: howto
title: Train, tune and publish the CLV model
summary: Write a config, run odds train run or tune --train-best, check the artifacts, run odds model publish, and confirm scoring picks the model up.
sources:
  - packages/odds-cli/odds_cli/commands/train.py::run_training
  - packages/odds-cli/odds_cli/commands/train.py::run_tuning
  - packages/odds-cli/odds_cli/commands/model.py::publish_model
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - docs/CLI.md#train-a-model
  - docs/CLI.md#tune-hyperparameters
  - docs/CLI.md#publish-model-to-s3
  - docs/DEPLOYMENT.md#s3-model-store
verified_at: 7887039
---

# Train, tune and publish the CLV model

1. **Write a config.** Start from `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` ([training-config](training-config.md)). Set `target_type: devigged_bookmaker` and `target_bookmaker` explicitly (code defaults are `raw` and `pinnacle`), `sport_key`, and the protocol from [decision-cv-protocol](decision-cv-protocol.md). Run `odds train validate --config <file>`.
2. **Train.** `odds train run --config <file>` (options: `--output`, `--dry-run`, `--verbose`, `--track`, `--tracking-uri`, `--sport`). It runs CV, fits a final model and writes `models/xgboost_line_movement_<experiment.name>.pkl` plus a sibling `.yaml` holding the full config ([training-pipeline](training-pipeline.md)).
3. **Or tune first.** `odds train tune --config <file> --n-trials 100`; the config needs a `tuning` section ([hyperparameter-tuner](hyperparameter-tuner.md)). The best parameters go to `<config>_best.yaml`. Then run step 2 on that file. `--train-best` also trains a model, but leaves only a metadata `.yaml` beside the `.pkl`, which is not what the scorer reads.
4. **Check the artifacts.** Read the CV metrics in the output and `experiments/results/<experiment.name>/verification/` (no all-NaN or constant features, expected feature count; see [howto-run-experiment](howto-run-experiment.md)).
5. **Publish.** `odds model publish --name epl-clv-home --path models/<file>.pkl [--version v1.2] [--bucket <bucket>]`.
   - Needs the `.yaml` next to the `.pkl`; uploads `model.pkl`, `config.yaml` and `metadata.json` to `{name}/{version}/` and `{name}/latest/`.
   - Version defaults to the short git SHA, else a timestamp.
   - Requires AWS credentials ([s3-model-store](s3-model-store.md)).
6. **Confirm scoring.** `load_model` reads `{name}/latest/` using `MODEL_NAME` and `MODEL_BUCKET` (or the arguments), caches by ETag, and takes the feature config and `sport_key` from the uploaded `config.yaml`. Check with `aws s3 ls s3://<bucket>/<name>/latest/` and the `model_loaded` log line, then see [job-score-predictions](job-score-predictions.md). The deployed S3 state cannot be verified from the repo.

## Hazards

- **Bucket default is a literal.** `odds model publish` defaults `--bucket` to `odds-models-685946576110`; `docs/CLI.md` documents `odds-models-{account_id}`. Pass `--bucket` for any other account.
- **`tune --train-best` artifacts are not scorable.** The sibling `.yaml` has no `training` key, so the loader builds a default `FeatureConfig` without `sport_key` and scoring refuses to run (read from code; not run).
- **Scoring uses the bundled config.** Feature groups, sharp roster and `feature_names` must match what training produced, including columns dropped by the variance filter ([feature-pipeline](feature-pipeline.md)).
- **`latest` is overwritten** on every publish; keep the version label if you may need to roll back ([clv-model-findings](clv-model-findings.md)).

## Related

- [Deployed model](clv-model-findings.md), [Walk-forward CV](walk-forward-cv.md)
