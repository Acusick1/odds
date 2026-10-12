---
type: component
title: S3 model store and loader
summary: The versioned model bucket layout, `odds model publish`, and ETag-cached model loading for scoring.
sources:
  - docs/DEPLOYMENT.md#s3-model-store
  - docs/CLI.md#model-management
  - docs/CLI.md#publish-model-to-s3
  - docs/MODELING.md#deployed-model
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - packages/odds-cli/odds_cli/commands/model.py::publish_model
  - packages/odds-core/odds_core/config.py::ModelConfig
  - deployment/aws/bootstrap/s3_models.tf
verified_at: 7887039
---

# S3 model store and loader

Trained CLV models are published to a shared, versioned S3 bucket and loaded by [inline scoring](job-score-predictions.md). The bucket is created in `deployment/aws/bootstrap/s3_models.tf` (versioning on, AES256, public access blocked, `prevent_destroy`) so a dev `terraform destroy` cannot delete it ([AWS infrastructure](aws-infrastructure.md)).

## Layout

`{model_name}/latest/` and `{model_name}/{version}/`, each with `model.pkl`, `config.yaml`, `metadata.json`. The loader reads only `latest/`.

## Publishing

`odds model publish --name N --path models/x.pkl [--version V] [--bucket B]` (`packages/odds-cli/odds_cli/commands/model.py::publish_model`):

- Requires a config YAML beside the pickle (`x.yaml`, via `with_suffix`), else exits 1.
- Version defaults to the short git SHA, else a UTC timestamp.
- `metadata.json` holds `model_name`, `version`, `git_sha`, `training_date`, `strategy_type`.
- Uploads to the versioned prefix first, then `latest/`.

Training and publishing end to end: [train and publish a model](howto-train-and-publish-model.md). The deployed model's details (`epl-clv-home`, XGBoost, walk-forward CV) are in [CLV model findings](clv-model-findings.md); the live S3 object is the authority.

## Loading

`packages/odds-lambda/odds_lambda/model_loader.py::load_model(model_name, bucket)`:

1. Name and bucket default to `ModelConfig` (`MODEL_NAME`, `MODEL_BUCKET`); missing values raise `ValueError`, a missing key raises `FileNotFoundError`.
2. `head_object` on `latest/model.pkl` for the ETag. If it equals the module-level cached ETag and key, return the cached dict (warm Lambda or long-lived process).
3. Otherwise download to the temp dir, `joblib.load`, then fetch `config.yaml` and build `FeatureConfig` (copying `training.data.sport_key` into it). A missing `config.yaml` gives `feature_config=None` with a warning.

`model_supports_sport(sport)` loads the model and compares its `sport_key`; load errors return false. `get_cached_version()` returns the ETag; `clear_cache()` resets it.

## Hazards

- `odds model publish` defaults `--bucket` to the literal `odds-models-685946576110`, not `MODEL_BUCKET`; `docs/CLI.md` describes it as `odds-models-{account_id}`.
- The loader module imports `odds_analytics` and `boto3` at import time, so it cannot be imported in the scraper Lambda image ([fetch-oddsportal](job-fetch-oddsportal.md)).
- The in-memory cache holds one model; loading a second model name evicts the first.
- `publish` only checks that the YAML exists; its `training.features` block is parsed first at load time by `FeatureConfig(**...)`, not at publish time.

## Related

- [Feature pipeline](feature-pipeline.md), [configuration](configuration.md).
