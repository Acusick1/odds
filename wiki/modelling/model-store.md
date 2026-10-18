---
type: reference
title: Model store
description: odds model publish, the S3 layout and bucket, load_model caching and model_supports_sport.
sources:
  - odds_cli.commands.model.publish_model
  - odds_lambda.model_loader.load_model
  - odds_lambda.model_loader.get_cached_version
  - odds_lambda.model_loader.clear_cache
  - odds_lambda.model_loader.model_supports_sport
  - odds_core.config.ModelConfig
  - deployment/aws/bootstrap/s3_models.tf
status: stable
---

# Model store

Trained CLV models are published to a shared, versioned S3 bucket and loaded by inline scoring ([prediction scoring](prediction-scoring.md)). The bucket is created in `deployment/aws/bootstrap/s3_models.tf` (versioning on, AES256, public access blocked, `prevent_destroy`) so a dev `terraform destroy` cannot delete it ([AWS infrastructure](../operations/aws-infrastructure.md)). The deployed model and its metrics are on [CLV model findings](../evaluation/clv-model-findings.md#deployed-model). The live S3 object is the authority and cannot be verified from the repo; check with `aws s3 ls s3://<model-bucket>/epl-clv-home/`.

## `odds model publish`

`odds model publish --name epl-clv-home --path models/<file>.pkl [--version v1.2] [--bucket <bucket>]` (`odds_cli.commands.model.publish_model`):

- It requires a config YAML beside the pickle (`x.yaml`, via `with_suffix`), else exits 1. That is the only validation: the `training.features` block is first parsed at load time by `odds_analytics.training.config.FeatureConfig`, which rejects feature group names it does not define, so a config naming a removed group fails there.
- The version defaults to the short git SHA, else a UTC timestamp.
- `metadata.json` holds `model_name`, `version`, `git_sha`, `training_date`, `strategy_type`.
- It uploads to the versioned prefix first, then `latest/`, and needs AWS credentials.
- `--bucket` defaults to the literal `odds-models-685946576110`, not `MODEL_BUCKET`, and is not derived from the caller's account (#444).

## Layout and loading

The bucket holds `{model_name}/latest/` and `{model_name}/{version}/`, each with `model.pkl`, `config.yaml`, `metadata.json`. The loader reads only `latest/`, which every publish overwrites.

`odds_lambda.model_loader.load_model(model_name, bucket)`:

1. Name and bucket default to `odds_core.config.ModelConfig` (`MODEL_NAME`, `MODEL_BUCKET`; [configuration](../foundations/configuration.md)); missing values raise `ValueError`, a missing key raises `FileNotFoundError`.
2. `head_object` on `latest/model.pkl` for the ETag. If it equals the module-level cached ETag and key, return the cached dict (warm Lambda or long-lived process).
3. Otherwise download to the temp dir, `joblib.load`, then fetch `config.yaml` and build `odds_analytics.training.config.FeatureConfig` (copying `training.data.sport_key` into it). A missing `config.yaml` gives `feature_config=None` with a warning.

The cache holds one model, so loading a second model name evicts the first. `odds_lambda.model_loader.model_supports_sport` loads the model and compares its `sport_key`; load errors return false. `odds_lambda.model_loader.get_cached_version` returns the ETag; `odds_lambda.model_loader.clear_cache` resets it. `odds_lambda.model_loader` imports `odds_analytics` and `boto3` at module level ([workspace packages](../foundations/workspace-packages.md), #454).

## Related

- [Train and publish a model](howto-train-and-publish-model.md), [Prediction scoring](prediction-scoring.md)
