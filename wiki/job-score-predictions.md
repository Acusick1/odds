---
type: component
title: Inline CLV scoring and predictions table
summary: "score_events() at the end of fetch-oddsportal: S3 model load, feature extraction and idempotent writes to the `predictions` table."
sources:
  - docs/ARCHITECTURE.md#score-predictions-inline
  - docs/DATABASE.md#prediction-table
  - docs/REPORTING.md#8-storage-backing-reports
  - packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events
  - packages/odds-lambda/odds_lambda/model_loader.py::load_model
  - packages/odds-analytics/odds_analytics/feature_groups.py::XGBoostAdapter
  - packages/odds-core/odds_core/prediction_models.py::Prediction
verified_at: 7887039
---

# Inline CLV scoring and predictions table

`packages/odds-lambda/odds_lambda/jobs/score_predictions.py::score_events` scores snapshots of upcoming events with the deployed CLV model and writes `Prediction` rows. It is not scheduled on its own: [fetch-oddsportal](job-fetch-oddsportal.md) calls it after ingest when `model_supports_sport(ctx.sport)`. `main(ctx)` still exists and is registered as `score-predictions`, so `odds scheduler run-once score-predictions-epl` runs it by hand ([job registry](job-registry.md)).

## Flow

1. `load_model` fetches the model from S3 with ETag caching ([S3 model store](s3-model-store.md)). `ValueError` (no `MODEL_NAME`/`MODEL_BUCKET`) or `FileNotFoundError` returns zero stats with an error log.
2. The bundled `FeatureConfig` supplies `sport_key`. If the `sport` argument differs from the model's sport, the run aborts (`sport_mismatch`).
3. `XGBoostAdapter().feature_names(config)` must equal the model's saved `feature_names`, else `feature_name_mismatch` and abort ([feature pipeline](feature-pipeline.md)).
4. For each SCHEDULED future event of that sport: select snapshots with no prediction yet for this `model_name`, collect event data once, then per snapshot `adapter.transform`, `model.predict`, and insert.
5. Insert is `INSERT ... ON CONFLICT DO NOTHING` on constraint `uq_prediction_event_snap_model`, so reruns are safe. `snapshots_skipped` counts snapshots the adapter returns `None` for.

Returned counts: `events_checked`, `snapshots_scored`, `snapshots_skipped`, `errors`.

## Table

`packages/odds-core/odds_core/prediction_models.py::Prediction` (`predictions`): `event_id`, `snapshot_id` (both FKs), `model_name`, `model_version` (the S3 ETag from `get_cached_version`, or `"unknown"`), `predicted_clv`, `created_at`, unique on `(event_id, snapshot_id, model_name)`, indexed on `(model_name, created_at)`. Readers: [daily digest](job-daily-digest.md) and the MCP `get_predictions` tool ([reporting surfaces](reporting-surfaces.md)). Predicted CLV is defined in [CLV](clv.md) and [CLV model findings](clv-model-findings.md).

## Hazards

- Every failure path inside `score_events` returns normally; as the call site in fetch-oddsportal also swallows exceptions, a broken model load never fails the scrape job and raises no alert.
- `model_version` is an S3 ETag, not the version label given to `odds model publish`.
- Scoring runs where `odds_analytics` and xgboost are installed. The scraper Lambda image has neither ([fetch-oddsportal hazards](job-fetch-oddsportal.md)).
- Each snapshot of an event is scored, so one event accumulates a prediction per snapshot; consumers must pick the latest (the digest uses the highest `snapshot_id`).

## Related

- [Table odds_snapshots](table-odds-snapshots.md), [self-scheduling](self-scheduling.md).
