---
type: reference
title: Prediction scoring
description: The score_events flow, the predictions table, where scoring runs and how failures surface.
sources:
  - odds_lambda.jobs.score_predictions.score_events
  - odds_lambda.model_loader.load_model
  - odds_analytics.feature_groups.XGBoostAdapter
  - odds_core.prediction_models.Prediction
  - odds_lambda.jobs.daily_digest
  - odds_mcp.server.get_predictions
status: stable
---

# Prediction scoring

`odds_lambda.jobs.score_predictions.score_events` scores snapshots of upcoming events with the published model ([model store](model-store.md)), handling multi-feature-group models, preloading the standings, match_stats and epl_schedule (fixtures) caches for the groups the config names, via `odds_analytics.feature_groups.XGBoostAdapter` (PR #359), and writes `odds_core.prediction_models.Prediction` rows.

## Where it runs

It is not scheduled on its own: the [fetch-oddsportal job](../collection/fetch-oddsportal-job.md) calls it after ingest when `model_supports_sport(ctx.sport)`. `main(ctx)` is also registered as `score-predictions`, so `odds scheduler run-once score-predictions-epl` runs it by hand ([job registry](../scheduling/job-registry.md)).

- The scraper image lacks `odds_analytics` and xgboost and `scraper.tf` sets no `MODEL_*`, so scoring runs only where a local scheduler runs `fetch-oddsportal` (#454; [local scheduler](../scheduling/local-scheduler.md), [AWS infrastructure](../operations/aws-infrastructure.md)).
- Every failure inside `odds_lambda.jobs.score_predictions.score_events` returns its stats dict normally (per-snapshot errors are only counted in `errors`) and the call site in fetch-oddsportal swallows exceptions, so a broken model never alerts and never fails the scrape job (#454).
- `odds_lambda.model_loader` imports `odds_analytics` at module level (see [model store](model-store.md)).

## Flow

1. `odds_lambda.model_loader.load_model` fetches the model. `ValueError` or `FileNotFoundError` returns zero stats with an error log.
2. The bundled `odds_analytics.training.config.FeatureConfig` supplies `sport_key`. If the `sport` argument differs, the run aborts (`sport_mismatch`). Scoring uses the bundled config, so feature groups, sharp roster and the feature names must match what training produced; a bundled config naming an unknown feature group fails `odds_analytics.training.config.FeatureConfig` validation when the loader builds it. A bundled `retail_ev` config (either `retail_ev_arm`) validates, with or without its as-of-t groups, but scoring builds `odds_analytics.feature_groups.XGBoostAdapter` features per snapshot, not retail-EV rows, so the run aborts with `feature_name_mismatch` before any prediction is written.
3. `odds_analytics.feature_groups.XGBoostAdapter.feature_names` on the config must equal the model's saved feature names, else `feature_name_mismatch` and abort, so the variance filter's dropped columns matter ([feature pipeline](feature-pipeline.md#nan-fill-and-variance-policy)).
4. For each SCHEDULED future event of that sport: select snapshots with no prediction yet for this `model_name`, collect event data once, then per snapshot `adapter.transform`, `model.predict`, and insert.
5. Insert is `INSERT ... ON CONFLICT DO NOTHING` on constraint `uq_prediction_event_snap_model`, so reruns are safe. `snapshots_skipped` counts snapshots the adapter returns `None` for.

Returned counts: `events_checked`, `snapshots_scored`, `snapshots_skipped`, `errors`.

## Predictions table

`odds_core.prediction_models.Prediction` (`predictions`): `event_id`, `snapshot_id` (both FKs), `model_name`, `model_version`, `predicted_clv`, `created_at`, unique on `(event_id, snapshot_id, model_name)`, indexed on `(model_name, created_at)`.

- `model_version` is the S3 ETag from `odds_lambda.model_loader.get_cached_version`, or `"unknown"`, not the version label given to `odds model publish`.
- Each snapshot is scored, so one event accumulates a prediction per snapshot; consumers pick the latest (the [daily digest](../operations/daily-digest.md) uses the highest `snapshot_id`), and the MCP `odds_mcp.server.get_predictions` tool returns stored rows only ([MCP market tools](../agent/mcp-market-tools.md)).
- Predictions exist only where scoring runs.

## Related

- [Model store](model-store.md), [Feature pipeline](feature-pipeline.md), [CLV target](clv-target.md)
