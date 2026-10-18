---
type: reference
title: The daily-digest job
description: Queries, embed layout, delivery, schedule and heartbeat of odds_lambda.jobs.daily_digest.
sources:
  - odds_lambda.jobs.daily_digest
  - odds_lambda.jobs.daily_digest.send_digest
  - odds_lambda.jobs.daily_digest.build_digest_embed
  - odds_lambda.jobs.daily_digest._get_completed_events_with_predictions
  - odds_lambda.jobs.daily_digest._get_upcoming_events_with_predictions
  - odds_lambda.jobs.daily_digest._value_side
  - odds_lambda.jobs.daily_digest._result_hit
  - odds_lambda.jobs.daily_digest.MAX_FIELD_CHARS
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
status: stable
---

# The daily-digest job

`odds_lambda.jobs.daily_digest` posts one Discord embed per sport with model predictions. It is opt-in and local-only: no EventBridge rule exists, it is commented out of the default `bootstrap_jobs`, and when listed in `SCHEDULER_BOOTSTRAP_JOBS` it runs as the fixed cron in its `_JOB_CRON_MAP` entry at `0 8 * * *`, for EPL only through its `_JOB_SPORTS` allowlist. `odds scheduler smoke --no-side-effects` drops it ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs), [local-scheduler](../scheduling/local-scheduler.md)). It writes heartbeat key `daily-digest-epl`, but `heartbeat_expectations` has no entry for it, so a missed digest does not alert ([alerting-and-heartbeats](alerting-and-heartbeats.md#heartbeats)).

## Flow

`main(ctx)` runs `odds_lambda.jobs.daily_digest.send_digest` inside `job_alert_context("daily-digest-<suffix>")`, with `ctx.lookback_hours` (24) and `ctx.lookahead_hours` (48).

- Completed: `odds_lambda.jobs.daily_digest._get_completed_events_with_predictions` selects FINAL events with `completed_at >= now - lookback`, joined to the prediction with the highest `snapshot_id` per event for `model_name`, ordered by kickoff.
- Upcoming: `odds_lambda.jobs.daily_digest._get_upcoming_events_with_predictions` selects SCHEDULED events kicking off in `(now, now + lookahead]` with the same join, ordered by `|predicted_clv|` descending.
- `odds_lambda.jobs.daily_digest.build_digest_embed` makes a green embed titled `<EPL|NBA|MLB> Daily Digest` with fields "Post-Match Results" (tick or cross, score, side and CLV, footer `X events | Y/X correct side`) and "Upcoming Predictions". Each field is truncated at `odds_lambda.jobs.daily_digest.MAX_FIELD_CHARS` (1024).
- Delivery is `odds_core.alerts.AlertManager.send_embed`. If both lists are empty nothing is sent and the log says `daily_digest_empty`. `sent` in the returned stats is 1 whenever an embed was built, including when alerts are disabled or the post failed (#384).

`model_name` defaults to `settings.model.name`, falling back to `"epl-clv-home"`.

## Conventions

- The digest is single-model and home-side: `odds_lambda.jobs.daily_digest._value_side` treats positive predicted CLV as "Home" and negative as "Away", and `odds_lambda.jobs.daily_digest._result_hit` counts a draw as a miss for both sides. There is no draw side and no multi-model digest.
- It is EPL-only by allowlist: `_SPORT_DISPLAY_NAMES` is hard-coded for EPL, NBA, MLB, and the `_JOB_SPORTS` allowlist limits the job to EPL.
- It never reads paper trades or P&L ([reporting-surfaces](reporting-surfaces.md)).

## Related

- [Reporting surfaces](reporting-surfaces.md)
- [Alerts and the alert_history table](alerts.md)
- [Local scheduler](../scheduling/local-scheduler.md)
