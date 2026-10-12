---
type: component
title: daily-digest job
summary: A local-only EPL Discord embed of recent results and upcoming predictions, built on a single-model, home-side convention.
sources:
  - docs/REPORTING.md#4-daily-digest--packagesodds-lambdaodds_lambdajobsdaily_digestpy
  - docs/REPORTING.md#9-known-limitations
  - docs/ARCHITECTURE.md#daily-digest-job
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py::send_digest
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py::build_digest_embed
verified_at: 7887039
---

# daily-digest job

`packages/odds-lambda/odds_lambda/jobs/daily_digest.py` posts one Discord embed per sport with model predictions. It is local-only: no EventBridge rule exists, it is commented out of the default `bootstrap_jobs`, and when listed in `SCHEDULER_BOOTSTRAP_JOBS` it runs from `_JOB_CRON_MAP` at `0 8 * * *` for EPL only ([deployment matrix](job-deployment-matrix.md), [local scheduler](local-scheduler.md)).

## Flow

`main(ctx)` runs `send_digest` inside `job_alert_context("daily-digest-<suffix>")`, with `ctx.lookback_hours` (24) and `ctx.lookahead_hours` (48).

- Completed: `_get_completed_events_with_predictions` selects FINAL events with `completed_at >= now - lookback`, joined to the prediction with the highest `snapshot_id` per event for `model_name`, ordered by kickoff.
- Upcoming: `_get_upcoming_events_with_predictions` selects SCHEDULED events kicking off in `(now, now + lookahead]` with the same join, ordered by `|predicted_clv|` descending.
- `build_digest_embed` makes a green embed titled `<EPL|NBA|MLB> Daily Digest` with fields "Post-Match Results" (tick or cross, score, side and CLV, footer `X events | Y/X correct side`) and "Upcoming Predictions". Each field is truncated at `MAX_FIELD_CHARS` (1024).
- Delivery is `alert_manager.send_embed`. If both lists are empty nothing is sent and the log says `daily_digest_empty`.

`model_name` defaults to `settings.model.name`, falling back to `"epl-clv-home"`. Predictions come from [inline scoring](job-score-predictions.md).

## Hazards

- Home-side convention: `_value_side` treats positive predicted CLV as "Home" and negative as "Away"; `_result_hit` counts a draw as a miss for both sides. There is no draw side and no multi-model digest.
- The digest never reads paper trades or P&L ([paper trading](paper-trading.md)); those exist only on the MCP and CLI side.
- `sent` in the returned stats is set to 1 whenever an embed was built, even if alerts are disabled or the webhook post failed ([alerts module](alerts-module.md)).
- `_SPORT_DISPLAY_NAMES` is hard-coded for EPL, NBA, MLB, and the cron allowlist limits the job to EPL, because the heartbeat expectation `daily-digest-epl` (26 h) is EPL-only ([health monitor](health-monitor.md)).
- `_get_upcoming_events_with_predictions` also selects `OddsSnapshot.snapshot_time`, which the embed never uses.

## Related

- [Reporting surfaces](reporting-surfaces.md), [job registry](job-registry.md).
