---
type: reference
title: The fetch-odds job
description: Event sync, gate, pre/post-schedule and log events of odds_lambda.jobs.fetch_odds.
sources:
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_odds.SOURCE
  - odds_lambda.jobs.fetch_odds.NBA_CADENCE
  - odds_lambda.jobs.fetch_odds.NBA_LEADS
  - odds_lambda.jobs.fetch_odds.NBA_QUIET_HOURS
  - odds_lambda.jobs.fetch_odds.NBA_SOURCE
  - odds_lambda.jobs.fetch_odds.FetchPlan
  - odds_lambda.jobs.fetch_odds.PLANS
  - odds_lambda.scheduling.helpers.next_slate_start
  - odds_lambda.scheduling.helpers.get_next_slate_start
  - odds_lambda.scheduling.helpers.main_tip
  - odds_lambda.scheduling.helpers.get_next_slate_main_tip
  - odds_lambda.scheduling.helpers.next_slate_main_tip
  - odds_lambda.event_sync.EventSyncService
  - odds_lambda.scheduling.helpers.get_next_kickoff
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
  - odds_core.config.SchedulerConfig
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# The fetch-odds job

`odds_lambda.jobs.fetch_odds.main(ctx)` is the scheduled caller of [the ingestion service](odds-ingestion-service.md). It is in the default local scheduler `bootstrap_jobs` and runs on the production host as `fetch-odds-epl` and `fetch-odds-nba`; its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist limits it to those two sports to save Odds API quota, each on its own key (the other bootstrapped collectors, including `fetch-polymarket-odds` and `fetch-injuries`, and `fetch-mlb-results` spend none; [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs), [odds-api-quota](odds-api-quota.md)). Its AWS rule (`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend`, a one-time cron in UTC) is dormant.

## Flow

It requires `ctx.sport` (a missing sport, or one with no entry in `odds_lambda.jobs.fetch_odds.PLANS`, raises `ValueError` before any work) and runs for that one sport. The sport's `odds_lambda.jobs.fetch_odds.FetchPlan` holds its cadence, leads, overnight window, kickoff lookup and the source its empty-run check uses:

1. **Event sync**: `odds_lambda.event_sync.EventSyncService.sync_sports` calls the free `/events` endpoint and resolves each listed game (an `odds_core.api_models.ApiEvent`, which carries the API's id) to its event (`odds_lambda.odds_api_events.resolve_odds_api_events`, [Odds API events](../storage/event-matching.md#odds-api-events)): a game another source stored is matched and given the API's id, and one with no event creates it. It runs on every invocation, before the gate, so `odds_lambda.scheduling.helpers.get_next_kickoff` sees fixtures. An existing event keeps its integer id, status, scores, teams, MLB `gamePk` and acknowledgements ([events](../storage/events.md#writing-events)).
2. **Gate**: `decide_forward_resilient(sports, plan.cadence, ...)` with the plan's overnight window, kickoff lookup and leads, and `lookahead_days=settings.scheduler.lookahead_days`. EPL's cadence is `odds_lambda.jobs.fetch_odds.CADENCE` ([decision-engine](../scheduling/decision-engine.md#cadence)); it polls every 2 h within 3 h of a kickoff and every 3 h from 3 to 12 h to fit the free-tier quota ([odds-api-quota](odds-api-quota.md#scheduled-spend)), with no overnight window, leads or custom kickoff lookup. NBA's is [below](#nba-schedule). fetch-odds is the only job that takes `lookahead_days` from `odds_core.config.SchedulerConfig` (the settings that also name the scheduler backend, dry-run flag and bootstrap jobs).
3. **Pre-schedule** under the compound name; a failed booking sends a `scheduling_warning:<job>` warning when another booking is still to come or has succeeded, and a `scheduling_failure:<job>` critical when none survives (a skipped run's only booking, or a post-fetch booking after a failed pre-schedule; [alerts](../operations/alerts.md#delivery)). The rest of the run is inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) under that name, so a gated run also writes a heartbeat.
4. **Skip** if the gate says so; otherwise **ingest**: `odds_lambda.ingestion.OddsIngestionService.ingest_sports`, then `odds_lambda.live_events.alert_dropped_games` for each sport's dropped MLB games.
5. **Quota alerts**: none; the health monitor is the only writer ([odds-api-quota](odds-api-quota.md)).
6. **Empty-run check**: `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a run that stored no snapshot while a SCHEDULED fixture lies within the plan's horizon: 48 hours for EPL (`odds_lambda.jobs.fetch_odds.SOURCE`) and 14 hours for NBA (`odds_lambda.jobs.fetch_odds.NBA_SOURCE`, so the morning run, about 13 h before a 23:00 tip-off, is covered); it passes no `accounted_by_sport`, and the failure handler below sends the critical. NBA is not in `odds_lambda.jobs.fetch_odds.SOURCE`, so the stale-source check, whose threshold follows EPL's `odds_lambda.jobs.fetch_odds.CADENCE`, does not cover it; the empty-run check and the heartbeat do ([health-monitor](../operations/health-monitor.md#stale-data-check)).
7. **Post-schedule** after re-querying the next kickoff.

## Failure behaviour

- A raised run sends the `job_failure:fetch-odds-epl` critical once per outage (the first failure since its last heartbeat, then a reminder every `odds_core.config.AlertConfig.stale_reminder_minutes`, 360 min) through `odds_core.alerts.job_alert_context` and re-raises; a clean or gated run writes `heartbeat:fetch-odds-epl`, which `odds_core.config.AlertConfig.heartbeat_expectations` expects every 49 h for EPL (the 48 h opening-tier interval plus 1 h) and every 35 h for `fetch-odds-nba` (its 24 h interval, pushed to 10:00 UTC by the quiet window for gaps of about 33 h, plus slack); EPL's run does not pause overnight, and NBA's cadence runs skip 00:00-10:00 UTC. The stale-source check leaves a source alone while that failure is recent ([health-monitor](../operations/health-monitor.md#stale-data-check)). On AWS the Lambda handler sends a second critical (#442; [lambda-handler](../scheduling/scheduler-backends.md#lambda-handler)).
- On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted. A failed run is re-booked 15 minutes later only when it booked nothing, and the pre-schedule has already booked the next tier interval (30 minutes at the closing tier), so a failing API call is retried at the normal cadence, not every 15 minutes ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
- A run without a sport raises `ValueError`, and `odds scheduler run-once fetch-odds` fails the same way; use `fetch-odds-epl` or `fetch-odds-nba`.
- A 401 or 429 fails the run with the response error; it does not try another key ([keys](odds-api-client.md#retry-and-keys)), and a sport whose key index has no key fails before any request.

## NBA schedule

NBA's second key has 500 units a month (3 units a call, about 160 calls), and one call prices the whole slate, so the schedule is per slate and not per game ([odds-api-quota](odds-api-quota.md#nba-spend)). A slate is the games each less than 6 h after the one before (`odds_lambda.scheduling.helpers.next_slate_start`, read from stored events by `odds_lambda.scheduling.helpers.get_next_slate_start`); the plan passes it as the kickoff lookup and the first-tip leads' lookup, so the cadence and those leads see a slate's first tip-off and games already under way never start a slate.

- Leads (`odds_lambda.jobs.fetch_odds.NBA_LEADS`): runs 3 h and 1 h before the slate's first tip-off, and 1 h before its main tip-off (`odds_lambda.scheduling.helpers.main_tip`: the kickoff most of the slate's games share, the latest of a tie; `odds_lambda.scheduling.helpers.get_next_slate_main_tip`). The main-tip lead has no run when the main tip is less than 90 minutes after the first, since the first-tip runs already price those games. A late lead run is still taken until 2 h (3 h lead) or 20 min (1 h leads) before its tip-off, and `odds_lambda.jobs.fetch_odds.main` passes the start of the run to the decision (`run_started`), so a run that began before a lead time and ended after it does not count as that lead's run ([lead runs](../scheduling/decision-engine.md#lead-runs)).
- Cadence (`odds_lambda.jobs.fetch_odds.NBA_CADENCE`): 12 h in the closing, pregame and sharp tiers, 24 h otherwise and when no game is within the lookahead. The run after the 1 h lead run falls 12 h later, at about 10:00 UTC, the morning run; a cadence run that would fall between 00:00 and 10:00 UTC (`odds_lambda.jobs.fetch_odds.NBA_QUIET_HOURS`) moves to 10:00, and leads ignore the window.
- A weeknight slate whose games mostly tip 1 h after its first (first tip 23:00, main 00:00 UTC) runs at 10:00, 20:00 and 22:00; a weekend slate that tips at 17:00 with most games at 20:00 runs at 10:00, 14:00, 16:00 and 19:00. The monthly cost, replayed over stored events, is in [odds-api-quota](odds-api-quota.md#nba-spend). Games after the main tip are priced at its run, and the closing prices come from [OddsPortal](fetch-oddsportal-job.md) and [Betfair](betfair-exchange.md).

NBA games the Odds API lists resolve to the events OddsPortal and Betfair stored, by team names and a start within 2 h, so a game is one event with an `odds_api` external id added ([event-matching](../storage/event-matching.md#odds-api-events)). The API's NBA team names are the canonical ones.

## Log events

`fetch_odds_job_started`, `fetch_odds_executing`, `fetch_odds_next_scheduled`, `fetch_odds_scheduling_failed` (`odds_lambda.jobs.fetch_odds.main`), `odds_snapshot_stored` (`odds_lambda.storage.writers.OddsWriter.store_envelope`, one per envelope stored, for an integer event id, with its bookmaker count and the number of snapshots written) and `eventbridge_rule_updated` (`odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._schedule_with_retry`).

## Related

- [Odds API ingestion service](odds-ingestion-service.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md)
