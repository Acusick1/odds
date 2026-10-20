---
type: reference
title: Health monitor, check-health job and odds status
description: HealthMonitor, the check-health job's seven checks, heartbeat purge, its own alert path, and what odds status show, quota and events print and read.
sources:
  - odds_lambda.health_monitor.HealthMonitor
  - odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_stale_sources
  - odds_lambda.health_monitor.StaleSource
  - odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes
  - odds_lambda.health_monitor.MissingBetfairClose
  - odds_core.betfair_close.LIVE_CLOSE_WINDOW
  - odds_core.betfair_close.is_in_play
  - odds_core.betfair_close.complete_exchange_prices
  - odds_lambda.health_monitor.collection_sources
  - odds_lambda.collection_calendar.CollectionSource
  - odds_lambda.collection_calendar.fixture_calendar
  - odds_lambda.collection_calendar.active_hours_between
  - odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures
  - odds_lambda.health_monitor.HealthMonitor.check_api_quota
  - odds_lambda.health_monitor.HealthMonitor.check_event_identity
  - odds_lambda.health_monitor.HealthMonitor.check_schedule_chains
  - odds_lambda.health_monitor.DeadChain
  - odds_lambda.health_monitor.SCHEDULE_OVERDUE_GRACE
  - odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_outstanding_job_names
  - odds_lambda.health_monitor.check_system_health
  - odds_lambda.health_monitor.HealthMetrics
  - odds_lambda.health_monitor.HealthStatus
  - odds_lambda.jobs.check_health.main
  - odds_core.config.AlertConfig
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.config.BetfairConfig.enabled
  - odds_lambda.scheduling.jobs.resolve_target_sports
  - odds_core.models.FetchLog
  - odds_cli.commands.status
  - odds_cli.commands.status._show_quota
  - odds_cli.commands.scheduler.health
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_lambda.storage.readers.OddsReader.get_fetch_logs
status: stable
---

# Health monitor, check-health job and odds status

`odds_lambda.jobs.check_health.main` runs `odds_lambda.health_monitor.check_system_health` inside `job_alert_context("check-health")` and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; see [self-scheduling](../scheduling/self-scheduling.md)). It is global and in the default production `odds_core.config.SchedulerConfig.bootstrap_jobs`, so it runs on the mini-PC against the production database, where every job's heartbeat lands ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Nothing alerts when the host itself stops (#489). `odds_lambda.health_monitor.check_system_health` collects `odds_lambda.health_monitor.HealthMetrics`, then runs, in order:

1. `odds_lambda.health_monitor.HealthMonitor.check_stale_sources`: per collection source and sport, active hours without a new snapshot versus a threshold derived from the source's cadence, only while the database has fixtures the source should price. Each stale source sends its own critical `stale_data:<source>:<sport>` ([Stale-data check](#stale-data-check)). It is followed by the MLB schedule check and by `odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes`, one critical `missing_betfair_close` listing EPL events whose Betfair close was not collected ([Betfair close check](#betfair-close-check)).
2. `odds_lambda.health_monitor.HealthMonitor.check_consecutive_failures`: leading run of `success=False` among the latest 10 `odds_core.models.FetchLog` rows versus `consecutive_failures_threshold` (3). It reads rows of any sport or source; API outages write no row (#472) and `odds_lambda.jobs.fetch_betfair_exchange` writes `success=False` when a run stores no snapshots, so those runs count as failures ([fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).
3. `odds_lambda.health_monitor.HealthMonitor.check_api_quota`: `api_quota_remaining` of the latest row that has one, divided by `settings.api.quota`; below 0.1 alert `quota_critical`, below 0.2 `quota_low`. This is the same fraction fetch-odds applies, but rate-limited ([odds-api-quota](../collection/odds-api-quota.md)).
4. `odds_lambda.health_monitor.HealthMonitor.check_data_quality`: `error` plus `critical` `odds_core.models.DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10). The validator itself only emits `warning` and `error`, so `critical` never occurs ([odds-validator](../storage/odds-validator.md)).
5. `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`: compares the newest `heartbeat:<job>` row per key in `odds_core.config.AlertConfig.heartbeat_expectations` to its allowed hours. The default keys and allowed hours are in [Heartbeats](#heartbeats). A job with no history is not alerted (#384).
6. `odds_lambda.health_monitor.HealthMonitor.check_event_identity`: per sport, matches stored as several events, EPL fixtures stored twice in a season, and events still `scheduled` a day after a kick-off in the last seven days, each its own warning (`duplicate_events:<sport>`, `repeated_fixtures:<sport>`, `stale_scheduled_events:<sport>`). Nothing is merged; `odds events merge-twins --sport <sport> --apply` clears twins ([Event merge](../storage/event-merge.md#detection)).
7. `odds_lambda.health_monitor.HealthMonitor.check_schedule_chains`: dead self-scheduling chains, each its own critical `dead_schedule_chain:<job>` ([Schedule chain check](#schedule-chain-check)).

Then `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` runs on every health check: it deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7) and is the only retention sweep in the system.

## Schedule chain check

`odds_lambda.health_monitor.HealthMonitor.check_schedule_chains` catches a self-scheduling job whose chain has ended, before its data goes stale. It runs only when `odds_lambda.health_monitor.check_system_health` is given the local scheduler store (`SCHEDULER_BACKEND=local`); a `odds_lambda.health_monitor.HealthMonitor` built without one skips it. It covers every compound job that `odds_core.config.SchedulerConfig.bootstrap_jobs` expands to, except:

- cron jobs, which APScheduler re-fires on its own;
- jobs that return before booking a run while disabled (Betfair while `odds_core.config.BetfairConfig.enabled` is false, both Polymarket jobs while Polymarket is disabled);
- jobs with a released, unfinished APScheduler job under their schedule id (`odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_outstanding_job_names`). A job that books its next run only after its work, such as check-health itself, has no schedule while it runs.

A covered job is dead when it has no stored schedule, or when its next run is more than `odds_lambda.health_monitor.SCHEDULE_OVERDUE_GRACE` (15 min) in the past. Restarting the scheduler with `deploy.sh` restarts a dead chain ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).

## Stale-data check

`odds_lambda.health_monitor.collection_sources` lists the `odds_lambda.collection_calendar.CollectionSource` each collection job declares at module level: its name, the `raw_data["source"]` tag of its snapshots (`odds_core.snapshot_sources.SnapshotSource`; the Odds API writes none, so untagged snapshots are its), cadence, a fixture horizon per sport and whether it honours the overnight window. The check runs for each source whose job is in `odds_core.config.SchedulerConfig.bootstrap_jobs` and each sport the job covers (`odds_lambda.scheduling.jobs.resolve_target_sports`) and flags a source when all of these hold:

- `odds_lambda.collection_calendar.fixture_calendar` finds a fixture of the sport inside the source's horizon, so international breaks, off-seasons and the days before the first fixture raise nothing. EPL fixtures are SCHEDULED events; MLB fixtures are the games of the latest `odds_core.mlb_data_models.MlbProbablePitchers` fetch (every row of a fetch shares one `fetched_at`, and the scheduled job is the only writer) when that fetch is under 36 h old, with a priced game type, a known start time and a status other than Postponed, Cancelled or If Necessary. So the MLB calendar does not depend on the monitored collectors creating events. When no fetch is under 36 h old between `mlb_season_first_month` and `mlb_season_last_month` (March to November), `odds_lambda.health_monitor.HealthMonitor.check_mlb_calendar` sends the critical `mlb_calendar_stale` instead of letting the MLB checks go quiet.
- The source has written a snapshot for the sport before, the same no-history rule as heartbeats, so a job that was never enabled raises nothing (#384).
- The job's own failure alert is not a recent, unanswered one: `job_failure:<job>` newer than its heartbeat and newer than the larger of the threshold and the reminder window. While that holds, the job's critical covers the outage and no stale alert is added; an older failure with no heartbeat since does not suppress.
- `odds_lambda.collection_calendar.active_hours_between` the later of the last snapshot and the moment the next kickoff entered the horizon, and now, excluding the sport's overnight window when the job honours it (`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`), exceeds the largest of `stale_data_hours` (2), `stale_cadence_multiplier` (3) times the job's cadence interval at the tier of the next kickoff, and `stale_last_interval_multiplier` (1.5) times the interval at the tier the next kickoff had when the last snapshot was written. The last term allows for the wait the job chose then, which can be longer than the one that applies now.

A stale source alerts once and then repeats every `stale_reminder_minutes` (360) while it stays stale ([alerts](alerts.md)). The window is a rate limit, not a state: a source that recovers and goes stale again within 6 h of its last stale alert is muted until the window ends. The last-snapshot lookup takes the newest matching row through the `snapshot_time` index (`ORDER BY snapshot_time DESC LIMIT 1`), so a source with no recent rows scans back further.

| Source | Fixture horizon | Overnight window | Threshold (active hours) |
|---|---|---|---|
| OddsPortal | 48 h | yes | 2 h under 3 h to kickoff, 3 h at 3-12 h, 6 h beyond (cadence 0.5, 1, then 2 h) |
| Betfair Exchange | 72 h EPL, 6 h MLB | yes | 2 h under 12 h to kickoff, 6 h beyond (cadence 0.25, 0.5, then 2 h) |
| Odds API | 48 h EPL | no | 2 h under 3 h to kickoff, 9 h at 3-12 h, 36 h at 12-24 h, 72 h at 24-72 h (cadence 0.5, 3, 12, 24 h) |
| Polymarket odds | 5 days | yes | 2 h under 3 h to kickoff, then 3 times the tier step (0.25 to 2 h, floored) |

The thresholds are computed from each job's cadence, not configured. The Odds API horizon bounds the gap measured before a fixture is 12 h away, so that source is effectively checked only inside 12 h of kickoff. Betfair's MLB horizon is short because its markets are posted late, and its MLB name parsing is a separate defect.

## Betfair close check

`odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes` records which EPL events lack the close that [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) takes from the live Betfair job. FDUK can leave Betfair closing prices blank for whole matchdays (2026-04-20 to 2026-05-04), and then the live snapshot is the only close. An EPL event is missing its close when all of these hold:

- Its kick-off passed within `odds_core.config.AlertConfig.betfair_close_lookback_hours` (24; environment variable `BETFAIR_CLOSE_LOOKBACK_HOURS`, since `odds_core.config.AlertConfig` has no prefix).
- `odds_core.config.BetfairConfig.enabled` is true, `fetch-betfair-exchange` is in `odds_core.config.SchedulerConfig.bootstrap_jobs`, `soccer_epl` is among its target sports (`odds_lambda.scheduling.jobs.resolve_target_sports`), and Betfair has written an EPL snapshot before.
- Its status is not postponed or cancelled, and it has at least one live Betfair snapshot (`odds_core.snapshot_sources.SnapshotSource.BETFAIR_API`) in the 72 h before kick-off. A ghost, an unannounced postponement and a twin whose prices sit on the other event therefore raise nothing.
- No live Betfair snapshot within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` (30 min) before kick-off is both not in-play (`odds_core.betfair_close.is_in_play`) and priced for every outcome (`odds_core.betfair_close.complete_exchange_prices`). The retail-EV builder uses the same two tests for its close.

The missing events of a run are one critical `missing_betfair_close` listing every event, so a matchday cannot storm. Its `alert_history` context holds the event ids. A later run reports only events not named in an alert inside the lookback, so an event drops out of `issues_detected` once its alert was recorded; a run whose send was rate-limited (`odds_core.config.AlertConfig.alert_rate_limit_minutes`) records nothing and lists the events again next run. The check reads the snapshot's prices, not their quality, so a wide back book (ADR-0014) still counts.

The alert fires after kick-off, when the close cannot be collected. It is a coverage record for the retail-EV evaluation and does not catch a gap in time to fill it. Protection before kick-off is in the job's schedule: it books a run 5 min before each kickoff ([close run](../scheduling/decision-engine.md#close-run)), and a scheduler restart keeps that pending run ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)). An alert therefore has one of these causes:

- the close run failed;
- the scheduler was down at the close;
- the job's chain had ended ([Schedule chain check](#schedule-chain-check));
- the kickoff was added or moved after the job's last run booked its next one, so no close run was booked for it.

## Heartbeats

A job inside `odds_core.alerts.job_alert_context` inserts `heartbeat:<name>` on clean exit ([alerts](alerts.md#alert-history)). `odds_core.config.AlertConfig.heartbeat_expectations` holds the allowed hours per key; the table lists every job module under `odds_lambda.jobs` and whether it writes a heartbeat. How the mechanisms fit together is on [alerting-and-heartbeats](alerting-and-heartbeats.md).

| Job | Writes a heartbeat | Default expectation |
|---|---|---|
| fetch-oddsportal (per sport) | yes | 11 h EPL, 12 h MLB (overnight window plus polling interval) |
| fetch-oddsportal-results (per sport) | yes | 26 h (EPL, MLB) |
| daily-digest (EPL) | yes | none |
| check-health | yes | 2 h |
| update-status | yes | none |
| fetch-betfair-exchange (per sport) | yes | 10 h EPL, 11 h MLB (overnight window plus the 2 h interval) |
| fetch-polymarket-odds (per sport) | yes | 11 h EPL (overnight window plus polling interval, plus 1 h slack) |
| score-predictions | yes | none |
| fetch-mlb-probables | yes | 30 h |
| fetch-espn-fixtures | yes | none |
| fetch-odds, fetch-scores, fetch-polymarket, backfill-polymarket, agent-run | no: not wrapped in `odds_core.alerts.job_alert_context` (#438) | none |

## Failure handling

- If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `odds_lambda.health_monitor.HealthStatus` instead of raising. `check_health` therefore writes a heartbeat after a failed run, and it schedules after the work, so an exception that did escape would end the chain (#438).
- `odds_lambda.health_monitor.HealthMetrics` is returned in `odds_lambda.health_monitor.HealthStatus` and logged but never sent to Discord; only issue strings are alerted (#384).

## Alert path

`odds_lambda.health_monitor.HealthMonitor._send_alert` checks `alert_enabled`, then `odds_lambda.health_monitor.HealthMonitor._should_send_alert` (a count over `alert_history` using `odds_core.config.AlertConfig.alert_rate_limit_minutes`), calls `odds_core.alerts.AlertManager.alert`, then `odds_lambda.health_monitor.HealthMonitor._record_alert` with the injected `AsyncSession`. This duplicates `odds_core.alerts.check_rate_limit` and `odds_core.alerts.record_to_alert_history` because the monitor carries its own session and settings ([alerts](alerts.md#alert-history)).

## odds status

`odds_cli.commands.status` is the on-demand view of the same tables. It prints Rich tables only; there is no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, odds and snapshot totals, quota and 24 h success rate from `odds_lambda.storage.readers.OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota plus the last 10 `odds_core.models.FetchLog` rows (`odds_cli.commands.status._show_quota`). It reads the newest 10 rows of any source through `odds_lambda.storage.readers.OddsReader.get_fetch_logs` and shows nothing when the newest row has a null quota, as Betfair rows do (#441).
- `odds scheduler health` (not part of `odds status`): prints a green or red overall status, the backend name, the passed and failed checks and any details from `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check`, and exits 1 when unhealthy or on error.
- `odds status events --days N --team T` (default 7 days): recent events with status and scores. It has no sport filter.
- `status show` takes the latest fetch row from all sources and computes the 24 h success rate over all of them, so Betfair rows can distort both. Its quota is the newest non-null `api_quota_remaining`, unlike `status quota` ([odds-api-quota](../collection/odds-api-quota.md#where-the-views-disagree), [fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
