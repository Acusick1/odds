---
type: reference
title: Health monitor, check-health job and odds status
description: HealthMonitor, the check-health job's six checks, heartbeat purge, its own alert path, and what odds status show, quota and events print and read.
sources:
  - odds_lambda.health_monitor.HealthMonitor
  - odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_stale_sources
  - odds_lambda.health_monitor.StaleSource
  - odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes
  - odds_lambda.health_monitor.MissingBetfairClose
  - odds_core.paper_trade_models.PaperTrade
  - odds_core.betfair_close.LIVE_CLOSE_WINDOW
  - odds_core.betfair_close.live_close_quotes
  - odds_core.betfair_close.event_close
  - odds_lambda.health_monitor.collection_sources
  - odds_lambda.collection_calendar.CollectionSource
  - odds_lambda.collection_calendar.fixture_calendar
  - odds_lambda.collection_calendar.active_hours_between
  - odds_lambda.collection_calendar.latest_snapshot_time
  - odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats
  - odds_lambda.health_monitor.HealthMonitor.check_api_quota
  - odds_lambda.health_monitor.HealthMonitor.check_event_identity
  - odds_lambda.health_monitor.HealthMonitor.find_mlb_events_without_game_pk
  - odds_lambda.health_monitor.HealthMonitor.check_unsettled_events
  - odds_lambda.storage.unsettled_events.find_unsettled_events
  - odds_lambda.health_monitor.HealthMonitor.check_schedule_chains
  - odds_lambda.health_monitor.DeadChain
  - odds_lambda.health_monitor.SCHEDULE_OVERDUE_GRACE
  - odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_outstanding_job_names
  - odds_lambda.health_monitor.check_system_health
  - odds_lambda.health_monitor.HealthMetrics
  - odds_lambda.health_monitor.HealthStatus
  - odds_lambda.jobs.check_health.main
  - odds_core.alerts.ping_dead_mans_switch
  - odds_core.config.AlertConfig.dead_mans_switch_url
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

`odds_lambda.jobs.check_health.main` runs `odds_lambda.health_monitor.check_system_health` inside `job_alert_context("check-health")`, pings the external [dead-man's switch](#dead-mans-switch), and then books itself 60 minutes ahead through a raw backend call (no pre-schedule; a failed booking sends a `scheduling_failure:check-health` critical; see [self-scheduling](../scheduling/self-scheduling.md)). It is global and in the default production `odds_core.config.SchedulerConfig.bootstrap_jobs` (the same list that holds the per-sport jobs), so it runs on the production host against the production database, where every job's heartbeat lands ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). `odds_lambda.health_monitor.check_system_health` collects `odds_lambda.health_monitor.HealthMetrics`, then runs, in order:

1. `odds_lambda.health_monitor.HealthMonitor.check_stale_sources`: per collection source, sport and market, active hours without a new snapshot of that market versus a threshold derived from the source's cadence, only while the database has fixtures the source should price. Each stale market sends its own critical `stale_data:<source>:<sport>:<market>` ([Stale-data check](#stale-data-check)). It is followed by the MLB schedule check and by `odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes`, one critical `missing_betfair_close` listing EPL events whose Betfair close was not collected ([Betfair close check](#betfair-close-check)).
2. `odds_lambda.health_monitor.HealthMonitor.check_api_quota`: for each Odds API key, `api_quota_remaining` of the latest `fetch_logs` row of the sports that spend it (`odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key`; rows over 31 days old or without `api_quota_used` are ignored, so an off-season key raises nothing), as a fraction of the key's plan, `api_quota_used` plus `api_quota_remaining` of that row; below 0.1 alert `quota_critical:key<N>`, below 0.2 `quota_low:key<N>`, N the key's index ([keys](../collection/odds-api-client.md#retry-and-keys)). It is the only quota alert; it fires when the condition starts and reminds every `stale_reminder_minutes` ([odds-api-quota](../collection/odds-api-quota.md)).
3. `odds_lambda.health_monitor.HealthMonitor.check_data_quality`: `error` plus `critical` `odds_core.models.DataQualityLog` rows in 24 h (each query capped at 1000) versus `data_quality_error_threshold` (10). The validator itself only emits `warning` and `error`, so `critical` never occurs ([odds-validator](../storage/odds-validator.md)).
4. `odds_lambda.health_monitor.HealthMonitor.check_job_heartbeats`: compares the newest `heartbeat:<job>` row per key in `odds_core.config.AlertConfig.heartbeat_expectations` to its allowed hours. The default keys and allowed hours are in [Heartbeats](#heartbeats). A job with no history is not alerted (#384).
5. `odds_lambda.health_monitor.HealthMonitor.check_event_identity`: per sport, matches stored as several events, EPL fixtures stored twice in a season, and MLB events (by integer id) over 24 h old holding no `gamePk` (`mlb_stats_api` external id) and no `missing_mlb_game_pk` acknowledgement (`odds_lambda.health_monitor.HealthMonitor.find_mlb_events_without_game_pk`), each its own warning (`duplicate_events:<sport>`, `repeated_fixtures:<sport>`, `missing_game_pk:baseball_mlb`). Nothing is merged; `odds events merge-twins --sport <sport> --apply` clears twins, and `odds events assign-game-pks --apply` the missing `gamePk`s ([Event merge](../storage/event-merge.md#detection)).
6. `odds_lambda.health_monitor.HealthMonitor.check_schedule_chains`: dead self-scheduling chains, each its own critical `dead_schedule_chain:<job>` ([Schedule chain check](#schedule-chain-check)).
7. `odds_lambda.health_monitor.HealthMonitor.check_unsettled_events`: per sport in `odds_lambda.health_monitor.UNSETTLED_SPORTS` (EPL, MLB and NBA), current-season events still SCHEDULED or LIVE more than 24 h after kick-off (`odds_lambda.storage.unsettled_events.find_unsettled_events`), one warning `unsettled_events:<sport>` naming them, repeated once a day. The season starts on 1 August for EPL and on 1 January for MLB (`odds_lambda.storage.unsettled_events.season_start`). POSTPONED and CANCELLED events are not pending, so a postponed MLB game waiting on its replay stays quiet, and neither are UNMATCHED ones, the MLB events with no game in MLB's schedule, which no results job can settle ([results jobs](../collection/results-jobs.md), [no game in the schedule](../storage/event-resolver.md#no-game-in-the-schedule)).

Then `odds_lambda.health_monitor.HealthMonitor.purge_old_heartbeats` runs on every health check, and it deletes `heartbeat:%` rows older than `heartbeat_retention_days` (7) and is the only retention sweep in the system.

## Dead-man's switch

Every check above runs on the production host, so none of them can report that the host or its scheduler has stopped. An external monitor covers that: it expects a ping every hour and alerts by its own channels (email or Discord) when pings stop.

- After a run leaves `job_alert_context("check-health")` without raising, `odds_lambda.jobs.check_health.main` calls `odds_core.alerts.ping_dead_mans_switch`, which sends a GET to `odds_core.config.AlertConfig.dead_mans_switch_url` (`DEAD_MANS_SWITCH_URL`) with a 10 s timeout. A run that found issues still pings, since those reach Discord already and the monitor answers only whether the host is running. A run that raised does not ping.
- It sends nothing when the URL is unset or empty, or under `SCHEDULER_DRY_RUN` (the deploy smoke is not a scheduled run).
- A failed ping (error status, timeout, no connection) is logged as `dead_mans_switch_ping_failed` or `dead_mans_switch_ping_error` and never fails the job; the missed ping is what the monitor alerts on.

The monitor is a [healthchecks.io](https://healthchecks.io/docs/) check (the free tier allows 20) with a period of 1 h and a grace time of 1 h, so a stopped scheduler or host alerts within about 2 h of its last ping. Its ping URL goes in the production host's `.env` ([add a setting](../foundations/howto-add-setting.md)); the check's period, grace and notification channels live in the healthchecks.io account, not in the repo.

## Schedule chain check

`odds_lambda.health_monitor.HealthMonitor.check_schedule_chains` catches a self-scheduling job whose chain has ended, before its data goes stale. It runs only when `odds_lambda.health_monitor.check_system_health` is given the local scheduler store (`SCHEDULER_BACKEND=local`); a `odds_lambda.health_monitor.HealthMonitor` built without one skips it. It covers every compound job that `odds_core.config.SchedulerConfig.bootstrap_jobs` expands to (`run-candidates` included), except:

- cron jobs, which APScheduler re-fires on its own;
- jobs that return before booking a run while disabled (Betfair while `odds_core.config.BetfairConfig.enabled` is false, `fetch-polymarket-odds` while Polymarket is disabled);
- jobs with a released, unfinished APScheduler job under their schedule id (`odds_lambda.scheduling.backends.local.LocalSchedulerBackend.get_outstanding_job_names`). A job that books its next run only after its work, such as check-health itself, has no schedule while it runs. A run cannot stay outstanding past `odds_lambda.scheduling.runner.JOB_TIMEOUT`, so a hung run is cancelled instead of hiding its chain here. A chain that keeps failing is re-booked every 15 minutes and so stays booked; the `job_chain_rebooked:<job>` warning is what shows it ([Alerts](alerts.md#alert-history)).

A covered job is dead when it has no stored schedule, or when its next run is more than `odds_lambda.health_monitor.SCHEDULE_OVERDUE_GRACE` (15 min) in the past. A run of `deploy.sh` always restarts the scheduler, which restarts a dead chain; its smoke step skips jobs that spend Odds API quota ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).

## Stale-data check

`odds_lambda.health_monitor.collection_sources` lists the `odds_lambda.collection_calendar.CollectionSource` each collection job declares at module level: its name, the `source` of its snapshots (`odds_core.vocabulary.Source`), cadence, a fixture horizon per sport, the markets it writes for each of those sports, and whether it honours the overnight window. The check runs for each source whose job is in `odds_core.config.SchedulerConfig.bootstrap_jobs` (a job that stores no snapshots, such as `run-candidates`, declares no source), each sport the job covers (`odds_lambda.scheduling.jobs.resolve_target_sports`) and each market the source declares for that sport, and flags a source's market when all of these hold:

- `odds_lambda.collection_calendar.fixture_calendar` finds a fixture of the sport inside the source's horizon (leaving out any the caller passes as `excluding`), so international breaks, off-seasons and the days before the first fixture raise nothing. EPL fixtures are SCHEDULED events; MLB fixtures are the games of the latest `odds_core.mlb_data_models.MlbProbablePitchers` fetch (every row of a fetch shares one `fetched_at`, and the scheduled job is the only writer) when that fetch is under 36 h old, with a priced game type, a known start time and a status other than Postponed, Cancelled or If Necessary. So the MLB calendar does not depend on the monitored collectors creating events. A fetch that finds no games writes no rows, so an MLB off-season, like any other, raises nothing; but when SCHEDULED MLB events lie within the next 72 h the job itself fails on the empty fetch ([MLB probable pitchers](../collection/mlb-probable-pitchers.md#flow)), so an in-season empty schedule does not silence the checks unnoticed. A probables job that stops completing shows as a missing `fetch-mlb-probables` heartbeat (30 h) and not as a calendar alert.
- The source has written an `observed` snapshot of the market for the sport before (`odds_lambda.collection_calendar.latest_snapshot_time`), the same no-history rule as heartbeats, so a job that was never enabled, or a market it has not yet written, raises nothing (#384). Backfilled history (time basis `upper_bound` or `source_stamped`) does not count.
- The job's own failure alert is not a recent, unanswered one: `job_failure:<job>` newer than its heartbeat (`odds_core.alerts.job_outage_alerted_at`) and at most two reminder intervals old, since a job writes that row only once per reminder interval during an outage. While that holds, the job's critical covers the outage and no stale alert is added; an older failure with no heartbeat since does not suppress.
- `odds_lambda.collection_calendar.active_hours_between` the later of the market's last snapshot and the moment the next kickoff entered the horizon, and now, excluding the sport's overnight window when the job honours it (`odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`), exceeds the largest of `stale_data_hours` (2), `stale_cadence_multiplier` (3) times the job's cadence interval at the tier of the next kickoff, and `stale_last_interval_multiplier` (1.5) times the interval at the tier the next kickoff had when the last snapshot was written, both tiers from `odds_lambda.tier_utils.calculate_tier` (a kickoff exactly 3 h away is closing; the tier at the last snapshot is computed from that snapshot's hours to the kickoff). The last term allows for the wait the job chose then, which can be longer than the one that applies now.

Each market is checked on its own, so a source that keeps writing one market cannot hide another it has stopped writing: on 2026-10-19 OddsPortal's EPL `totals` stayed fresh while its `1x2` stopped 4h47m before a kickoff (#666). A stale market alerts once and then repeats every `stale_reminder_minutes` (360) while it stays stale, like every other warning of this monitor ([alerts](alerts.md#alert-history)). The interval is measured from the last alert, not from the condition: a market that recovers and goes stale again within 6 h of its last stale alert is muted until the interval ends. The last-snapshot lookup takes the newest row of the source and market with time basis `observed` through the `snapshot_time` index (`ORDER BY snapshot_time DESC LIMIT 1`), so a source with no recent rows scans back further.

| Source | Fixture horizon | Markets | Overnight window | Threshold (active hours) |
|---|---|---|---|---|
| OddsPortal | 48 h | EPL `1x2`, `totals`; MLB `h2h`; NBA `h2h`, `totals`, `spreads` (from each `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS` entry's feed markets) | yes | 2 h under 3 h to kickoff, 3 h at 3-12 h, 6 h beyond (cadence 0.5, 1, then 2 h) |
| Betfair Exchange | 72 h EPL, 6 h MLB and NBA | EPL `1x2`, `totals`; MLB and NBA `h2h` (from `odds_lambda.betfair.constants.SPORT_CONFIG`) | yes | 2 h under 12 h to kickoff, 6 h beyond (cadence 0.25, 0.5, then 2 h) |
| Odds API | 48 h EPL (NBA is not checked: its schedule is sparse, so the EPL cadence's threshold would fire between runs) | EPL `1x2`, `spreads`, `totals` | no | 2 h under 3 h to kickoff, 9 h at 3-12 h, 36 h at 12-24 h, 72 h at 24-72 h (cadence 0.5, 3, 12, 24 h) |
| Polymarket odds | 5 days | EPL `1x2`; NBA `h2h` (from `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`) | yes | 2 h under 3 h to kickoff, then 3 times the tier step (0.25 to 2 h, floored) |

A collector that is dormant (no fixture within 28 days, [dormancy](../scheduling/decision-engine.md#dormancy)) never meets this check: every source horizon above is shorter, and the chain wakes when a fixture enters the 28 days.

The thresholds are computed from each job's cadence, not configured. The Odds API horizon bounds the gap measured before a fixture is 12 h away, so that source is effectively checked only inside 12 h of kickoff. Betfair's MLB horizon is short because its markets are posted late, and its MLB name parsing is a separate defect.

## Betfair close check

`odds_lambda.health_monitor.HealthMonitor.check_missing_betfair_closes` records which EPL events lack the close, on the three-way market and on the line of each totals paper trade, that [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) takes from the live Betfair job. FDUK can leave Betfair closing prices blank for whole matchdays (2026-04-20 to 2026-05-04), and then the live snapshot is the only close. An EPL event is missing its close when all of these hold:

- Its kick-off passed within `odds_core.config.AlertConfig.betfair_close_lookback_hours` (default 24; environment variable `BETFAIR_CLOSE_LOOKBACK_HOURS`, since `odds_core.config.AlertConfig` has no prefix).
- `odds_core.config.BetfairConfig.enabled` is true, `fetch-betfair-exchange` is in the default or configured `odds_core.config.SchedulerConfig.bootstrap_jobs`, `soccer_epl` is among its target sports (`odds_lambda.scheduling.jobs.resolve_target_sports`), and Betfair has written an EPL snapshot before (a snapshot with source `betfair`).
- Its status is not postponed or cancelled, and it has at least one live Betfair snapshot (source `betfair`) in the 72 h before kick-off. A ghost, an unannounced postponement and a twin whose prices sit on the other event therefore raise nothing.
- It has no live close: `odds_core.betfair_close.event_close` over the Betfair Exchange backs of its live Betfair snapshots within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` (30 min) before kick-off (`odds_core.betfair_close.live_close_quotes`) finds none. That is the close the [candidate report](../evaluation/candidate-registry.md#odds-candidates-report) scores against: a snapshot that is not in-play, prices every outcome of the market and has a back overround of at most `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND` ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)).
- It has no live close on the line of a totals paper trade (`odds_core.paper_trade_models.PaperTrade`, market `totals`) placed on it: `odds_core.betfair_close.event_close` with that `line`, over the `totals` quotes from `odds_core.betfair_close.live_close_quotes`. An event is missing when it lacks either close, and the alert names which.

The missing events of a run are one critical `missing_betfair_close` listing every event, so a matchday cannot storm. Its `alert_history` context holds the integer event ids (`event_ids`). A later run reports only events not named in an alert inside the lookback, so an event drops out of `issues_detected` once its alert was recorded; a run whose send failed, or whose alerts are off, records nothing and lists the events again next run. A snapshot whose back book is wider than the cap is not a close, so an event whose only window snapshots are wide is reported.

The alert fires after kick-off, when the close cannot be collected. It is a coverage record for the retail-EV evaluation and does not catch a gap in time to fill it. Protection before kick-off is in the job's schedule: it books a run 5 min before each kickoff ([lead runs](../scheduling/decision-engine.md#lead-runs)), and a scheduler restart keeps that pending run ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)). An alert therefore has one of these causes:

- the close run failed;
- the scheduler was down at the close;
- the job's chain had ended ([Schedule chain check](#schedule-chain-check));
- the kickoff was added or moved after the job's last run booked its next one, so no close run was booked for it.

## Heartbeats

A job inside [`odds_core.alerts.job_alert_context`](alerts.md#delivery) inserts `heartbeat:<name>` on clean exit and none when the job raises ([alerts](alerts.md#alert-history)). A dormant `fetch-oddsportal`, `fetch-betfair-exchange` or `fetch-polymarket-odds` run enters the context without scraping, so its 6 h check-ins keep the heartbeat inside the expectations below. `odds_core.config.AlertConfig.heartbeat_expectations` holds the allowed hours per key; the table lists every job module under `odds_lambda.jobs` and whether it writes a heartbeat. How the mechanisms fit together is on [alerting-and-heartbeats](alerting-and-heartbeats.md).

| Job | Writes a heartbeat | Default expectation |
|---|---|---|
| fetch-oddsportal (per sport) | yes | 11 h EPL, 12 h MLB, 11 h NBA (overnight window plus polling interval) |
| fetch-oddsportal-results (EPL) | yes | 26 h |
| fetch-mlb-results | yes | 26 h |
| check-health | yes | 2 h |
| update-status | yes | none |
| fetch-betfair-exchange (per sport) | yes | 10 h EPL, 11 h MLB, 11 h NBA (overnight window plus the 2 h interval) |
| fetch-polymarket-odds (per sport) | yes | 11 h EPL and NBA (overnight window plus polling interval, plus 1 h slack) |
| fetch-mlb-probables | yes | 30 h |
| fetch-injuries (NBA) | yes | 15 h (up to 14 h between runs over the NBA overnight window, since lead runs restart the 6 h cadence, plus 1 h slack; a dormant run still writes it) |
| fetch-espn-fixtures (per sport: bare key for EPL, `-nba` for NBA) | yes | 26 h (daily cron plus slack) |
| fetch-odds (EPL) | yes | 49 h (48 h opening-tier interval plus 1 h; no overnight pause) |
| fetch-odds (NBA) | yes | 35 h (24 h interval pushed to 10:00 UTC by the 00:00-10:00 quiet window, about 33 h, plus slack) |
| fetch-scores (EPL) | yes | 13 h (12 h idle interval plus 1 h; no overnight pause) |
| agent-run | no: not wrapped in `odds_core.alerts.job_alert_context` (#438) | none |

## Failure handling

- If any step raises, the monitor sends `health_check_failure` (critical) and returns an unhealthy `odds_lambda.health_monitor.HealthStatus` instead of raising. `check_health` therefore writes a heartbeat after a failed run, and it schedules after the work, so an exception that did escape would end the chain (#438).
- `odds_lambda.health_monitor.HealthMetrics` is returned in `odds_lambda.health_monitor.HealthStatus` and logged but never sent to Discord; only issue strings are alerted (#384).

## Alert path

`odds_lambda.health_monitor.HealthMonitor._send_alert` checks `odds_core.config.Settings.alerts_deliverable` (false when alerts are off or `SCHEDULER_DRY_RUN` is on, so a deploy smoke neither posts nor writes a dedup row), then hands off to `odds_core.alerts.send_recorded_alert` with the injected `AsyncSession`, the alert's context and an interval of `odds_core.config.AlertConfig.stale_reminder_minutes`; `odds_lambda.health_monitor.HealthMonitor.check_unsettled_events` passes 24 h and `odds_lambda.health_monitor.HealthMonitor.check_system_health` passes no interval for `missing_betfair_close`, whose per-event dedup is the listing above ([alerts](alerts.md#alert-history)).

## odds status

`odds_cli.commands.status` is the on-demand view of the same tables. It prints Rich tables only; there is no machine-readable output.

- `odds status show [--verbose]`: latest fetch (minutes ago, tick or cross), event counts by status, price-row and snapshot totals (snapshots counted as distinct event and snapshot-time pairs), one quota row per key from `odds_lambda.storage.readers.OddsReader.get_database_stats`; `--verbose` adds data-quality rows from the last 24 h (limit 10).
- `odds status quota`: latest quota and plan size per key plus the last 10 `odds_core.models.FetchLog` rows that have a quota (`odds_cli.commands.status._show_quota`). Rows with a null quota are skipped ([odds-api-quota](../collection/odds-api-quota.md#where-quota-is-recorded-alerted-and-checked)); when none has a quota it prints `No fetch log records an API quota`.
- `odds scheduler health` (not part of `odds status`): prints a green or red overall status, the backend name, the passed and failed checks and any details from `odds_lambda.scheduling.backends.base.SchedulerBackend.health_check`, and exits 1 when unhealthy or on error.
- `odds status events --days N --team T` (default 7 days): recent events with integer id, status and scores. It has no sport filter.
- `status show` takes the latest fetch row from all sources, so Betfair rows can distort it. It has no success rate: `fetch_logs` has no source column and a failed Odds API call writes no row, so a rate over it could not show an Odds API outage. Its quota is read as in `status quota` ([odds-api-quota](../collection/odds-api-quota.md#where-quota-is-recorded-alerted-and-checked), [fetch logs](../collection/odds-ingestion-service.md#fetch-logs)).

## Related

- [Alerts and the alert_history table](alerts.md)
- [How failures are detected](alerting-and-heartbeats.md)
- [Reporting surfaces](reporting-surfaces.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
